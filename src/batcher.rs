//! Concurrency core: bounded request queue (backpressure) + a scheduler thread
//! that groups requests into batches, plus a **native-worker pool** for
//! data-parallel inference (NATIVE_RUST_PLAN §3.5).
//!
//! Batching (unchanged from the proven design): the scheduler blocks for the
//! first job, sweeps the backlog into a batch, optionally lingers
//! (`batch_timeout_ms`) to coalesce near-simultaneous arrivals, length-buckets
//! by padded-cell budget, and runs.
//!
//! Data-parallel (native backend — multiple Engines share ONE `Arc<Weights>`
//! copy): when a batch is large enough, its length-sorted unique rows are split
//! into contiguous groups (length-homogeneous by construction) and dispatched
//! to pool workers; the scheduler runs group 0 itself and waits for acks.
//! Below the threshold the scheduler runs alone — the low-latency path
//! performs zero cross-thread synchronization. Because CT4 is bit-exactly
//! batch-invariant, ANY row partitioning yields identical results.
//!
//! Dedup: identical strings are common in name workloads (11 % of a generated
//! 10k sample); identical inputs embed identically by determinism, so each
//! batch embeds each distinct text once and replicates the vector.
//!
//! Backpressure: the queue is bounded; a full queue is shed with `Rejected`
//! (→ HTTP 503), so memory stays flat under overload.

use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TryRecvError, TrySendError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::oneshot;

use crate::engine::Engine;
use crate::metrics::Metrics;

/// Result sent back per item: the unit vector and its token count (for `usage`).
pub type JobResult = Result<(Vec<f32>, usize), String>;

/// One queued embedding request (a single input string).
pub struct Job {
    pub text: String,
    pub enqueued: Instant,
    pub resp: oneshot::Sender<JobResult>,
}

#[derive(Debug)]
pub struct Rejected;

/// Handle held by the HTTP layer to submit work with backpressure.
#[derive(Clone)]
pub struct Submitter {
    tx: SyncSender<Job>,
    queue_depth: Arc<AtomicI64>,
    metrics: Arc<Metrics>,
}

impl Submitter {
    /// Non-blocking submit. Returns a receiver for the result, or `Rejected`
    /// when the bounded queue is full (→ HTTP 503).
    pub fn submit(&self, text: String) -> Result<oneshot::Receiver<JobResult>, Rejected> {
        let (resp, rx) = oneshot::channel();
        let job = Job { text, enqueued: Instant::now(), resp };
        match self.tx.try_send(job) {
            Ok(()) => {
                self.queue_depth.fetch_add(1, Ordering::Relaxed);
                self.metrics.queue_depth.inc();
                Ok(rx)
            }
            Err(_) => {
                self.metrics.rejected_total.inc();
                Err(Rejected)
            }
        }
    }
}

pub struct BatchConfig {
    pub max_batch: usize,
    pub batch_timeout_ms: u64,
    pub queue_capacity: usize,
    /// Padded-cell budget per inference sub-batch (size × max_len). Keeps a long
    /// outlier from forcing the short names to pad up to it.
    pub pad_cell_budget: usize,
}

/// A unit of inference work for a pool worker (or the scheduler itself):
/// length-similar UNIQUE rows + the job groups that share them (dedup:
/// identical texts embed once; the vector is replicated to its group).
struct Task {
    ids: Vec<Vec<i64>>,
    /// Parallel to `ids`: the jobs awaiting that row's vector.
    groups: Vec<Vec<Job>>,
    done: Option<std::sync::mpsc::Sender<()>>,
}

/// Wire up the bounded queue and spawn the scheduler (+ pool workers).
/// `engines[0]` belongs to the scheduler; `engines[1..]` to pool workers.
/// A single engine ⇒ no pool, pure single-thread path (the ORT mode).
pub fn start(engines: Vec<Engine>, cfg: BatchConfig, metrics: Arc<Metrics>) -> Submitter {
    assert!(!engines.is_empty(), "at least one engine required");
    let mut engines = engines.into_iter();
    let scheduler_engine = engines.next().unwrap();

    let (tx, rx) = sync_channel::<Job>(cfg.queue_capacity);
    let queue_depth = Arc::new(AtomicI64::new(0));

    // Pool workers: each owns an Engine (native: shares Arc<Weights>) and
    // fulfills job oneshots directly; `done` acks the scheduler.
    let mut pool: Vec<SyncSender<Task>> = Vec::new();
    for engine in engines {
        let (wtx, wrx) = sync_channel::<Task>(4);
        let m = metrics.clone();
        std::thread::Builder::new()
            .name("ct4-pool".into())
            .spawn(move || pool_worker(engine, wrx, m))
            .expect("spawn pool worker");
        pool.push(wtx);
    }

    {
        let metrics = metrics.clone();
        let queue_depth = queue_depth.clone();
        std::thread::Builder::new()
            .name("ct4-infer".into())
            .spawn(move || scheduler_loop(scheduler_engine, rx, metrics, queue_depth, cfg, pool))
            .expect("spawn inference scheduler");
    }
    Submitter { tx, queue_depth, metrics }
}

fn pool_worker(engine: Engine, rx: Receiver<Task>, metrics: Arc<Metrics>) {
    loop {
        let task = match rx.recv() {
            Ok(t) => t,
            Err(_) => break, // scheduler dropped senders → shutdown
        };
        run_task(&engine, task, &metrics);
    }
}

fn run_task(engine: &Engine, task: Task, metrics: &Metrics) {
    let Task { ids, groups, done } = task;
    debug_assert_eq!(ids.len(), groups.len());
    let token_counts: Vec<usize> = ids.iter().map(|v| v.len()).collect();
    match engine.infer_ids(&ids) {
        Ok((vecs, timing)) => {
            metrics.lat_inference_s.observe(timing.inference_s);
            // Replicate each unique row's vector across its whole job group.
            for ((jobs, v), tokens) in groups.into_iter().zip(vecs.into_iter()).zip(token_counts.into_iter()) {
                for job in jobs {
                    let _ = job.resp.send(Ok((v.clone(), tokens)));
                }
            }
        }
        Err(e) => {
            let msg = format!("inference: {e}");
            for jobs in groups {
                for job in jobs {
                    metrics.errors_total.inc();
                    let _ = job.resp.send(Err(msg.clone()));
                }
            }
        }
    }
    if let Some(d) = done {
        let _ = d.send(());
    }
}

fn scheduler_loop(
    engine: Engine,
    rx: Receiver<Job>,
    metrics: Arc<Metrics>,
    queue_depth: Arc<AtomicI64>,
    cfg: BatchConfig,
    pool: Vec<SyncSender<Task>>,
) {
    let timeout = Duration::from_millis(cfg.batch_timeout_ms);
    loop {
        // Block for the first job of the next batch.
        let first = match rx.recv() {
            Ok(j) => j,
            Err(_) => break, // all senders dropped → shutdown (pool senders drop too)
        };
        let mut batch = Vec::with_capacity(cfg.max_batch);
        batch.push(first);

        // Opportunistically sweep up everything already queued (the backlog that
        // built up while the previous batch was running).
        while batch.len() < cfg.max_batch {
            match rx.try_recv() {
                Ok(j) => batch.push(j),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }
        // Optional linger: at low load the queue is empty, so wait briefly to
        // coalesce near-simultaneous arrivals (skipped entirely if timeout==0).
        if !timeout.is_zero() && batch.len() < cfg.max_batch {
            let deadline = Instant::now() + timeout;
            while batch.len() < cfg.max_batch {
                let now = Instant::now();
                if now >= deadline {
                    break;
                }
                match rx.recv_timeout(deadline - now) {
                    Ok(j) => batch.push(j),
                    Err(_) => break,
                }
            }
        }

        let n = batch.len() as i64;
        queue_depth.fetch_sub(n, Ordering::Relaxed);
        metrics.queue_depth.sub(n);
        metrics.batch_size.observe(batch.len() as f64);
        // queue-wait per job, observed once the scheduler picks the batch up.
        for j in &batch {
            metrics.lat_queue_wait_s.observe(j.enqueued.elapsed().as_secs_f64());
        }
        process_batch(&engine, batch, &metrics, cfg.max_batch, cfg.pad_cell_budget, &pool);
    }
}

/// Rows-per-worker minimum before data-parallel dispatch pays for the
/// cross-thread handoff (channel + wakeup ≈ tens of µs).
const MIN_ROWS_PER_WORKER: usize = 4;

/// Take the job groups of the given unique-row indices out of `groups`.
fn take_groups(indices: &[usize], groups: &mut [Vec<Job>]) -> Vec<Vec<Job>> {
    indices.iter().map(|&i| std::mem::take(&mut groups[i])).collect()
}

fn process_batch(
    engine: &Engine,
    batch: Vec<Job>,
    metrics: &Metrics,
    max_batch: usize,
    pad_cell_budget: usize,
    pool: &[SyncSender<Task>],
) {
    // ---- dedup: embed each distinct text once ----
    let mut unique: Vec<String> = Vec::with_capacity(batch.len());
    let mut index_of: HashMap<String, usize> = HashMap::with_capacity(batch.len());
    let mut groups: Vec<Vec<Job>> = Vec::with_capacity(batch.len());
    for job in batch {
        let text = job.text.clone();
        match index_of.get(&text) {
            Some(&u) => groups[u].push(job),
            None => {
                let u = unique.len();
                index_of.insert(text.clone(), u);
                unique.push(text);
                groups.push(vec![job]);
            }
        }
    }
    let n_unique = unique.len();

    let tok_t0 = Instant::now();
    let ids = match engine.tokenizer().encode_ids(&unique) {
        Ok(v) => v,
        Err(e) => {
            let msg = format!("tokenize: {e}");
            for g in groups {
                for job in g {
                    metrics.errors_total.inc();
                    let _ = job.resp.send(Err(msg.clone()));
                }
            }
            return;
        }
    };
    metrics.lat_tokenize_s.observe(tok_t0.elapsed().as_secs_f64());

    // Length-bucket: sort unique rows by token length; chunks by padded-cell
    // budget so a long outlier runs apart from the short names.
    let mut order: Vec<usize> = (0..n_unique).collect();
    order.sort_by_key(|&i| ids[i].len());
    let chunks = chunk_by_budget(&order, &ids, max_batch, pad_cell_budget);

    // ---- dispatch or run locally ----
    let workers = pool.len() + 1;
    let dispatch = !pool.is_empty() && n_unique >= MIN_ROWS_PER_WORKER * workers;
    if dispatch {
        // Contiguous slices of the length-sorted order → length-homogeneous
        // groups; batch-invariance makes the split exact.
        let per = n_unique.div_ceil(workers);
        let (done_tx, done_rx) = std::sync::mpsc::channel::<()>();
        let mut dispatched = 0usize;
        for (wi, group) in order.chunks(per).enumerate() {
            let task = Task {
                ids: group.iter().map(|&i| ids[i].clone()).collect(),
                groups: take_groups(group, &mut groups),
                done: if wi == 0 { None } else { Some(done_tx.clone()) },
            };
            if wi == 0 {
                run_task(engine, task, metrics);
            } else {
                let w = &pool[(wi - 1) % pool.len()];
                match w.try_send(task) {
                    Ok(()) => dispatched += 1,
                    Err(TrySendError::Full(task)) => {
                        // Worker saturated — run it here rather than block.
                        run_task(engine, task, metrics);
                    }
                    Err(TrySendError::Disconnected(task)) => run_task(engine, task, metrics),
                }
            }
        }
        for _ in 0..dispatched {
            let _ = done_rx.recv_timeout(Duration::from_secs(60));
        }
    } else {
        for chunk in chunks {
            let task = Task {
                ids: chunk.iter().map(|&i| ids[i].clone()).collect(),
                groups: take_groups(&chunk, &mut groups),
                done: None,
            };
            run_task(engine, task, metrics);
        }
    }
}

/// Greedy length-bucketing over length-sorted indices: start a new sub-batch
/// when adding the next item would exceed `max_batch` or push the padded-cell
/// count (`size × max_len`) past `budget`.
fn chunk_by_budget(order: &[usize], ids: &[Vec<i64>], max_batch: usize, budget: usize) -> Vec<Vec<usize>> {
    let mut chunks: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    let mut cur_max = 0usize;
    for &i in order {
        let len = ids[i].len().max(1);
        let new_max = cur_max.max(len);
        let cells = new_max * (cur.len() + 1);
        if !cur.is_empty() && (cur.len() >= max_batch || cells > budget) {
            chunks.push(std::mem::take(&mut cur));
            cur_max = 0;
        }
        cur_max = cur_max.max(len);
        cur.push(i);
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    chunks
}
