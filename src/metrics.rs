//! Prometheus metrics: request/error counts, in-flight & queue-depth gauges,
//! batch-size histogram, per-stage latency histograms, and an RSS gauge
//! refreshed on scrape from /proc/self/statm.

use prometheus::{
    Histogram, HistogramOpts, IntCounter, IntGauge, Registry, TextEncoder,
};

pub struct Metrics {
    pub registry: Registry,
    pub requests_total: IntCounter,
    pub items_total: IntCounter,
    pub errors_total: IntCounter,
    pub rejected_total: IntCounter, // 503 backpressure
    pub in_flight: IntGauge,
    pub queue_depth: IntGauge,
    pub batch_size: Histogram,
    pub lat_tokenize_s: Histogram,
    pub lat_queue_wait_s: Histogram,
    pub lat_inference_s: Histogram,
    pub lat_total_s: Histogram,
    pub rss_bytes: IntGauge,
}

fn latency_buckets() -> Vec<f64> {
    // 0.5 ms … 2 s
    vec![
        0.0005, 0.001, 0.002, 0.003, 0.005, 0.008, 0.012, 0.020, 0.035, 0.060,
        0.100, 0.200, 0.500, 1.0, 2.0,
    ]
}

impl Metrics {
    pub fn new() -> anyhow::Result<Self> {
        let registry = Registry::new();
        macro_rules! ctr {
            ($name:expr, $help:expr) => {{
                let c = IntCounter::new($name, $help)?;
                registry.register(Box::new(c.clone()))?;
                c
            }};
        }
        macro_rules! gauge {
            ($name:expr, $help:expr) => {{
                let g = IntGauge::new($name, $help)?;
                registry.register(Box::new(g.clone()))?;
                g
            }};
        }
        macro_rules! hist {
            ($name:expr, $help:expr, $buckets:expr) => {{
                let h = Histogram::with_opts(HistogramOpts::new($name, $help).buckets($buckets))?;
                registry.register(Box::new(h.clone()))?;
                h
            }};
        }
        Ok(Self {
            requests_total: ctr!("ct4_requests_total", "HTTP /embeddings requests"),
            items_total: ctr!("ct4_items_total", "input strings embedded"),
            errors_total: ctr!("ct4_errors_total", "requests that errored (5xx/inference)"),
            rejected_total: ctr!("ct4_rejected_total", "requests shed with 503 (queue full)"),
            in_flight: gauge!("ct4_in_flight", "accepted requests not yet responded"),
            queue_depth: gauge!("ct4_queue_depth", "items queued awaiting a batch"),
            batch_size: hist!(
                "ct4_batch_size",
                "inference batch size",
                vec![1.0, 2.0, 4.0, 8.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0]
            ),
            lat_tokenize_s: hist!("ct4_tokenize_seconds", "tokenize stage", latency_buckets()),
            lat_queue_wait_s: hist!("ct4_queue_wait_seconds", "enqueue→batch wait", latency_buckets()),
            lat_inference_s: hist!("ct4_inference_seconds", "ORT run + head", latency_buckets()),
            lat_total_s: hist!("ct4_request_seconds", "end-to-end request", latency_buckets()),
            rss_bytes: gauge!("ct4_rss_bytes", "process resident set size"),
            registry,
        })
    }

    /// Render the exposition text, refreshing RSS first.
    pub fn render(&self) -> String {
        if let Some(rss) = read_rss_bytes() {
            self.rss_bytes.set(rss);
        }
        let mut buf = String::new();
        let mf = self.registry.gather();
        let _ = TextEncoder::new().encode_utf8(&mf, &mut buf);
        buf
    }
}

/// RSS from /proc/self/statm (field 2 = resident pages) × page size.
fn read_rss_bytes() -> Option<i64> {
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let resident_pages: i64 = statm.split_whitespace().nth(1)?.parse().ok()?;
    let page = 4096i64; // Linux x86_64 default
    Some(resident_pages * page)
}
