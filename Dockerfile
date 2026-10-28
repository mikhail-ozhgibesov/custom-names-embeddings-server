# syntax=docker/dockerfile:1
# Multi-stage build: compile a static-ORT binary, ship it on a minimal
# distroless/cc base (has libstdc++/libgcc/libm/libc — all the binary needs).
# Bases pinned by digest (docker/images.lock.json); ONNX Runtime is pinned by
# the `ort` crate version (=2.0.0-rc.12 → ONNX Runtime 1.24), fetched as a
# static libonnxruntime.a.

# Trixie (glibc 2.41), NOT bookworm: the prebuilt ONNX Runtime static lib needs
# glibc >= 2.38 symbols (e.g. __isoc23_strtol). Builder AND runtime must match.
FROM rust:1.96-trixie@sha256:6df234c1eb92b0545468fab8c18fc5f9adfb994e7d4f67d81d45fe2fcabf5657 AS builder

# openssl-sys (transitive via ort's downloader) needs these; oniguruma (HF
# tokenizers `onig`) is compiled from bundled C using the image's gcc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libssl-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY tests ./tests
# Build the release binary (ort fetches + statically links ONNX Runtime 1.24).
RUN cargo build --release --bin custom-names-embeddings-server \
    && strip target/release/custom-names-embeddings-server

FROM gcr.io/distroless/cc-debian13@sha256:a017e74bd2a12d98342dbecd33d121d2b160415ed777573dc1808969e989d94d
COPY --from=builder /build/target/release/custom-names-embeddings-server /usr/local/bin/custom-names-embeddings-server
# Model baked in at build time from the repo's LFS-tracked model/ dir
# (native engine: weights.bin + manifest + tokenizer; ort engine: model.onnx).
# Serve a different model by bind-mounting another dir at /model.
COPY model /model
ENV CT4_PORT=8080 \
    CT4_MODEL_DIR=/model
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/custom-names-embeddings-server"]
