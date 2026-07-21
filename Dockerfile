# Stage 1: Plan dependencies with cargo-chef for layer caching
FROM rust:1.85-bookworm AS planner
RUN cargo install cargo-chef
WORKDIR /app
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# Stage 2: Build dependencies then application
FROM rust:1.85-bookworm AS builder
RUN cargo install cargo-chef
WORKDIR /app

# Cook dependencies first (cached unless Cargo.toml/lock changes)
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json --bin lattice-server

# Copy source and build
COPY . .
RUN cargo build --release --bin lattice-server

# Stage 3: Minimal runtime image
FROM debian:bookworm-slim AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN useradd --create-home --shell /bin/bash lattice

# Data directory for durable collections
RUN mkdir -p /data && chown lattice:lattice /data

COPY --from=builder /app/target/release/lattice-server /usr/local/bin/lattice-server

USER lattice

# Default: persist to /data, override with -e to change behavior
ENV LATTICE_DATA_DIR=/data
ENV RUST_LOG=info

EXPOSE 6334

ENTRYPOINT ["lattice-server"]
CMD ["0.0.0.0:6334"]
