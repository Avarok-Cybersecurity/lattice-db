# Development Setup

This chapter guides you through setting up a development environment for contributing to LatticeDB.

## Prerequisites

### Required Tools

- **Rust 1.75+**: Install via [rustup](https://rustup.rs/)
- **Git**: For version control
- **wasm-pack**: For WASM builds (optional)

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install wasm-pack (for WASM development)
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
```

### Optional Tools

- **Docker**: For running comparison databases (Qdrant, Neo4j)
- **mdbook**: For building documentation
- **criterion**: For benchmarking

```bash
# Install mdbook
cargo install mdbook

# Install criterion CLI
cargo install cargo-criterion
```

## Getting the Source

```bash
# Clone the repository
git clone https://github.com/Avarok-Cybersecurity/lattice-db.git
cd lattice-db

# Check out a feature branch
git checkout -b my-feature
```

## Project Structure

```
lattice-db/
├── Cargo.toml           # Workspace configuration
├── crates/
│   ├── lattice-core/    # Core library (pure logic)
│   ├── lattice-storage/ # Storage implementations
│   ├── lattice-server/  # HTTP/WASM server
│   └── lattice-bench/   # Benchmarks
├── book/                # Documentation (mdbook)
├── .github/
│   └── workflows/       # CI/CD pipelines
└── README.md
```

## Building

### Native Build

```bash
# Debug build (fast compilation)
cargo build

# Release build (optimized)
cargo build --release

# Build specific crate
cargo build -p lattice-core
```

### WASM Build

```bash
# Build WASM package
wasm-pack build crates/lattice-server \
    --target web \
    --out-dir pkg \
    --no-default-features \
    --features wasm

# Output is in crates/lattice-server/pkg/
```

### Build with Features

```bash
# Build with SIMD optimization
cargo build --release --features simd

# Build with memory-mapped storage
cargo build --release --features mmap

# Build with OpenAPI documentation
cargo build -p lattice-server --features openapi
```

## Running Tests

### All Tests

```bash
# Run all tests
cargo test --workspace

# Run with output
cargo test --workspace -- --nocapture
```

### Specific Tests

```bash
# Test specific crate
cargo test -p lattice-core

# Test specific module
cargo test -p lattice-core --lib hnsw

# Test specific function
cargo test -p lattice-core test_search_returns_k_results
```

### WASM Tests

```bash
# Requires Chrome installed
wasm-pack test --headless --chrome crates/lattice-core

# Run in Firefox
wasm-pack test --headless --firefox crates/lattice-core
```

## Running Benchmarks

### Quick Benchmark

```bash
# Fast iteration benchmark
cargo run -p lattice-bench --release --example quick_vector_bench
```

### Full Criterion Benchmarks

```bash
# All benchmarks
cargo bench -p lattice-bench

# Specific benchmark
cargo bench -p lattice-bench --bench vector_ops

# With comparison to baseline
cargo bench -p lattice-bench -- --baseline main
```

### View Reports

```bash
# Open HTML report
open target/criterion/report/index.html
```

## Documentation

### Build API Docs

```bash
# Generate rustdoc
cargo doc --workspace --no-deps

# Open in browser
cargo doc --workspace --no-deps --open
```

### Build the Book

```bash
# Build mdbook
mdbook build book

# Serve locally with hot reload
mdbook serve book --open
```

## Code Style

### Formatting

```bash
# Format all code
cargo fmt --all

# Check formatting
cargo fmt --all -- --check
```

### Linting

```bash
# Run clippy
cargo clippy --workspace -- -D warnings

# With all targets
cargo clippy --workspace --all-targets -- -D warnings
```

### Pre-commit Hook

```bash
# Install pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/sh
cargo fmt --all -- --check
cargo clippy --workspace -- -D warnings
cargo test --workspace
EOF
chmod +x .git/hooks/pre-commit
```

## IDE Setup

### VS Code

Recommended extensions:
- **rust-analyzer**: Rust language support
- **CodeLLDB**: Debugging
- **Even Better TOML**: TOML syntax
- **Error Lens**: Inline error display

Settings (`.vscode/settings.json`):
```json
{
  "rust-analyzer.check.command": "clippy",
  "rust-analyzer.cargo.features": ["simd"],
  "[rust]": {
    "editor.formatOnSave": true
  }
}
```

### JetBrains (RustRover/CLion)

- Install Rust plugin
- Enable "Run rustfmt on save"
- Configure Clippy as external linter

## Debugging

### Native

```bash
# Debug build
cargo build

# Run with debugger (VS Code/CodeLLDB)
# Or use lldb/gdb directly
lldb target/debug/lattice-server
```

### WASM

```bash
# Build with debug info
wasm-pack build --dev crates/lattice-server --target web

# Use browser DevTools:
# - Sources tab for breakpoints
# - Console for WASM errors
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RUST_LOG` | Log level | `info` |
| `RUST_BACKTRACE` | Enable backtraces | `0` |
| `LATTICE_PORT` | Server port | `6333` |

```bash
# Example
RUST_LOG=debug cargo run -p lattice-server
```

## Docker Development

### Run Comparison Databases

```bash
# Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Neo4j
docker run -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    neo4j:community
```

### Build LatticeDB Image

```bash
docker build -t lattice-db .
docker run -p 6333:6333 lattice-db
```

## Troubleshooting

### Build Failures

```bash
# Clean and rebuild
cargo clean
cargo build

# Update dependencies
cargo update
```

### Test Failures

```bash
# Run with backtrace
RUST_BACKTRACE=1 cargo test

# Run single-threaded (for debugging)
cargo test -- --test-threads=1
```

### WASM Issues

```bash
# Ensure wasm32 target is installed
rustup target add wasm32-unknown-unknown

# Check wasm-pack version
wasm-pack --version  # Should be 0.12+
```

## Next Steps

- [Testing](./testing.md) - Testing guidelines and patterns
- [Architecture](../architecture/overview.md) - Understand the codebase
