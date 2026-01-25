# Installation

LatticeDB can be used in multiple environments: as a Rust library, a standalone server, or in the browser via WebAssembly.

## Rust Library

Add LatticeDB to your `Cargo.toml`:

```toml
[dependencies]
lattice-core = "0.1"
lattice-storage = "0.1"
```

For server applications with HTTP endpoints:

```toml
[dependencies]
lattice-server = "0.1"
```

## Standalone Server

### Pre-built Binaries

Download pre-built binaries from the [releases page](https://github.com/Avarok-Cybersecurity/lattice-db/releases):

- `lattice-db-linux-x64` - Linux (x86_64)
- `lattice-db-macos-x64` - macOS (Intel)
- `lattice-db-macos-arm64` - macOS (Apple Silicon)
- `lattice-db-windows-x64` - Windows (x86_64)

### From Source

```bash
# Clone the repository
git clone https://github.com/Avarok-Cybersecurity/lattice-db.git
cd lattice-db

# Build release binary
cargo build --release -p lattice-server

# Binary is at target/release/lattice-server
```

### Running the Server

```bash
# Start server on default port 6333
./lattice-server

# Or specify a custom port
./lattice-server --port 8080
```

## WASM / Browser

### NPM Package

```bash
npm install lattice-db
```

### CDN

```html
<script type="module">
  import init, { LatticeDB } from 'https://unpkg.com/lattice-db/lattice_db.js';

  await init();
  const db = new LatticeDB();
</script>
```

### Build from Source

```bash
# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build WASM package
wasm-pack build crates/lattice-server --target web --out-dir pkg --no-default-features --features wasm
```

## Docker

```bash
docker run -p 6333:6333 Avarok-Cybersecurity/lattice-db:latest
```

## System Requirements

### Native

- **Rust 1.75+** (for building from source)
- **64-bit OS** (Linux, macOS, or Windows)
- **2GB RAM** minimum (varies by dataset size)

### WASM

- **Modern browser** with WebAssembly support:
  - Chrome 89+
  - Firefox 89+
  - Safari 15+
  - Edge 89+
- **SIMD support** recommended for best performance (enabled by default in modern browsers)

## Feature Flags

LatticeDB uses feature flags for optional functionality:

| Feature | Description | Default |
|---------|-------------|---------|
| `simd` | SIMD-accelerated distance calculations | Enabled |
| `mmap` | Memory-mapped vector storage | Disabled |
| `openapi` | OpenAPI/Swagger documentation | Disabled |

Enable features in `Cargo.toml`:

```toml
[dependencies]
lattice-core = { version = "0.1", features = ["simd", "mmap"] }
```

## Verify Installation

### Native

```bash
# Run tests
cargo test --workspace

# Run benchmarks
cargo run -p lattice-bench --release --example quick_vector_bench
```

### WASM

```bash
# Run WASM tests in headless Chrome
wasm-pack test --headless --chrome crates/lattice-core
```

## Next Steps

- [Quick Start](./quickstart.md) - Create your first collection
- [WASM Browser Setup](./wasm.md) - Detailed browser integration guide
