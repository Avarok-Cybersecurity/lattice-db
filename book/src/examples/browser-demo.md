# Browser Demo

LatticeDB runs entirely in the browser via WebAssembly. No server required.

## Live Demo

Open the standalone HTML demo:

**[examples/browser-demo.html](https://github.com/Avarok-Cybersecurity/lattice-db/blob/main/examples/browser-demo.html)**

Or download and open locally:

```bash
curl -O https://raw.githubusercontent.com/Avarok-Cybersecurity/lattice-db/main/examples/browser-demo.html
open browser-demo.html  # macOS
# or: xdg-open browser-demo.html  # Linux
# or: start browser-demo.html     # Windows
```

## Using from CDN

Import directly from GitHub Pages:

```html
<script type="module">
    const CDN = 'https://avarok-cybersecurity.github.io/lattice-db';

    const { LatticeDB } = await import(`${CDN}/js/lattice-db.min.js`);
    const db = await LatticeDB.init(`${CDN}/wasm/lattice_server_bg.wasm`);

    // Create a collection
    db.createCollection('vectors', {
        vectors: { size: 128, distance: 'Cosine' }
    });

    // Insert data
    db.upsert('vectors', [
        { id: 1, vector: new Array(128).fill(0.1), payload: { name: 'example' } }
    ]);

    // Search
    const results = db.search('vectors', new Array(128).fill(0.1), 5);
    console.log(results);
</script>
```

## Available Bundles

| File | Format | Size | Use Case |
|------|--------|------|----------|
| `lattice-db.min.js` | ESM (minified) | ~15KB | Production |
| `lattice-db.esm.js` | ESM | ~25KB | Development |
| `lattice-db.js` | CommonJS | ~25KB | Node.js/bundlers |
| `lattice_server_bg.wasm` | WASM | ~500KB | Required runtime |

## NPM Installation

For bundled applications, install from npm:

```bash
npm install lattice-db
```

```typescript
import { LatticeDB } from 'lattice-db';

const db = await LatticeDB.init();
```

See [TypeScript API](../api/typescript.md) for complete documentation.
