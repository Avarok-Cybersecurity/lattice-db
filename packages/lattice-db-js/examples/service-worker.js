// LatticeDB in a service worker: the page calls fetch('/api/lattice/...') and
// gets the same Qdrant-compatible REST API the native server exposes.
//
// Register from the page as a module worker:
//   navigator.serviceWorker.register('/sw.js', { type: 'module' });

import { LatticeHost } from 'lattice-db';

const PREFIX = '/api/lattice';

// Load once per service-worker lifetime; every fetch awaits the same promise.
const ready = (async () => {
  const host = await LatticeHost.create({ readOnly: true });
  const response = await fetch('/corpus.lattice');
  if (!response.ok) throw new Error(`corpus fetch returned ${response.status}`);
  host.importSnapshot(new Uint8Array(await response.arrayBuffer()), 'docs', 'create');
  return host.fetchHandler({ prefix: PREFIX });
})();

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()));

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  if (url.origin !== self.location.origin) return;
  if (url.pathname !== PREFIX && !url.pathname.startsWith(`${PREFIX}/`)) return;
  event.respondWith(ready.then((api) => api.handleFetch(event.request)));
});
