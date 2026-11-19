// Shared by the benchmark entries: time each handled request in-isolate.
//
// Local workerd advances performance.now() during execution (production
// Workers freeze it until I/O), so `x-handler-ms` is the handler's wall time
// inside the isolate; with no I/O on the path that is its CPU time. workerd
// rounds timers to whole milliseconds, so expect +/-1 ms.
import type { LatticeWorkerHandler } from '../../../src/index';

export function timed(handler: LatticeWorkerHandler): ExportedHandler {
  return {
    async fetch(request: Request): Promise<Response> {
      if (new URL(request.url).pathname === '/noop') return new Response('ok');
      const start = performance.now();
      const response = (await handler.fetch(request, {})) ?? new Response('not found', { status: 404 });
      const body = await response.arrayBuffer();
      const elapsed = performance.now() - start;
      const headers = new Headers(response.headers);
      headers.set('x-handler-ms', elapsed.toFixed(3));
      return new Response(body, { status: response.status, headers });
    },
  };
}
