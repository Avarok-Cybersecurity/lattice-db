import { defineConfig } from 'vite';

export default defineConfig({
  base: process.env.GITHUB_ACTIONS ? '/lattice-db/chat/' : '/',
  optimizeDeps: {
    exclude: ['lattice-db']
  },
  build: {
    target: 'esnext'
  }
});
