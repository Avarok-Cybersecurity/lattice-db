import { readFileSync, writeFileSync, readdirSync, existsSync } from 'fs';
import { join, dirname, relative } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(__dirname, '../../..');

interface DocChunk {
  id: string;
  section: string;
  content: string;
  metadata: {
    category: string;
    source: string;
    method?: string;
    path?: string;
  };
}

interface DocSource {
  path: string;
  label: string;
}

// Recursively collect every markdown file under a directory.
function walkMarkdown(dir: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      out.push(...walkMarkdown(full));
    } else if (entry.name.endsWith('.md')) {
      out.push(full);
    }
  }
  return out;
}

// The full documentation corpus bundled into the assistant's knowledge base:
// every mdBook chapter plus the API-bearing crate/package READMEs.
function collectSources(): DocSource[] {
  const sources: DocSource[] = [];

  for (const file of walkMarkdown(join(REPO_ROOT, 'book/src'))) {
    if (file.endsWith('SUMMARY.md')) continue; // table of contents, not content
    sources.push({ path: file, label: relative(REPO_ROOT, file) });
  }

  // Committed crate READMEs only. (The generated wasm-pack README is a build
  // artifact and is not present in all CI checkouts, so it is not a source.)
  const readmes = ['crates/lattice-server/README.md', 'crates/lattice-core/README.md'];
  for (const rel of readmes) {
    sources.push({ path: join(REPO_ROOT, rel), label: rel });
  }

  // Skip any source that is absent so a missing optional file never fails the
  // docs build; warn so it is visible in logs.
  return sources.filter(src => {
    if (existsSync(src.path)) return true;
    console.warn(`  (skipping missing source: ${src.label})`);
    return false;
  });
}

// Derive a category from the source's directory (book/src/<category>/...),
// falling back to a heading heuristic for top-level READMEs.
function categoryFor(sourceLabel: string, h2: string): string {
  const match = sourceLabel.match(/book\/src\/([^/]+)\//);
  if (match) return match[1];
  return h2.toLowerCase().includes('graph') ? 'graph' : 'api';
}

interface DocsManifest {
  version: string;
  generated_at: string;
  source: string;
  chunks: DocChunk[];
}

function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '');
}

function extractApiMetadata(content: string): { method?: string; path?: string } {
  const httpMatch = content.match(/^(GET|POST|PUT|DELETE|PATCH)\s+(\S+)/m);
  if (httpMatch) {
    return { method: httpMatch[1], path: httpMatch[2] };
  }
  return {};
}

function parseMarkdownSections(markdown: string, sourceLabel: string): DocChunk[] {
  const chunks: DocChunk[] = [];
  const lines = markdown.split('\n');
  const sourceSlug = slugify(sourceLabel);

  let currentH2 = '';
  let currentH3 = '';
  let currentContent: string[] = [];
  let chunkIndex = 0;

  function flushChunk() {
    if (currentContent.length === 0) return;

    const content = currentContent.join('\n').trim();
    if (content.length < 50) {
      currentContent = [];
      return;
    }

    const section = currentH3
      ? `${currentH2} > ${currentH3}`
      : currentH2 || 'Overview';

    const id = slugify(section) || `chunk-${chunkIndex}`;
    const apiMeta = extractApiMetadata(content);

    chunks.push({
      // Prefix with the source slug so ids stay unique across all files.
      id: `${sourceSlug}-${id}-${chunkIndex}`,
      section,
      content,
      metadata: {
        category: categoryFor(sourceLabel, currentH2),
        source: sourceLabel,
        ...apiMeta
      }
    });

    chunkIndex++;
    currentContent = [];
  }

  for (const line of lines) {
    if (line.startsWith('## ')) {
      flushChunk();
      currentH2 = line.replace('## ', '').trim();
      currentH3 = '';
    } else if (line.startsWith('### ')) {
      flushChunk();
      currentH3 = line.replace('### ', '').trim();
    } else if (line.startsWith('---')) {
      flushChunk();
    } else {
      currentContent.push(line);
    }
  }

  flushChunk();
  return chunks;
}

function main() {
  const outputPath = join(__dirname, '../public/docs/lattice-docs.json');
  const sources = collectSources();

  console.log(`Ingesting ${sources.length} documentation sources...`);
  const chunks: DocChunk[] = [];
  for (const src of sources) {
    const md = readFileSync(src.path, 'utf-8');
    const parsed = parseMarkdownSections(md, src.label);
    chunks.push(...parsed);
    console.log(`  ${src.label}: ${parsed.length} chunks`);
  }

  const manifest: DocsManifest = {
    version: '2.0.0',
    generated_at: new Date().toISOString(),
    source: `${sources.length} files (book/src chapters + crate/package READMEs)`,
    chunks
  };

  console.log(`\nGenerated ${chunks.length} chunks from ${sources.length} sources`);

  writeFileSync(outputPath, JSON.stringify(manifest, null, 2));
  console.log(`Written to ${outputPath}`);

  // Print per-category summary.
  const categories = new Map<string, number>();
  for (const c of chunks) {
    categories.set(c.metadata.category, (categories.get(c.metadata.category) ?? 0) + 1);
  }
  console.log('\nChunks by category:');
  for (const [category, count] of [...categories].sort()) {
    console.log(`  - ${category}: ${count}`);
  }
}

main();
