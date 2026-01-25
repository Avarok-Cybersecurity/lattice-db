import { readFileSync, writeFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

interface DocChunk {
  id: string;
  section: string;
  content: string;
  metadata: {
    category: string;
    method?: string;
    path?: string;
  };
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

function parseMarkdownSections(markdown: string): DocChunk[] {
  const chunks: DocChunk[] = [];
  const lines = markdown.split('\n');

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
      id: `${id}-${chunkIndex}`,
      section,
      content,
      metadata: {
        category: currentH2.toLowerCase().includes('graph') ? 'graph' : 'api',
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
  const readmePath = join(__dirname, '../../../crates/lattice-server/README.md');
  const outputPath = join(__dirname, '../public/docs/lattice-docs.json');

  console.log('Reading README.md...');
  const readme = readFileSync(readmePath, 'utf-8');

  console.log('Parsing markdown sections...');
  const chunks = parseMarkdownSections(readme);

  const manifest: DocsManifest = {
    version: '1.0.0',
    generated_at: new Date().toISOString(),
    source: 'crates/lattice-server/README.md',
    chunks
  };

  console.log(`Generated ${chunks.length} chunks`);

  writeFileSync(outputPath, JSON.stringify(manifest, null, 2));
  console.log(`Written to ${outputPath}`);

  // Print summary
  const sections = new Set(chunks.map(c => c.section.split(' > ')[0]));
  console.log('\nSections:');
  for (const section of sections) {
    const count = chunks.filter(c => c.section.startsWith(section)).length;
    console.log(`  - ${section}: ${count} chunks`);
  }
}

main();
