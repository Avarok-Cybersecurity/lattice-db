import type { DocChunk, DocsManifest } from './types';

const CORS_PROXY = 'https://corsproxy.io/?';

export async function loadLatticeDBDocs(): Promise<DocChunk[]> {
  const response = await fetch('/docs/lattice-docs.json');
  if (!response.ok) {
    throw new Error(`Failed to load docs: ${response.status}`);
  }
  const manifest: DocsManifest = await response.json();
  return manifest.chunks;
}

export async function fetchUrlContent(url: string): Promise<string> {
  // Validate URL
  let parsedUrl: URL;
  try {
    parsedUrl = new URL(url);
  } catch {
    throw new Error('Invalid URL format');
  }

  // Only allow http/https
  if (!['http:', 'https:'].includes(parsedUrl.protocol)) {
    throw new Error('Only HTTP/HTTPS URLs are supported');
  }

  let response: Response | null = null;

  // Try direct fetch first (works for CORS-enabled sites like raw.githubusercontent.com)
  try {
    response = await fetch(url);
    if (!response.ok) {
      response = null;
    }
  } catch {
    // CORS blocked, will try proxy
  }

  // Fall back to CORS proxy
  if (!response) {
    const proxyUrl = CORS_PROXY + encodeURIComponent(url);
    response = await fetch(proxyUrl);

    if (!response.ok) {
      throw new Error(`Failed to fetch URL: ${response.status}`);
    }
  }

  const text = await response.text();

  // Return raw text for markdown/text files, extract from HTML otherwise
  const path = parsedUrl.pathname.toLowerCase();
  if (path.endsWith('.md') || path.endsWith('.txt')) {
    return text;
  }

  return extractTextFromHtml(text);
}

function extractTextFromHtml(html: string): string {
  // Create a temporary element to parse HTML
  const parser = new DOMParser();
  const doc = parser.parseFromString(html, 'text/html');

  // Remove script and style elements
  doc.querySelectorAll('script, style, nav, header, footer, aside').forEach(el => el.remove());

  // Try to get main content
  const main = doc.querySelector('main, article, .content, #content, .post, .article');
  const container = main ?? doc.body;

  // Get text content
  let text = container.textContent ?? '';

  // Clean up whitespace
  text = text
    .replace(/\s+/g, ' ')
    .replace(/\n\s*\n/g, '\n\n')
    .trim();

  // Truncate if too long (max ~10k chars)
  if (text.length > 10000) {
    text = text.slice(0, 10000) + '...';
  }

  return text;
}

export async function parseFile(file: File): Promise<string> {
  const MAX_SIZE = 1024 * 1024; // 1MB

  if (file.size > MAX_SIZE) {
    throw new Error(`File too large: ${(file.size / 1024).toFixed(1)}KB (max 1MB)`);
  }

  const text = await file.text();
  const ext = file.name.split('.').pop()?.toLowerCase();

  switch (ext) {
    case 'txt':
      return text;

    case 'md':
      // Keep markdown as-is, it's already text
      return text;

    case 'json':
      try {
        const json = JSON.parse(text);
        // Convert JSON to readable text
        return JSON.stringify(json, null, 2);
      } catch {
        throw new Error('Invalid JSON file');
      }

    case 'csv':
      return parseCsv(text);

    default:
      throw new Error(`Unsupported file type: .${ext}`);
  }
}

function parseCsv(csv: string): string {
  const lines = csv.trim().split('\n');
  if (lines.length === 0) return '';

  // Get headers
  const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));

  // Convert each row to readable text
  const rows = lines.slice(1).map(line => {
    const values = line.split(',').map(v => v.trim().replace(/^"|"$/g, ''));
    return headers.map((h, i) => `${h}: ${values[i] ?? ''}`).join(', ');
  });

  return rows.join('\n');
}

export function getFileTitle(file: File): string {
  return file.name.replace(/\.[^.]+$/, '');
}

export function getUrlTitle(url: string): string {
  try {
    const parsed = new URL(url);
    const path = parsed.pathname.replace(/\/$/, '');
    const lastSegment = path.split('/').pop();
    return lastSegment || parsed.hostname;
  } catch {
    return url.slice(0, 50);
  }
}
