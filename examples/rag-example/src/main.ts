import { RAGEngine } from './rag';
import { loadLatticeDBDocs, fetchUrlContent, parseFile, getFileTitle, getUrlTitle } from './documents';
import * as opfs from './opfs';
import { DEFAULT_EMBEDDING_MODEL } from './openrouter';
import { marked } from 'marked';
import type { Message, ManagedDocument } from './types';

// Configure marked for safe rendering
marked.setOptions({
  breaks: true,
  gfm: true
});

let engine: RAGEngine | null = null;
let chatHistory: Message[] = [];

// --- OPFS persistence state ---------------------------------------------
/// Snapshot the current session writes into. Created on the first autosave,
/// or adopted when the user restores an existing snapshot.
let currentSnapshotId: string | null = null;
let currentSnapshotCreatedAt = 0;
let currentSnapshotRevision = 0;
let autosaveTimer: number | undefined;
/// Coalesce bursts of document changes (a 492-chunk load fires many) into one
/// write once things settle.
const AUTOSAVE_DEBOUNCE_MS = 1200;

function getElement<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Element not found: ${id}`);
  return el as T;
}

function setStatus(message: string, isError = false): void {
  const status = getElement<HTMLDivElement>('status');
  status.textContent = message;
  status.className = `status ${isError ? 'error' : 'success'}`;
  status.style.display = 'block';
  setTimeout(() => { status.style.display = 'none'; }, 3000);
}

function escapeHtml(text: string): string {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function addMessageToChat(role: 'user' | 'assistant', content: string, sources?: { text: string; score: number }[]): void {
  const chatMessages = getElement<HTMLDivElement>('chat-messages');

  const messageDiv = document.createElement('div');
  messageDiv.className = `chat-message ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? 'You' : 'AI';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = marked.parse(content) as string;

  messageDiv.appendChild(avatar);
  messageDiv.appendChild(bubble);

  if (sources && sources.length > 0) {
    const sourcesDiv = document.createElement('div');
    sourcesDiv.className = 'message-sources';
    sourcesDiv.innerHTML = `
      <button class="sources-toggle" onclick="this.parentElement.classList.toggle('expanded')">
        Sources (${sources.length})
      </button>
      <div class="sources-content">
        ${sources.map((s, i) => `
          <div class="source-item">
            <span class="source-badge">[${i + 1}] ${(s.score * 100).toFixed(0)}%</span>
            <span class="source-text">${escapeHtml(s.text.slice(0, 150))}${s.text.length > 150 ? '...' : ''}</span>
          </div>
        `).join('')}
      </div>
    `;
    messageDiv.appendChild(sourcesDiv);
  }

  chatMessages.appendChild(messageDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showLoadingBubble(): HTMLDivElement {
  const chatMessages = getElement<HTMLDivElement>('chat-messages');

  const messageDiv = document.createElement('div');
  messageDiv.className = 'chat-message assistant';
  messageDiv.id = 'loading-bubble';

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = 'AI';

  const bubble = document.createElement('div');
  bubble.className = 'bubble loading';
  bubble.innerHTML = '<span class="typing-indicator"><span></span><span></span><span></span></span>';

  messageDiv.appendChild(avatar);
  messageDiv.appendChild(bubble);
  chatMessages.appendChild(messageDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  return messageDiv;
}

function removeLoadingBubble(): void {
  const loadingBubble = document.getElementById('loading-bubble');
  if (loadingBubble) {
    loadingBubble.remove();
  }
}

// Step 1 completed: collapse the connect form, unlock the knowledge step.
function onConnected(): void {
  const stepConnect = getElement<HTMLElement>('step-connect');
  stepConnect.classList.remove('is-active');
  stepConnect.classList.add('is-done');
  getElement<HTMLElement>('connect-form').hidden = true;
  getElement<HTMLElement>('connected-badge').hidden = false;

  const stepKnowledge = getElement<HTMLElement>('step-knowledge');
  stepKnowledge.classList.remove('is-locked');
  stepKnowledge.classList.add('is-active');
  getElement<HTMLButtonElement>('source-docs').disabled = false;
  getElement<HTMLButtonElement>('source-byo').disabled = false;

  getElement<HTMLButtonElement>('send-btn').disabled = false;
  getElement<HTMLInputElement>('message-input').disabled = false;
}

// Select a knowledge source and reveal its panel.
function selectSource(source: 'docs' | 'byo'): void {
  const isDocs = source === 'docs';
  getElement<HTMLButtonElement>('source-docs').classList.toggle('is-selected', isDocs);
  getElement<HTMLButtonElement>('source-byo').classList.toggle('is-selected', !isDocs);
  getElement<HTMLElement>('docs-panel').hidden = !isDocs;
  getElement<HTMLElement>('byo-panel').hidden = isDocs;
}

// Return to step 1 to edit the API key.
function changeKey(): void {
  const stepConnect = getElement<HTMLElement>('step-connect');
  stepConnect.classList.add('is-active');
  stepConnect.classList.remove('is-done');
  getElement<HTMLElement>('connect-form').hidden = false;
  getElement<HTMLElement>('connected-badge').hidden = true;
  const initBtn = getElement<HTMLButtonElement>('init-btn');
  initBtn.disabled = false;
  initBtn.textContent = 'Connect';
  getElement<HTMLInputElement>('api-key').focus();
}

async function initializeEngine(): Promise<void> {
  const apiKey = getElement<HTMLInputElement>('api-key').value.trim();

  if (!apiKey) {
    setStatus('Please enter your OpenRouter API key', true);
    return;
  }

  const initBtn = getElement<HTMLButtonElement>('init-btn');
  initBtn.disabled = true;
  initBtn.textContent = 'Connecting…';

  getElement<HTMLElement>('chat-section').style.display = 'flex';
  getElement<HTMLElement>('welcome-section').style.display = 'none';
  showLoadingBubble();

  try {
    engine = new RAGEngine({ apiKey });
    await engine.init();

    removeLoadingBubble();
    setStatus('Connected — free models ready');
    onConnected();
    localStorage.setItem('openrouter-api-key', apiKey);

    addMessageToChat('assistant', 'Connected! Now pick a **knowledge source** on the left — load the **LatticeDB Docs**, or add **your own documents** — then ask me anything. LatticeDB runs **inside your own browser** via WebAssembly.');

    // Surface anything already saved on this device so the user can pick up
    // where they left off instead of re-embedding.
    await refreshSavedSessions();
  } catch (error) {
    removeLoadingBubble();
    setStatus(`Connection failed: ${error}`, true);
    initBtn.disabled = false;
    initBtn.textContent = 'Connect';
    getElement<HTMLElement>('chat-section').style.display = 'none';
    getElement<HTMLElement>('welcome-section').style.display = 'flex';
  }
}

async function loadLatticeDBDocsHandler(): Promise<void> {
  if (!engine) {
    setStatus('Please initialize first', true);
    return;
  }

  const loadBtn = getElement<HTMLButtonElement>('load-docs');
  loadBtn.disabled = true;
  showLoadingBubble();

  try {
    const chunks = await loadLatticeDBDocs();

    const docs = chunks.map((chunk, index) => ({
      id: index,
      text: chunk.content,
      metadata: { ...chunk.metadata, section: chunk.section }
    }));
    const sections = chunks.map(c => c.section);

    await engine.addDocuments(
      docs,
      'docs',
      doc => sections[doc.id],
      (done, total) => setStatus(`Embedding documentation… ${done}/${total}`)
    );

    const sourceCount = new Set(chunks.map(c => c.metadata.source)).size;

    removeLoadingBubble();
    setStatus(`Loaded ${chunks.length} sections from ${sourceCount} sources`);
    loadBtn.textContent = '✓ Documentation loaded';
    updateDocCount();

    addMessageToChat('assistant', `I've loaded ${chunks.length} documentation sections from ${sourceCount} sources — the full LatticeDB book (getting started, REST/TypeScript/Rust APIs, vector search, graph/Cypher, architecture, performance) plus crate READMEs. Ask me anything about collections, points, vector search, graph traversal, HNSW, quantization, and more!`);
  } catch (error) {
    removeLoadingBubble();
    setStatus(`Failed to load docs: ${error}`, true);
    loadBtn.disabled = false;
  }
}

async function addFromUrl(): Promise<void> {
  if (!engine) {
    setStatus('Please initialize first', true);
    return;
  }

  const urlInput = getElement<HTMLInputElement>('url-input');
  const addBtn = getElement<HTMLButtonElement>('add-url');
  const url = urlInput.value.trim();

  if (!url) {
    setStatus('Please enter a URL', true);
    return;
  }

  addBtn.disabled = true;
  addBtn.textContent = '...';

  try {
    const content = await fetchUrlContent(url);
    const title = getUrlTitle(url);

    await engine.addDocument(
      { id: Date.now(), text: content },
      'url',
      title
    );

    setStatus(`Added document from ${title}`);
    urlInput.value = '';
    updateDocCount();
  } catch (error) {
    setStatus(`Failed to fetch URL: ${error}`, true);
  } finally {
    addBtn.disabled = false;
    addBtn.textContent = 'Add';
  }
}

async function uploadFiles(): Promise<void> {
  const fileInput = getElement<HTMLInputElement>('file-input');
  fileInput.click();
}

async function handleFileUpload(event: Event): Promise<void> {
  if (!engine) return;

  const input = event.target as HTMLInputElement;
  const files = input.files;

  if (!files || files.length === 0) return;

  const uploadBtn = getElement<HTMLButtonElement>('upload-btn');
  uploadBtn.disabled = true;
  uploadBtn.textContent = 'Uploading...';

  try {
    let addedCount = 0;

    for (const file of files) {
      try {
        const content = await parseFile(file);
        const title = getFileTitle(file);

        await engine.addDocument(
          { id: Math.floor(Date.now() + Math.random() * 1000), text: content },
          'file',
          title
        );
        addedCount++;
      } catch (error) {
        setStatus(`Error processing ${file.name}: ${error}`, true);
      }
    }

    if (addedCount > 0) {
      setStatus(`Added ${addedCount} file(s)`);
      updateDocCount();
    }
  } finally {
    input.value = '';
    uploadBtn.disabled = false;
    uploadBtn.textContent = 'Upload Files';
  }
}

async function addDocument(): Promise<void> {
  if (!engine) {
    setStatus('Please initialize first', true);
    return;
  }

  const textArea = getElement<HTMLTextAreaElement>('doc-text');
  const addBtn = getElement<HTMLButtonElement>('add-doc');
  const text = textArea.value.trim();

  if (!text) {
    setStatus('Please enter document text', true);
    return;
  }

  addBtn.disabled = true;
  addBtn.textContent = 'Adding...';

  try {
    await engine.addDocument({ id: Date.now(), text }, 'manual');
    setStatus('Document added!');
    textArea.value = '';
    updateDocCount();
  } catch (error) {
    setStatus(`Failed to add document: ${error}`, true);
  } finally {
    addBtn.disabled = false;
    addBtn.textContent = 'Add Document';
  }
}

async function sendMessage(): Promise<void> {
  if (!engine) {
    setStatus('Please initialize first', true);
    return;
  }

  const input = getElement<HTMLInputElement>('message-input');
  const message = input.value.trim();

  if (!message) return;

  input.value = '';
  addMessageToChat('user', message);
  chatHistory.push({ role: 'user', content: message });

  showLoadingBubble();

  try {
    const { answer, sources } = await engine.query(message, chatHistory.slice(0, -1));
    removeLoadingBubble();
    chatHistory.push({ role: 'assistant', content: answer });
    addMessageToChat('assistant', answer, sources);
  } catch (error) {
    removeLoadingBubble();
    const errorMsg = `Sorry, I encountered an error: ${error}`;
    addMessageToChat('assistant', errorMsg);
    setStatus(`Query failed: ${error}`, true);
  } finally {
    input.focus();
  }
}

function clearChat(): void {
  chatHistory = [];
  getElement<HTMLDivElement>('chat-messages').innerHTML = '';
  addMessageToChat('assistant', "Chat cleared. How can I help you?");
}

function updateDocCount(): void {
  const count = engine?.getDocumentCount() ?? 0;
  getElement<HTMLSpanElement>('doc-count').textContent = count.toString();
  getElement<HTMLSpanElement>('doc-noun').textContent = count === 1 ? 'document' : 'documents';
  // Reveal the knowledge-base status row once documents exist.
  getElement<HTMLElement>('kb-status').hidden = count === 0;
  // Every mutation funnels through here, so this is the single autosave hook.
  scheduleAutosave();
}

// ===========================================================================
// OPFS persistence
// ===========================================================================

/** Human label for a snapshot, derived from where its documents came from. */
function describeDocuments(docs: ManagedDocument[]): { label: string; sources: Record<string, number> } {
  const sources: Record<string, number> = {};
  for (const doc of docs) {
    sources[doc.source] = (sources[doc.source] ?? 0) + 1;
  }

  const names: Record<string, string> = {
    docs: 'LatticeDB Docs',
    url: 'Web pages',
    file: 'Uploaded files',
    manual: 'Pasted text'
  };
  const primary = Object.entries(sources).sort((a, b) => b[1] - a[1])[0];
  const label = primary ? names[primary[0]] ?? 'Knowledge base' : 'Knowledge base';
  const extra = Object.keys(sources).length - 1;

  return { label: extra > 0 ? `${label} +${extra} more` : label, sources };
}

function setSaveState(text: string, saving: boolean): void {
  const row = getElement<HTMLElement>('save-state');
  row.hidden = false;
  row.classList.toggle('is-saving', saving);
  getElement<HTMLSpanElement>('save-text').textContent = text;
}

/** Debounced write of the current knowledge base to OPFS. */
function scheduleAutosave(): void {
  if (!opfs.isSupported() || !engine) return;

  window.clearTimeout(autosaveTimer);
  autosaveTimer = window.setTimeout(() => {
    void autosave();
  }, AUTOSAVE_DEBOUNCE_MS);
}

async function autosave(): Promise<void> {
  if (!engine) return;

  const exported = engine.exportSnapshot();
  const dimension = engine.getVectorDimension();

  // Nothing to persist: drop the snapshot so the UI doesn't advertise an
  // empty one.
  if (!exported || dimension === null) {
    if (currentSnapshotId) {
      await opfs.deleteSnapshot(currentSnapshotId).catch(() => undefined);
      currentSnapshotId = null;
      currentSnapshotRevision = 0;
      getElement<HTMLElement>('save-state').hidden = true;
      await refreshSavedSessions();
    }
    return;
  }

  setSaveState('Saving…', true);
  try {
    const now = Date.now();
    if (!currentSnapshotId) {
      currentSnapshotId = `kb-${now.toString(36)}`;
      currentSnapshotCreatedAt = now;
    }
    currentSnapshotRevision += 1;

    const { label, sources } = describeDocuments(exported.documents);
    await opfs.saveSnapshot(
      {
        id: currentSnapshotId,
        label,
        revision: currentSnapshotRevision,
        createdAt: currentSnapshotCreatedAt,
        updatedAt: now,
        documentCount: exported.documents.length,
        vectorDimension: dimension,
        embeddingModel: DEFAULT_EMBEDDING_MODEL,
        sources
      },
      exported.documents,
      exported.vectors
    );

    setSaveState(`Saved · v${currentSnapshotRevision}`, false);
    await refreshSavedSessions();
  } catch (error) {
    setSaveState('Save failed', false);
    console.error('OPFS autosave failed', error);
  }
}

/** Render the "Saved on this device" panel in step 2. */
async function refreshSavedSessions(): Promise<void> {
  const panel = getElement<HTMLElement>('saved-sessions');
  const list = getElement<HTMLElement>('saved-list');

  if (!opfs.isSupported()) {
    panel.hidden = true;
    return;
  }

  const snapshots = (await opfs.listSnapshots()).filter(s => s.id !== currentSnapshotId);
  panel.hidden = snapshots.length === 0;
  if (snapshots.length === 0) return;

  list.innerHTML = snapshots
    .map(
      s => `
      <button class="saved-item" type="button" data-restore="${escapeHtml(s.id)}">
        <span class="saved-item-icon">${s.schemaVersion === opfs.SNAPSHOT_SCHEMA_VERSION ? '📦' : '⚠️'}</span>
        <span class="saved-item-body">
          <span class="saved-item-name">${escapeHtml(s.label)}</span>
          <span class="saved-item-meta">${plural(s.documentCount, 'doc')} · ${opfs.formatBytes(s.bytes)} · ${formatWhen(s.updatedAt)}</span>
        </span>
        <span class="saved-item-action">Restore</span>
      </button>`
    )
    .join('');
}

function plural(n: number, word: string): string {
  return `${n.toLocaleString()} ${word}${n === 1 ? '' : 's'}`;
}

function formatWhen(timestamp: number): string {
  const seconds = Math.round((Date.now() - timestamp) / 1000);
  if (seconds < 60) return 'just now';
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return new Date(timestamp).toLocaleDateString();
}

/** Load a stored snapshot into the engine — no embedding calls. */
async function restoreSnapshot(id: string): Promise<void> {
  if (!engine) {
    setStatus('Connect first', true);
    return;
  }

  showLoadingBubble();
  try {
    const snapshot = await opfs.readSnapshot(id);
    if (!snapshot) {
      setStatus('That saved session could not be read', true);
      return;
    }

    engine.restoreSnapshot(snapshot.documents, snapshot.vectors, snapshot.meta.vectorDimension);

    // Continue writing into the snapshot we just restored.
    currentSnapshotId = snapshot.meta.id;
    currentSnapshotCreatedAt = snapshot.meta.createdAt;
    currentSnapshotRevision = snapshot.meta.revision;

    removeLoadingBubble();
    updateDocCount();
    setSaveState(`Saved · v${currentSnapshotRevision}`, false);
    await refreshSavedSessions();

    setStatus(`Restored ${snapshot.documents.length} ${snapshot.documents.length === 1 ? 'document' : 'documents'}`);
    addMessageToChat(
      'assistant',
      `Restored **${snapshot.documents.length.toLocaleString()} ${snapshot.documents.length === 1 ? 'document' : 'documents'}** from this device — no re-embedding needed. Ask me anything.`
    );
  } catch (error) {
    removeLoadingBubble();
    setStatus(`Restore failed: ${error}`, true);
  }
}

// --- Storage manager ------------------------------------------------------

async function openStorageManager(): Promise<void> {
  await renderStorageList();
  getElement<HTMLDivElement>('storage-modal').style.display = 'flex';
}

function closeStorageManager(): void {
  getElement<HTMLDivElement>('storage-modal').style.display = 'none';
}

async function renderStorageList(): Promise<void> {
  const list = getElement<HTMLElement>('storage-list');
  const summary = getElement<HTMLElement>('storage-summary');

  if (!opfs.isSupported()) {
    summary.textContent = 'This browser does not support OPFS storage.';
    list.innerHTML = '<p class="empty-state">Persistent storage unavailable.</p>';
    return;
  }

  const snapshots = await opfs.listSnapshots();
  const total = snapshots.reduce((sum, s) => sum + s.bytes, 0);
  summary.textContent = snapshots.length
    ? `${snapshots.length} saved ${snapshots.length === 1 ? 'session' : 'sessions'} · ${opfs.formatBytes(total)} used`
    : 'Nothing stored yet';

  if (snapshots.length === 0) {
    list.innerHTML = '<p class="empty-state">Nothing saved on this device yet.</p>';
    return;
  }

  list.innerHTML = snapshots
    .map(s => {
      const active = s.id === currentSnapshotId;
      const stale = s.schemaVersion !== opfs.SNAPSHOT_SCHEMA_VERSION;
      const sourceChips = Object.entries(s.sources)
        .map(([name, n]) => `<span class="chip">${escapeHtml(name)}: ${n}</span>`)
        .join('');

      return `
        <div class="storage-item">
          <div class="storage-item-icon">${stale ? '⚠️' : '📦'}</div>
          <div class="storage-item-body">
            <div class="storage-item-name">${escapeHtml(s.label)}${active ? ' · in use' : ''}</div>
            <div class="storage-item-meta">
              <span class="chip">v${s.revision}</span>
              <span class="chip">format v${s.schemaVersion}</span>
              <span>${plural(s.documentCount, 'doc')}</span>
              <span>${s.vectorDimension}-dim</span>
              <span>${opfs.formatBytes(s.bytes)}</span>
              <span>${formatWhen(s.updatedAt)}</span>
              ${sourceChips}
            </div>
          </div>
          <div class="storage-item-actions">
            ${
              stale || active
                ? ''
                : `<button class="secondary" data-restore="${escapeHtml(s.id)}">Restore</button>`
            }
            <button class="icon-btn" data-delete="${escapeHtml(s.id)}">Delete</button>
          </div>
        </div>`;
    })
    .join('');
}

async function deleteStoredSnapshot(id: string): Promise<void> {
  await opfs.deleteSnapshot(id);
  if (id === currentSnapshotId) {
    currentSnapshotId = null;
    currentSnapshotRevision = 0;
    getElement<HTMLElement>('save-state').hidden = true;
  }
  await renderStorageList();
  await refreshSavedSessions();
  setStatus('Deleted from this device');
}

async function clearAllStorage(): Promise<void> {
  if (!confirm('Delete every saved session from this device? This cannot be undone.')) return;
  await opfs.clearAll();
  currentSnapshotId = null;
  currentSnapshotRevision = 0;
  getElement<HTMLElement>('save-state').hidden = true;
  await renderStorageList();
  await refreshSavedSessions();
  setStatus('All device storage cleared');
}

// Document Manager Modal
function openDocManager(): void {
  renderDocList();
  getElement<HTMLDivElement>('doc-manager-modal').style.display = 'flex';
}

function closeDocManager(): void {
  getElement<HTMLDivElement>('doc-manager-modal').style.display = 'none';
}

function renderDocList(): void {
  const docList = getElement<HTMLDivElement>('doc-list');
  const docs = engine?.getDocuments() ?? [];

  if (docs.length === 0) {
    docList.innerHTML = '<p class="empty-state">No documents loaded yet.</p>';
    return;
  }

  docList.innerHTML = docs.map(doc => `
    <div class="doc-item" data-id="${doc.id}">
      <div class="doc-item-icon">
        <span class="badge badge-${doc.source}">${getSourceIcon(doc.source)}</span>
      </div>
      <div class="doc-item-info">
        <div class="doc-item-title">${escapeHtml(doc.title)}</div>
        <div class="doc-item-meta">
          <span class="badge badge-${doc.source}">${doc.source}</span>
          <span>${new Date(doc.addedAt).toLocaleTimeString()}</span>
        </div>
      </div>
      <button class="doc-item-remove" onclick="window.removeDocument(${doc.id})">Remove</button>
    </div>
  `).join('');
}

function getSourceIcon(source: string): string {
  switch (source) {
    case 'docs': return 'DOC';
    case 'url': return 'URL';
    case 'file': return 'FILE';
    default: return 'TXT';
  }
}

function removeDocument(id: number): void {
  if (engine?.removeDocument(id)) {
    renderDocList();
    updateDocCount();
    setStatus('Document removed');
  }
}

function clearAllDocuments(): void {
  if (!engine) return;

  if (confirm('Remove all documents from the knowledge base?')) {
    engine.clearDocuments();
    renderDocList();
    updateDocCount();
    setStatus('All documents cleared');
  }
}

// Expose removeDocument to window for onclick handlers
(window as unknown as { removeDocument: typeof removeDocument }).removeDocument = removeDocument;

function setupEventListeners(): void {
  getElement<HTMLButtonElement>('init-btn').addEventListener('click', initializeEngine);
  getElement<HTMLButtonElement>('change-key').addEventListener('click', changeKey);
  getElement<HTMLButtonElement>('source-docs').addEventListener('click', () => selectSource('docs'));
  getElement<HTMLButtonElement>('source-byo').addEventListener('click', () => selectSource('byo'));
  getElement<HTMLButtonElement>('load-docs').addEventListener('click', loadLatticeDBDocsHandler);
  getElement<HTMLButtonElement>('add-url').addEventListener('click', addFromUrl);
  getElement<HTMLButtonElement>('upload-btn').addEventListener('click', uploadFiles);
  getElement<HTMLInputElement>('file-input').addEventListener('change', handleFileUpload);
  getElement<HTMLButtonElement>('add-doc').addEventListener('click', addDocument);
  getElement<HTMLButtonElement>('send-btn').addEventListener('click', sendMessage);
  getElement<HTMLButtonElement>('clear-chat').addEventListener('click', clearChat);
  getElement<HTMLButtonElement>('manage-docs-btn').addEventListener('click', openDocManager);

  // --- OPFS storage manager ---
  getElement<HTMLButtonElement>('open-storage').addEventListener('click', () => {
    void openStorageManager();
  });
  getElement<HTMLButtonElement>('open-storage-inline').addEventListener('click', () => {
    void openStorageManager();
  });
  getElement<HTMLButtonElement>('close-storage').addEventListener('click', closeStorageManager);
  getElement<HTMLButtonElement>('clear-all-storage').addEventListener('click', () => {
    void clearAllStorage();
  });
  getElement<HTMLDivElement>('storage-modal').addEventListener('click', (e) => {
    if (e.target === e.currentTarget) closeStorageManager();
  });

  // Restore / delete are rendered dynamically, so delegate from the containers.
  getElement<HTMLElement>('saved-list').addEventListener('click', (e) => {
    const target = (e.target as HTMLElement).closest<HTMLElement>('[data-restore]');
    if (target?.dataset.restore) void restoreSnapshot(target.dataset.restore);
  });
  getElement<HTMLElement>('storage-list').addEventListener('click', (e) => {
    const el = e.target as HTMLElement;
    const restore = el.closest<HTMLElement>('[data-restore]');
    if (restore?.dataset.restore) {
      closeStorageManager();
      void restoreSnapshot(restore.dataset.restore);
      return;
    }
    const del = el.closest<HTMLElement>('[data-delete]');
    if (del?.dataset.delete) void deleteStoredSnapshot(del.dataset.delete);
  });
  getElement<HTMLButtonElement>('close-modal').addEventListener('click', closeDocManager);
  getElement<HTMLButtonElement>('clear-all-docs').addEventListener('click', clearAllDocuments);

  getElement<HTMLInputElement>('message-input').addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  getElement<HTMLInputElement>('url-input').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      addFromUrl();
    }
  });

  // Close modal on backdrop click
  getElement<HTMLDivElement>('doc-manager-modal').addEventListener('click', (e) => {
    if (e.target === e.currentTarget) {
      closeDocManager();
    }
  });

  // Restore API key from localStorage
  const savedKey = localStorage.getItem('openrouter-api-key');
  if (savedKey) {
    getElement<HTMLInputElement>('api-key').value = savedKey;
  }
}

document.addEventListener('DOMContentLoaded', setupEventListeners);
