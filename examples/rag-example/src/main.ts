import { RAGEngine } from './rag';
import { loadLatticeDBDocs, fetchUrlContent, parseFile, getFileTitle, getUrlTitle } from './documents';
import { marked } from 'marked';
import type { Message } from './types';

// Configure marked for safe rendering
marked.setOptions({
  breaks: true,
  gfm: true
});

let engine: RAGEngine | null = null;
let chatHistory: Message[] = [];

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

function enableControls(): void {
  getElement<HTMLButtonElement>('load-docs').disabled = false;
  getElement<HTMLButtonElement>('add-url').disabled = false;
  getElement<HTMLButtonElement>('upload-btn').disabled = false;
  getElement<HTMLButtonElement>('add-doc').disabled = false;
  getElement<HTMLButtonElement>('send-btn').disabled = false;
  getElement<HTMLButtonElement>('manage-docs-btn').disabled = false;
  getElement<HTMLInputElement>('message-input').disabled = false;
  getElement<HTMLInputElement>('url-input').disabled = false;
  getElement<HTMLTextAreaElement>('doc-text').disabled = false;
}

async function initializeEngine(): Promise<void> {
  const apiKey = getElement<HTMLInputElement>('api-key').value.trim();

  if (!apiKey) {
    setStatus('Please enter your OpenRouter API key', true);
    return;
  }

  const initBtn = getElement<HTMLButtonElement>('init-btn');
  initBtn.disabled = true;
  initBtn.textContent = 'Initializing...';

  getElement<HTMLElement>('chat-section').style.display = 'flex';
  getElement<HTMLElement>('welcome-section').style.display = 'none';
  showLoadingBubble();

  try {
    engine = new RAGEngine({ apiKey });
    await engine.init();

    removeLoadingBubble();
    setStatus('LatticeDB initialized!');
    initBtn.textContent = 'Initialized';
    enableControls();
    localStorage.setItem('openrouter-api-key', apiKey);

    addMessageToChat('assistant', 'Welcome! Click **Load LatticeDB Docs** to load the API documentation, then ask me anything about LatticeDB. We will do so using LatticeDB itself, running **inside** your **own** browser!');
  } catch (error) {
    removeLoadingBubble();
    setStatus(`Initialization failed: ${error}`, true);
    initBtn.disabled = false;
    initBtn.textContent = 'Initialize';
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

    for (const chunk of chunks) {
      await engine.addDocument(
        { id: Math.floor(Date.now() + Math.random() * 1000), text: chunk.content, metadata: chunk.metadata },
        'docs',
        chunk.section
      );
    }

    removeLoadingBubble();
    setStatus(`Loaded ${chunks.length} documentation sections`);
    updateDocCount();

    addMessageToChat('assistant', `I've loaded ${chunks.length} documentation sections covering the LatticeDB API. You can now ask me about collections, points, vector search, graph operations, and more!`);
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
  getElement<HTMLButtonElement>('load-docs').addEventListener('click', loadLatticeDBDocsHandler);
  getElement<HTMLButtonElement>('add-url').addEventListener('click', addFromUrl);
  getElement<HTMLButtonElement>('upload-btn').addEventListener('click', uploadFiles);
  getElement<HTMLInputElement>('file-input').addEventListener('change', handleFileUpload);
  getElement<HTMLButtonElement>('add-doc').addEventListener('click', addDocument);
  getElement<HTMLButtonElement>('send-btn').addEventListener('click', sendMessage);
  getElement<HTMLButtonElement>('clear-chat').addEventListener('click', clearChat);
  getElement<HTMLButtonElement>('manage-docs-btn').addEventListener('click', openDocManager);
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
