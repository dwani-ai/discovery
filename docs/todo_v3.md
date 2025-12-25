### Implemented Features (Current State of dwani.ai)

#### Core Document Processing
- **Multimodal OCR**: Page-by-page image-to-text extraction using vision model (gemma3)
- **Privacy-first**: All processing happens server-side with no external storage
- **Clean PDF regeneration**: Text-only regenerated PDFs using extracted content
- **Merged clean PDFs**: Combine multiple documents into one clean version

#### Advanced RAG System
- **High-quality embeddings**: Using `BAAI/bge-small-en-v1.5` (top-tier small embedding model)
- **Page-level chunking & metadata**: Chunks preserve original page ranges
- **Page-level citations**: Sources show exact "Page X" or "Pages X–Y" references
- **Clickable page links**: Citations link directly to `#page=X` in regenerated PDF (opens at correct page)
- **Top-20 retrieval**: Increased recall without relevance threshold filtering
- **Strict truthfulness**: System prompt enforces "I don't know" only when truly no evidence

#### User Experience
- **Multi-file upload** with queue and progress feedback
- **No duplicate entries** during upload (deduplicated local + server state)
- **Global Chat**: Ask across all completed documents
- **Conversation history**: Persistent chats saved in localStorage
- **Document table**: Select, chat, download, delete, merge actions
- **Always-visible action buttons** (Chat/Download)
- **Source excerpts with highlighting** of query terms

#### Backend Reliability
- Background processing with status tracking (pending → processing → completed/failed)
- ChromaDB vector store with per-file cleanup
- FastAPI with proper dependency injection and CORS

### Suggested Future Features (To Make a Best-in-Class RAG + Search System)

#### 1. In-App PDF Viewer with Highlighting
- Side-by-side or embedded PDF viewer
- Auto-scroll + highlight matching chunks when clicking a source
- Search within loaded documents with highlights

#### 2. Hybrid Search (Keyword + Semantic)
- Add BM25 keyword search alongside vector search
- Combine results (reciprocal rank fusion) for better recall on specific terms/numbers

#### 3. Multi-Document Aware Answers
- When chatting with multiple docs, explicitly cite which document each fact comes from
- Detect contradictions between documents and flag them

#### 4. Advanced Query Understanding
- Query rewriting/decomposition for complex questions
- Follow-up question context awareness
- Entity extraction and filtering (e.g., "show me clauses about payment terms")

#### 5. Document-Level Metadata
- Auto-extract title, author, date, type (contract, invoice, report)
- Filter/search documents by metadata

#### 6. Collections & Folders
- Organize documents into projects/folders
- Chat scoped to specific collections

#### 7. Export & Sharing
- Export chat conversations with citations
- Share document sets or chat sessions via link

#### 8. Performance & Accuracy Boosts
- Optional larger embedding model (`bge-large-en-v1.5`)
- Re-ranking step (e.g., using cohere-rerank or cross-encoder)
- Post-processing to merge adjacent chunks

#### 9. Search Within Document
- Dedicated "Search in [document]" mode with full-text + semantic results
- Highlight all matches in PDF preview

#### 10. Analytics & Insights
- Summarize document (key entities, dates, obligations)
- Timeline view for contracts with dates
- Risk/clause detection templates

#### 11. Accessibility & Polish
- Dark mode
- Mobile-responsive layout
- Drag-and-drop upload
- Keyboard shortcuts

With the current foundation (especially page-level citations + strong embeddings), adding **#1 (in-app viewer with highlights)** and **#2 (hybrid search)** would immediately make this one of the best private RAG systems available.
