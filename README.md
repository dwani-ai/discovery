Discovery - Agentic Intelligence Platform 

Try Demo :  [https://app.dwani.ai](https://app.dwani.ai)


[Agents Presentation](https://docs.google.com/presentation/d/e/2PACX-1vQnIJbTjX_2v9B5QmboQCv61FFCcnK7Wz0MMujsW3G_szlBNhtpVy-hH6Ao-jg5psjOYiHzYoqsT22g/pub?start=false&loop=false&delayms=3000)

- How to setup Discovery 
    - for server
        - [README.md](backend/README.md)

    - for client
        - [README.md](frontend/README.md)

    - for VLM
        -  [README.md](vlm/README.md)


- For GCP Setup- 
    - [GCP README](gcp-READMe.md)


 - Buy vs Build - [TCO Analysis](TCO-analysis.md)


-  Example 
    - [Real-estate Insurance Claim](examples/insurance/insurance-agent.md)

- Features
    - Source audit  - done
    - file delete option -  done
    - file generate option -  done
    - chat with all documents -  done
    - add page level citations - done

- To run locally
    - update the environment with your local vllm/llama-cpp IP/port 


- for models with 32K context windows 
        
```
    export DWANI_API_BASE_URL=vllm/llama.cpp/IP
    export MAX_CONTEXT_TOKENS=28000 
    export MAX_HISTORY_TOKENS=4000
```  
    
- Run Docker 

```
    docker compose -f docker-compose.yml up -d
```

```

┌─────────────────────────────────────────────────────────────────────────────┐
│                               CLIENT (Browser / App)                        │
│                                                                             │
│   • Upload PDF           • Chat with documents           • Download clean PDF │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼           HTTP / multipart
                         ┌───────────────────────────────┐
                         │          FastAPI Server       │
                         │   (uvicorn + ASGI)            │
                         └───────────────┬───────────────┘
                                         │
                ┌────────────────────────┼─────────────────────────────┐
                │                        │                             │
       ┌────────▼────────┐      ┌────────▼────────┐         ┌─────────▼─────────┐
       │   File Upload   │      │  /chat-with-    │         │  /files/{id}/pdf  │
       │   endpoint      │      │  document       │         │  (regenerate)     │
       └────────┬────────┘      └────────┬────────┘         └─────────┬─────────┘
                │                        │                             │
                │ BackgroundTasks        │                             │
                ▼                        ▼                             ▼
       ┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
       │ PDF → Images        │   │ Hybrid Retrieval    │   │ Text → Clean PDF    │
       │ (pdf2image)         │   │ (Chroma + BM25)     │   │ (fpdf + DejaVuSans) │
       └────────┬────────────┘   └────────┬────────────┘   └────────┬────────────┘
                │                        │                             │
                ▼                        ▼                             │
       ┌─────────────────────┐   ┌─────────────────────┐             │
       │ OCR per page        │   │ RRF Fusion          │             │
       │ (gemma3 vision)     │   │ → top chunks        │             │
       └────────┬────────────┘   └────────┬────────────┘             │
                │                        │                             │
                ▼                        ▼                             │
       ┌─────────────────────┐   ┌─────────────────────┐             │
       │ Store full text     │   │ Build context       │◄────────────┘
       │   → SQLite          │   │ (token limited)     │
       └────────┬────────────┘   └────────┬────────────┘
                │                        │
                ▼                        ▼
       ┌─────────────────────┐   ┌─────────────────────┐
       │ Chunk + Embed       │   │ LLM generation      │
       │   → ChromaDB        │   │ (gemma3 text-only)  │
       └─────────────────────┘   └────────┬────────────┘
                                           │
                                           ▼
                                 Answer + Sources +
                             Contradiction Warning (if any)
                                           │
                                           ▼
                                    Back to Client

```


----

1. Lane 1: Document Ingestion & OCR



```
Upload PDF
   ↓
Background Task
   ↓
pdf2image → list of PIL Images
   ↓ (parallel / sequential)
gemma3-vision OCR per page
   ↓
page_texts: List[str]
   ├─→ SQLite (FileRecord.extracted_text)
   └─→ chunk_text_with_pages() → ChromaDB (with page metadata)

```
----

2. Lane 2: Chat / RAG



```
User question + file_ids
   ↓
Hybrid search:
   ├─→ Chroma vector search (bge-small-en-v1.5)
   └─→ BM25 on top-20 vector results
        ↓
Reciprocal Rank Fusion (RRF)
        ↓
Top ~10–20 chunks (with page & file metadata)
        ↓
Context building (greedy token limit)
        ↓
System prompt + history (trimmed) + context
        ↓
gemma3 (text-only)
        ↓
Answer + Sources list + optional Contradiction warning
```