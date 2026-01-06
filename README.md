Discovery - Agentic Intelligence Platform 

Try Demo :  [https://app.dwani.ai](https://app.dwani.ai)



- How to setup Discovery 
    - for server
        - [README.md](backend/README.md)

    - for client
        - [README.md](frontend/README.md)

    - for VLM
        -  [README.md](vlm/README.md)


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