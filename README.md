## This repository is DEPRECATED. Go to [CogMaps](https://github.com/merlin-intelligence/cog-maps) for the new version. 

---

The software is licensed under MIT.

The Eigenmind and Merlin Intelligence names, logos and branding are not covered by the MIT license and may not be used without permission.
# Eigenmind - /accelerate clarity/

Eigenmind is a sophisticated knowledge management and exploration application built with Streamlit. It leverages a local Vector Database (Qdrant via Docker) and advanced AI models to ingest, map, and query your proprietary document corpus. LLM generation supports two interchangeable backends: **Nebius AI** (cloud, Llama / Kimi / OSS models) and **Ollama** (fully local, no data leaves the machine).

## Core Features

1. **Multi-Source Ingestion (`/add files to your corpus/`)**
   - Local directories (PDF, Word, Excel, CSV, PowerPoint, Text, Markdown) with OCR support — each format is parsed to markdown (PDF/DOCX/XLSX/CSV via ChunkNorris, PowerPoint via MarkItDown) and chunked uniformly by ChunkNorris. JSON is not supported yet.
   - Direct synchronization with Google Drive (OAuth or Service Account).
   - Direct synchronization with Microsoft SharePoint.
   - **Browser-resilient async ingestion**: jobs run in a background daemon thread decoupled from the browser session. Tab closure, network interruption, or browser timeout no longer kills the process. Job state (status, progress, logs) is persisted to SQLite (`user_data/jobs.db`) and the UI reconnects automatically to a running job on page reload.
   - **Smart Resume**: File-level resume feature using Qdrant filename tracking to skip already processed documents.
   - Local multilingual embedding (`intfloat/multilingual-e5-base`, 768-dim) via `sentence-transformers`, with the E5 `query:` / `passage:` prefixes applied automatically.

2. **Knowledge Graph Navigation (`/navigate your experience/`)**
   - Generates interactive subgraphs of knowledge relationships based on specific prompts.
   - Performs eigenvector/Laplacian analysis and identifies **Singular**, **Hinge**, and **Theta** nodes for unique, non-obvious insights.

3. **Advanced Question Answering (`/ask/`)**
   - Hybrid Retrieval-Augmented Generation (RAG) using both standard semantic similarity and singular chunk analysis.
   - **Nebius AI** backend: Llama 3.3, Kimi 2.5, OpenAI OSS 120B via cloud API (requires `NEBIUS_API_KEY`).
   - **Ollama** backend: any locally installed model (e.g. `qwen2.5:7b`), fully on-premise. Switch between backends from the sidebar toggle.

4. **Corpus Analysis (`/analyze corpus/`)**
   - Document inventory by type, character-length distribution, and a TF-IDF-stopworded wordcloud.
   - **Embedding-based topic modeling**: KMeans on the mean per-document embedding (reused from Qdrant — no re-encoding), `k` chosen by silhouette (cosine) over `[2..20]`. Cluster keywords are extracted *post-hoc* by TF-IDF so the partition stays semantic while the labels stay readable.
   - 2D PCA map coloured by cluster + pie-chart of cluster distribution.
   - Near-duplicate detection: only document pairs with cosine similarity ≥ 0.99 are surfaced, each flagged as **near-duplicate** and ranked by descending score.

## Stability & Performance

- **Shared Model Cache**: The embedding model is loaded once on first use for search and analysis pages, kept resident via `@st.cache_resource` — a single ~300 MB copy is reused across all sessions. The background ingestion runner manages its own scoped model per job (loaded at job start, released at job end).
- **CPU-First**: CPU-only PyTorch by default for broad compatibility and predictable memory usage; CUDA / MPS auto-detected when available.
- **Cold-Start Buffer**: On 4 GB-RAM VMs a 3.3 GB swap file is recommended to absorb the first-load spike (model download + load). See the [Deployment Guide](docs/DEPLOYMENT_GUIDE_EIGENMIND.md).
- **Persistent Storage**: All ingested data and vector embeddings are stored persistently in the `qdrant_storage/` directory. Job state and logs are persisted in `user_data/jobs.db`.

## Background

For an illustrated overview of the Cognitive Maps approach behind Eigenmind — what they are, why they matter, and how the app turns a corpus into navigable knowledge — see [Eigenmind — Cognitive Maps (PDF)](docs/260522_Eigenmind_Cognitive_Maps.pdf).

## Getting Started

Please refer to the [Architecture and Installation Guide](docs/ARCHITECTURE_AND_INSTALLATION_GUIDE.md) for detailed instructions on deploying the Qdrant database, installing dependencies, and configuring API keys.

## Production Deployment

For running Eigenmind as a long-lived service on a Google Cloud Compute Engine VM — including project creation, VM provisioning, firewall setup, `systemd` service installation, and the manual update procedure when the `main` branch evolves — see the [Eigenmind Production Deployment Guide](docs/DEPLOYMENT_GUIDE_EIGENMIND.md).

## Project layout (v0.2)

```
eigenmind/                  importable package
├── config.py               single source of truth for constants and env-based secrets
├── core/                   embeddings, LLM, chunking, document loaders, corpus analysis
├── vectordb/               Qdrant client, ingestion, retrieval helpers
├── graph/                  graph algorithms (singular, connectivity, theta, exploration)
├── connectors/             Google Drive, SharePoint
├── pipelines/              orchestration: ingest, rag
├── jobs/                   async ingestion job system (SQLite store + daemon runner)
└── ui/                     Streamlit-only code (auth, components, styles)

streamlit_app.py            entry point — `streamlit run streamlit_app.py`
pages/                      Streamlit multipage navigation (Eigenmind Cognitive Maps, /enrich corpus/, /analyze corpus/, /ask/, /explore graphs/, /manage/)
scripts/ingest_recursive.py CLI ingestion (also exposed as `eigenmind-ingest`)
tests/unit/                 pure-numpy graph-math tests
legacy/                     pre-refactor flat scripts, kept for reference
```

Run the app:

```bash
pip install -e .
streamlit run streamlit_app.py
```

CLI ingestion:

```bash
eigenmind-ingest directories.txt my_collection --device cpu
```

Secrets come from `.env` or `.streamlit/secrets.toml` (see `.env.example`).
The HuggingFace token is **never** hard-coded — set `HF_TOKEN` in your env if
you need a gated local LLM.

## Contributing

Contributions are welcome! Please read the [Contributing Guide](docs/CONTRIBUTING.md) before opening an issue or a pull request — it covers the workflow, naming conventions, and a few basic rules that keep collaboration smooth.

---
© 2026 Merlin Intelligence
