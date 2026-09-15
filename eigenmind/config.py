"""Single source of truth for constants and env-based secrets.

All other modules read from here instead of redefining their own constants.
Secrets (NEBIUS_API_KEY, ...) are read from environment variables only — never hard-coded.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(override=False)


# ── Embedding ──
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
EMBEDDING_DIM_DEFAULT = 768

# ── Chunking ──
BATCH_SIZE = 100
# Formats handled by the ChunkNorris pipeline: each is parsed to markdown then chunked.
# PDF via PdfParser, DOCX via DocxParser, XLSX via ExcelParser, CSV via CSVParser,
# PPTX via MarkItDown, TXT/MD treated as markdown.
CHUNKNORRIS_EXTENSIONS = (".pdf", ".docx", ".xlsx", ".csv", ".pptx", ".txt", ".md")
# Formats we recognise but do not ingest yet — reported with a "not supported" notice.
UNSUPPORTED_EXTENSIONS = (".json",)
# Global toggle for OCR (Tesseract).
ENABLE_OCR = os.getenv("ENABLE_OCR", "false").lower() == "true"

# ── Graph exploration ──
MAX_CHUNKS = 100
MAX_CHUNKS_FOR_CONTEXT = 30
NEIGHBORS_TO_FETCH = 5
SIMILARITY_THRESHOLD = 0.65

# ── Theta (conflict-graph) parameters ──
THETA_CONFLICT_THRESHOLD = SIMILARITY_THRESHOLD
THETA_RANK = 24
THETA_MAX_ITERS = 400
THETA_STEP0 = 0.25
THETA_DIAG_SHIFT = 1e-2

# ── Chat-page LLM backend ──
# The /ask/ page can answer either through the Nebius / AI Hub cloud API or a
# local Ollama server. The provider is selected with the LLM_PROVIDER env var.
DEFAULT_LLM_PROVIDER = "nebius"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
NEBIUS_MODELS = (
    "meta-llama/Llama-3.3-70B-Instruct",
    "moonshotai/Kimi-K2.5-fast",
    "openai/gpt-oss-120b",
)

# ── Visualization ──
PROMPT_NODE_COLOR = "#FF0000"
PROMPT_NODE_SIZE = 22
NODE_MIN_SIZE = 6
NODE_MAX_SIZE = 20
DOCUMENT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]
METHOD_COLORS = {
    "Singular": "#FFD700",
    "Hinge":    "#32CD32",
    "Theta":    "#1E90FF",
}

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
USER_DATA_DIR = PROJECT_ROOT / "user_data"
DOWNLOADED_CORPUS_DIR = PROJECT_ROOT / "downloaded_corpus"
TEMP_GRAPH_OUTPUTS = PROJECT_ROOT / "temp_graph_outputs"


def qdrant_host() -> str:
    return os.getenv("QDRANT_HOST", "localhost")


def qdrant_port() -> int:
    return int(os.getenv("QDRANT_PORT", "6333"))


def nebius_api_key() -> str:
    """Nebius / AI Hub API key from env (or empty string if not set)."""
    return os.getenv("NEBIUS_API_KEY", "").strip()


def llm_provider() -> str:
    """Active LLM backend for the Chat page: 'ollama' (local) or 'nebius' (cloud).

    Defaults to 'nebius' so the original cloud behaviour is preserved when unset.
    """
    return os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip().lower() or DEFAULT_LLM_PROVIDER


def ollama_host() -> str:
    """Base URL of the local Ollama server (e.g. http://localhost:11434)."""
    return os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).strip() or DEFAULT_OLLAMA_HOST


def ollama_models() -> list[str]:
    """Optional comma-separated fallback model list (e.g. 'qwen2.5:7b,llama3.1').

    Used only when the Ollama server cannot be queried for its installed models.
    """
    raw = os.getenv("OLLAMA_MODELS", "").strip()
    return [m.strip() for m in raw.split(",") if m.strip()]


def sharepoint_credentials() -> tuple[str, str]:
    """SharePoint OAuth credentials from st.secrets if available, else env."""
    client_id = os.getenv("SHAREPOINT_CLIENT_ID", "").strip()
    client_secret = os.getenv("SHAREPOINT_CLIENT_SECRET", "").strip()
    try:
        import streamlit as st
        if "SHAREPOINT_CLIENT_ID" in st.secrets:
            client_id = st.secrets["SHAREPOINT_CLIENT_ID"]
        if "SHAREPOINT_CLIENT_SECRET" in st.secrets:
            client_secret = st.secrets["SHAREPOINT_CLIENT_SECRET"]
    except Exception:
        pass
    return client_id, client_secret


def ocr_available() -> bool:
    """OCR (ChunkNorris via PyMuPDF + Tesseract) is available when TESSDATA_PREFIX is set.

    PyMuPDF uses the integrated Tesseract engine and locates its ``*.traineddata`` files
    through this env var — no extra Python OCR packages are needed.
    """
    if not ENABLE_OCR:
        return False
    prefix = os.environ.get("TESSDATA_PREFIX")
    if not prefix:
        return False
    # ChunkNorris/PyMuPDF expects TESSDATA_PREFIX to point directly to the folder
    # containing .traineddata files. Some systems set it to the parent.
    if os.path.isdir(os.path.join(prefix, "tessdata")):
        os.environ["TESSDATA_PREFIX"] = os.path.join(prefix, "tessdata")
    return True


def get_ocr_languages() -> str:
    """Return a string of available Tesseract languages (e.g. 'eng+fra').

    Defaults to 'eng' if TESSDATA_PREFIX is not set or no data files are found.
    """
    if not ocr_available():
        return "eng"

    tessdata_dir = os.environ["TESSDATA_PREFIX"]
    try:
        langs = [f.split(".")[0] for f in os.listdir(tessdata_dir) if f.endswith(".traineddata")]
        # Exclude 'osd' (Orientation and Script Detection) which is not a language
        langs = [l for l in langs if l != "osd"]
        return "+".join(sorted(langs)) if langs else "eng"
    except Exception:
        return "eng"
