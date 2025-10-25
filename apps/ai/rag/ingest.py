import os
import hashlib
import json
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from .utils import ensure_env_loaded
ensure_env_loaded()

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import re

from .config import settings
from .llm_setup import configure_llamaindex

STATE_FILE = Path(__file__).resolve().parents[1] / "storage/.ingest_state.json"


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_state() -> dict[str, str]:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_state(state: dict[str, str]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def discover_files(data_dir: str) -> list[Path]:
    base = Path(__file__).resolve().parents[1]
    data_root = (base / data_dir).resolve()
    exts = {".txt", ".md", ".pdf", ".docx", ".csv", ".json"}
    paths: list[Path] = []
    for root, _, files in os.walk(data_root):
        for name in files:
            p = Path(root) / name
            if p.suffix.lower() in exts:
                paths.append(p)
    return sorted(paths)


def build_or_update_index() -> VectorStoreIndex:
    configure_llamaindex()

    # chroma client + persistent storage
    base = Path(__file__).resolve().parents[1]
    persist_dir = str((base / settings.chroma_path).resolve())
    os.makedirs(persist_dir, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=persist_dir)

    # distinct collection name keyed ONLY by embedding provider+model
    if settings.embed_provider == "ollama":
        embed_tag = settings.ollama_embed_model
        embed_prefix = "ollama"
    else:
        embed_tag = settings.openai_embed_model
        embed_prefix = "openai"
    safe_tag = re.sub(r"[^a-zA-Z0-9_.-]+", "-", embed_tag).lower()
    collection_name = f"{settings.index_name}-{embed_prefix}-{safe_tag}"

    collection = chroma_client.get_or_create_collection(collection_name)
    print(f"[ingest] Using Chroma collection: {collection_name}")
    vector_store = ChromaVectorStore(chroma_collection=collection)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # only re-load changed files
    paths = discover_files(settings.data_dir)
    prev = _load_state()
    # determine which files changed
    changed: list[Path] = []
    new_state: dict[str, str] = {}

    for p in paths:
        digest = _hash_file(p)
        new_state[str(p)] = digest
        if prev.get(str(p)) != digest:
            changed.append(p)

    need_full_rebuild = changed or set(prev.keys()) - {str(p) for p in paths} or (collection.count() == 0)
    if need_full_rebuild:
        # If changes or deletions, rebuild from all docs for simplicity & correctness
        print(f"[ingest] Rebuilding index from {len(paths)} files...")
        reader = SimpleDirectoryReader(input_files=paths)
        docs = reader.load_data(show_progress=True, num_workers=4)
        # This will lose history, but is required for correctness
        # Re-create collection
        chroma_client.delete_collection(collection_name)
        collection = chroma_client.get_or_create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            docs,
            storage_context=storage_context,
            show_progress=True,
        )
        _save_state(new_state)
        print("[ingest] Index rebuild complete.")
    else:
        print("[ingest] No file changes detected, loading existing index.")
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
        )

    return index


get_rag_index = build_or_update_index


def main() -> None:
    build_or_update_index()


if __name__ == "__main__":
    main()
