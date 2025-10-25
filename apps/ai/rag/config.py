from dataclasses import dataclass
import os
from pathlib import Path
import json
from .utils import ensure_env_loaded
try:
    import yaml
except Exception:
    yaml = None

ensure_env_loaded()

def _load_yaml_config() -> dict:
    ai_root = Path(__file__).resolve().parents[1]
    yml = ai_root / "config.yaml"
    if not yml.exists():
        return {}
    try:
        if yaml is not None:
            return yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
        # simple fallback for very small YAML subset using json if possible
        return json.loads(yml.read_text(encoding="utf-8"))
    except Exception:
        return {}

_cfg = _load_yaml_config()


@dataclass
class Settings:
    # Providers
    llm_provider: str = ((_cfg.get("providers", {}) or {}).get("llm_provider", "openai")).lower()
    embed_provider: str = ((_cfg.get("providers", {}) or {}).get("embed_provider", "openai")).lower()

    # API Key
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")

    # COnfig & settings
    openai_model: str = (_cfg.get("models", {}) or {}).get("openai_model", "gpt-4o-mini")
    openai_embed_model: str = (_cfg.get("models", {}) or {}).get("openai_embed_model", "text-embedding-3-small")

    ollama_model: str = (_cfg.get("models", {}) or {}).get("ollama_model", "gemma3:4b")
    ollama_embed_model: str = (_cfg.get("models", {}) or {}).get("ollama_embed_model", "nomic-embed-text")

    chroma_path: str = (_cfg.get("paths", {}) or {}).get("chroma_path", os.path.join("storage", "chroma"))
    data_dir: str = (_cfg.get("paths", {}) or {}).get("data_dir", "data")
    index_name: str = (_cfg.get("paths", {}) or {}).get("index_name", "default")

    # Prompt
    _prompt_path: str | None = (_cfg.get("prompt", {}) or {}).get("path")
    _prompt_text: str | None = (_cfg.get("prompt", {}) or {}).get("text")

    # Ollama knobs
    ollama_host: str = (_cfg.get("ollama", {}) or {}).get("host", "http://localhost:11434")
    ollama_request_timeout: float = float((_cfg.get("ollama", {}) or {}).get("request_timeout", 300))
    ollama_num_ctx: int = int((_cfg.get("ollama", {}) or {}).get("num_ctx", 4096))
    ollama_temperature: float = float((_cfg.get("ollama", {}) or {}).get("temperature", 0.2))

    # Chat knobs
    similarity_threshold: float = float((_cfg.get("chat", {}) or {}).get("similarity_threshold", 0.25))
    rag_top_k: int = int((_cfg.get("chat", {}) or {}).get("rag_top_k", 5))
    history_max_turns: int = int((_cfg.get("chat", {}) or {}).get("history_max_turns", 10))
    retry_on_timeouts: int = int((_cfg.get("chat", {}) or {}).get("retry_on_timeouts", 1))


settings = Settings()
