import os
from .config import settings

from llama_index.core import Settings as LlamaSettings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama


def configure_llamaindex() -> None:
    # Embeddings selected independently of LLM
    if settings.embed_provider == "ollama":
        embed_model = OllamaEmbedding(
            model_name=settings.ollama_embed_model,
            base_url=settings.ollama_host,
            request_timeout=settings.ollama_request_timeout,
        )
    elif settings.embed_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not set but EMBED_PROVIDER=openai. Set OPENAI_API_KEY.")
        embed_model = OpenAIEmbedding(model=settings.openai_embed_model, api_key=settings.openai_api_key)
    else:
        raise RuntimeError(f"Unsupported EMBED_PROVIDER: {settings.embed_provider}")

    # LLM provider
    if settings.llm_provider == "ollama":
        llm = Ollama(
            model=settings.ollama_model,
            base_url=settings.ollama_host,
            request_timeout=settings.ollama_request_timeout,
            context_window=settings.ollama_num_ctx,
            temperature=settings.ollama_temperature,
        )
    elif settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not set but LLM_PROVIDER=openai. Set OPENAI_API_KEY or use LLM_PROVIDER=ollama.")
        llm = OpenAI(model=settings.openai_model, api_key=settings.openai_api_key)
    else:
        raise RuntimeError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")

    LlamaSettings.embed_model = embed_model
    LlamaSettings.llm = llm
