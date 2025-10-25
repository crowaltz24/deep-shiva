import os
from llama_index.core import VectorStoreIndex, Settings as LlamaSettings, PromptTemplate
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.llms import ChatMessage, MessageRole

from .config import settings
from .llm_setup import configure_llamaindex
from .ingest import build_or_update_index
from .utils import get_weather_data_for_place, format_weather_response
from .utils import ensure_env_loaded

ensure_env_loaded()

import re


SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.25"))  # env tunable
HISTORY_MAX_TURNS = int(os.getenv("HISTORY_MAX_TURNS", "10"))  # how many prior turns to include for history
TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RETRY_ON_TIMEOUTS = int(os.getenv("RETRY_ON_TIMEOUTS", "1"))


def _load_system_prompt() -> str:
    base = BASE_DIR.parent  # ai/
    # config.yaml defines prompt.path
    conf_path = None
    try:
        from pathlib import Path as _P
        # relative to apps/ai
        conf_path = getattr(settings, "_prompt_path", None)
    except Exception:
        pass
    path = os.getenv("SYSTEM_PROMPT_PATH", conf_path)
    prompt_file = Path(path) if path else (base / "system_prompt.txt")
    if prompt_file.exists():
        try:
            return prompt_file.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    # fallback to env inline prompt
    inline = os.getenv("SYSTEM_PROMPT")
    if inline:
        return inline.strip()
    # default system prompt
    return (
        "You are a concise, helpful tourism assistant. Prefer short, accurate answers. "
        "When context is provided, ground your answer in it and avoid fabricating details. "
        "If context is insufficient, still answer from your general knowledge without citing sources."
    )


def _format_history(history: list[tuple[str, str]], max_turns: int) -> str:
    if not history:
        return ""
    chunk = history[-max_turns:]
    lines = []
    for u, a in chunk:
        lines.append(f"User: {u}")
        lines.append(f"Assistant: {a}")
    return "\n".join(lines)


def interactive_chat(index: VectorStoreIndex) -> None:
    print("RAG Chat. Type 'exit' to quit.")
    system_prompt = _load_system_prompt()
    text_qa_template = PromptTemplate(
        (
            f"{system_prompt}\n\n"
            "Given the following context, answer the user's question.\n"
            "- If context is not relevant, say so briefly or answer succinctly from general knowledge.\n\n"
            "Context:\n{context_str}\n\n"
            "Question: {query_str}\n\n"
            "Answer:"
        )
    )
    query_engine = index.as_query_engine(
        streaming=False,
        similarity_top_k=TOP_K,
        text_qa_template=text_qa_template,
    )
    history: list[tuple[str, str]] = []
    last_place: str | None = None
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit", ":q"}:
            break
        # weather command
        if q.startswith("/weather "):
            place = q[len("/weather ") :].strip()
            if place:
                # fetch structured data, then ask LLM to summarize concisely
                try:
                    disp, wx = get_weather_data_for_place(place)
                    llm = LlamaSettings.llm
                    if llm is not None:
                        prompt = (
                            "Summarize this weather data in 3-6 concise sentences suitable for a tourist. Only return the summary, nothing else."
                            "Include today’s conditions briefly, past conditions if relevant and a compact 7-day outlook with temps, rain risk, and wind.\n\n"
                            f"Location: {disp}\n\nData (JSON):\n{wx}\n\nSummary:"
                        )
                        resp = llm.complete(prompt)
                        msg = getattr(resp, 'text', str(resp))
                    else:
                        msg = format_weather_response(disp, wx)
                    print(msg)
                    last_place = disp
                    history.append((q, msg))
                    continue
                except Exception as e:
                    print(f"[weather error] {e}")
                # don't store command in history
                continue
        # intent: weather queries (auto tool call)
        low = q.lower()
        if any(w in low for w in [
            "weather", "forecast", "temperature", "temp", "rain", "raining", "climate",
            "cold", "hot", "chilly", "warm", "heat", "humid", "humidity", "windy", "wind", "storm", "sunny",
            "how hot", "how cold", "how warm", "how chilly",
        ]):
            # parse days (default to 7)
            days = 7
            m_days = re.search(r"next\s+(\d{1,2})\s+day", low) or re.search(r"(\d{1,2})-day", low) or re.search(r"for\s+(\d{1,2})\s+days", low)
            if m_days:
                try:
                    days = max(1, min(14, int(m_days.group(1))))
                except Exception:
                    pass
            elif "tomorrow" in low:
                days = 2
            elif "today" in low:
                days = 1
            elif "tonight" in low or "this evening" in low or "this morning" in low:
                days = 1
            elif "this week" in low or "next week" in low:
                days = 7
            elif "weekend" in low:
                days = 3

            # parse place: look for "in <place>" or "for <place>" phrases
            place = None
            m_in = re.search(r"\b(?:in|for)\s+([a-zA-Z ,.-]{2,})", q)
            if m_in:
                candidate = m_in.group(1).strip().rstrip("?.! ")
                # trim trailing day qualifiers
                candidate = re.sub(r"\b(next\s+\d+\s+days?|today|tomorrow)\b", "", candidate, flags=re.IGNORECASE).strip(", .-")
                if candidate:
                    place = candidate
            # default to last mentioned place, else Dehradun
            if not place:
                place = last_place or "Dehradun"

            try:
                disp, wx = get_weather_data_for_place(place, days=days)
                llm = LlamaSettings.llm
                if llm is not None:
                    prompt = (
                        "Summarize this weather data in 3-6 concise sentences suitable for a tourist. "
                        "Include today’s conditions briefly, past conditions if relevant and a compact 7-day outlook with temps, rain risk, and wind.\n\n"
                        f"Location: {disp}\n\nData (JSON):\n{wx}\n\nSummary:"
                    )
                    resp = llm.complete(prompt)
                    weather = getattr(resp, 'text', str(resp))
                else:
                    weather = format_weather_response(disp, wx)
                print(weather)
                last_place = disp
                history.append((q, weather))
                continue
            except Exception as e:
                print(f"[weather error] {e}")
                # fall through to RAG
        try:
            # query with recent chat history
            hist_str = _format_history(history, HISTORY_MAX_TURNS)
            query_input = (
                f"Conversation so far:\n{hist_str}\n\nUser question: {q}" if hist_str else q
            )
            # basic retry
            attempt = 0
            while True:
                try:
                    resp = query_engine.query(query_input)
                    break
                except Exception as e:
                    if "timed out" in str(e).lower() and attempt < RETRY_ON_TIMEOUTS:
                        attempt += 1
                        continue
                    raise
            answer = getattr(resp, 'response', getattr(resp, 'text', str(resp)))
            # retrieval had useful context? if not, fallback to base LLM
            source_nodes = getattr(resp, 'source_nodes', None) or []
            max_score = max((sn.score or 0.0) for sn in source_nodes) if source_nodes else 0.0
            if not source_nodes or max_score < SIMILARITY_THRESHOLD:
                # Fallback
                llm = LlamaSettings.llm
                if llm is not None:
                    try:
                        direct_prompt = (
                            f"{system_prompt}\n\n"
                            f"Conversation so far:\n{hist_str}\n\n" if hist_str else f"{system_prompt}\n\n"
                        ) + f"User question: {q}\n\nAnswer:"
                        direct = llm.complete(direct_prompt)
                        answer = getattr(direct, 'text', str(direct))
                    except Exception:
                        # keep our original answer if direct call fails
                        pass
            print(f"Assistant: {answer}\n")
            # save turn to history
            history.append((q, answer))
        except Exception as e:
            print(f"[error] {e}")


# def main() -> None:
#     configure_llamaindex()
#     index = build_or_update_index()
#     interactive_chat(index)


# if __name__ == "__main__":
#     main()

# apps/ai/rag/chat.py

def main() -> None:
    configure_llamaindex()
    index = build_or_update_index()
    interactive_chat(index)


if __name__ == "__main__":
    main()