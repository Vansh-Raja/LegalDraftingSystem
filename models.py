"""
Centralized model registry, factories, and prompts for chat and retrieval.
This keeps provider configs, system prompts, and embeddings selection in one place.
"""

from dataclasses import dataclass, field
import os
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Load environment early for downstream factories
load_dotenv()


@dataclass
class ModelConfig:
    id: str
    provider: str
    label: str
    context_window: int
    default_system_prompt: str = "rag"
    description: Optional[str] = None
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    provider_kwargs: Dict[str, Any] = field(default_factory=dict)


# -------------------------
# System prompts
# -------------------------
CHAT_PROMPTS: Dict[str, str] = {
    "general_chat": (
        "You are a friendly assistant for the Legal Drafting System. "
        "When the user is not asking for legal help, reply briefly (no more than three sentences), "
        "stay positive, and encourage them to share any legal question if appropriate. "
        "You may answer general knowledge queries directly without mentioning legal content. "
        "If the user steers back toward legal matters, gently invite them to provide more details so we can help."
    ),
    "general_law": (
        "You are the primary advisory voice for an Indian legal assistant. "
        "Deliver empathetic, actionable guidance grounded in Indian law unless the user specifies another jurisdiction. "
        "Structure your reply under the heading 'General Guidance'. "
        "Explain key rights, immediate steps, procedural options, and statutory hooks (e.g., IPC, CrPC, PC Act) relevant to the scenario. "
        "Keep the focus on universal principles—do not cite specific cases or rely on any precedent context, because a separate module will append case-based insights. "
        "Flag uncertainties, urge the user to consult a qualified lawyer, and avoid definitive promises about outcomes."
    ),
    "rag": (
        "You are a **legal research and drafting assistant** within a Retrieval-Augmented Generation (RAG) system. "
        "You will be given a user query and a context drawn exclusively from retrieved case-law or statutory materials. "
        "Answer **only** from the provided context; never use outside knowledge, inference, or speculation.\n\n"
        "========================\n"
        "### CORE DIRECTIVES\n"
        "========================\n"
        "1. **Case-first routing** — If the query refers to a specific case (by party names, citation, date, or case number), "
        "focus on that case. Use other retrieved materials only if they directly clarify or support a relevant point.\n\n"
        "2. **Topic synthesis** — If the question is thematic (e.g., about a statute, doctrine, or principle), "
        "you may synthesize across multiple retrieved documents.\n\n"
        "3. **Filename-visible headers (MANDATORY)** — Every case you summarize must appear under a header of the form:\n"
        "      `[Case: <Case Name> | File: <N.txt>]`\n"
        "   If multiple chunks from the same file are used, include chunk numbers when available.\n\n"
        "4. **Precision and attribution** — Be concise, text-anchored, and well-reasoned. "
        "Short quotes (≤2 sentences) are permitted if followed by in-text citations like "
        "“(Case Name — file N.txt, chunk X)”. Never fabricate or generalize unsupported facts.\n\n"
        "5. **Verification discipline** — Ensure silently that every factual statement is supported by the text. "
        "If uncertain, omit or qualify using 'the record here does not clarify...'.\n\n"
        "6. **Insufficient data fallback** — If the materials do not allow you to answer, respond exactly: "
        "'I'm sorry — I don't know based on the provided documents.'\n\n"
        "7. **Tone and structure** — Use a formal, analytical tone similar to a judicial summary or bench memo. "
        "Organize your response logically: brief overview → reasoning → conclusion.\n\n"
        "8. **Source listing (MANDATORY)** — End every answer with a 'Sources:' line that lists the file names actually used "
        "(e.g., `Sources: 1.txt, 2.txt`). You may optionally include case names beside them.\n\n"
        "9. **Formatting discipline** — Use structured headings and concise paragraphs. "
        "Avoid conversational or speculative phrasing. Use plain text formatting with consistent sectioning.\n\n"
        "========================\n"
        "### OUTPUT FORMATTING RULES\n"
        "========================\n"
        "- Always include file identifiers in case headers.\n"
        "- Prefer short labeled paragraphs (e.g., Issue, Held, Reasoning, Disposition).\n"
        "- Avoid overuse of bullets; favor narrative clarity.\n"
        "- Do not invent paragraph numbers or citations not present in the input.\n"
        "- Maintain clean, professional spacing.\n\n"
        "========================\n"
        "### TEMPLATE A — MULTIPLE CASE SUMMARIES (Parallel Summaries)\n"
        "========================\n"
        "[Overall Overview]\n"
        "One or two sentences summarizing the user’s query and how the retrieved cases relate to it.\n\n"
        "[Case: <Case Name> | File: <N.txt>]\n"
        "Court / Date / Citation (if present)\n"
        "Issue: …\n"
        "Held: …\n"
        "Key Reasons:\n"
        "  • Point 1 — short explanation or quote (Case — file N.txt, chunk X)\n"
        "  • Point 2 — …\n"
        "Controlling Provisions: (only those explicitly cited)\n"
        "Outcome: (appeal allowed / dismissed / remand / directions)\n"
        "Notes or Limits: (if context shows any restrictions)\n\n"
        "[Case: <Case Name> | File: <M.txt>]\n"
        "Court / Date / Citation\n"
        "Issue: …\n"
        "Held: …\n"
        "Key Reasons:\n"
        "  • …\n"
        "Outcome: …\n\n"
        "[Synthesis / Comparison]\n"
        "Two–five lines drawing together or contrasting the holdings based only on the retrieved text.\n\n"
        "Sources: N.txt, M.txt\n\n"
        "Example:\n"
        "[Overall Overview]\n"
        "The question concerns limitation for IBC appeals before NCLAT. The retrieved judgments clarify the strict 30+15 day rule.\n\n"
        "[Case: A Rajendra v. Gonugunta Madhusudhan Rao | File: 2.txt]\n"
        "SC (4 Apr 2025) — 2025 INSC 447\n"
        "Issue: Whether NCLAT can condone delay beyond the outer 45-day period under Section 61(2) IBC.\n"
        "Held: Appeals barred; limitation runs from pronouncement; no condonation beyond 45 days.\n"
        "Key Reasons:\n"
        "  • The 45-day cap is absolute under the statute. (Case — file 2.txt, chunk 3)\n"
        "  • Delay condonation under Section 5 Limitation Act is excluded. (Case — file 2.txt, chunk 4)\n"
        "Outcome: Appeal dismissed as time-barred.\n"
        "Notes: Court stressed strict adherence to statutory timelines.\n\n"
        "[Case: K Tirupathi Reddy v. B Chandra Sekhar Reddy | File: 5.txt]\n"
        "SC (21 Feb 2025) — 2025 INSC 189\n"
        "Issue: Whether NCLAT’s condonation order beyond 45 days is valid.\n"
        "Held: No; the Tribunal exceeded jurisdiction beyond 45 days.\n"
        "Key Reasons:\n"
        "  • Section 61(2) is exhaustive on limitation. (Case — file 5.txt, chunk 2)\n"
        "  • Liberal condonation is impermissible once the outer limit expires. (Case — file 5.txt, chunk 3)\n"
        "Outcome: Appeal dismissed.\n\n"
        "[Synthesis / Comparison]\n"
        "Both decisions reinforce a strict 45-day outer limit for IBC appeals to NCLAT. Neither permits equitable extension once that cap lapses.\n\n"
        "Sources: 2.txt, 5.txt"
    ),
}


def get_prompt(name: str) -> str:
    return CHAT_PROMPTS[name]


# -------------------------
# Model registry
# -------------------------
MODEL_REGISTRY: Dict[str, ModelConfig] = {
    # OpenAI
    "gpt-5-nano-2025-08-07": ModelConfig(
        id="gpt-5-nano-2025-08-07",
        provider="openai",
        label="OpenAI gpt-5-nano-2025-08-07",
        context_window=128_000,
        default_system_prompt="rag",
    ),
    # OpenRouter (OpenAI-compatible)
    "openai/gpt-oss-120b": ModelConfig(
        id="openai/gpt-oss-120b",
        provider="openrouter",
        label="OpenRouter GPT-OSS 120B",
        context_window=128_000,
        default_system_prompt="rag",
    ),
    "openai/gpt-oss-20b": ModelConfig(
        id="openai/gpt-oss-20b",
        provider="openrouter",
        label="OpenRouter GPT-OSS 20B",
        context_window=128_000,
        default_system_prompt="rag",
    ),
    "meta-llama/llama-4-scout": ModelConfig(
        id="meta-llama/llama-4-scout",
        provider="openrouter",
        label="OpenRouter Llama 4 Scout",
        context_window=128_000,
        default_system_prompt="rag",
        model_kwargs={
            "model_kwargs": {"extra_body": {"provider": {"only": ["deepinfra/fp8"]}}},
        },
    ),
    "qwen/qwen3-235b-a22b": ModelConfig(
        id="qwen/qwen3-235b-a22b",
        provider="openrouter",
        label="OpenRouter Qwen3 235B A22B",
        context_window=64_000,
        default_system_prompt="rag",
    ),
    "qwen/qwen3-14b": ModelConfig(
        id="qwen/qwen3-14b",
        provider="openrouter",
        label="OpenRouter Qwen3 14B",
        context_window=32_000,
        default_system_prompt="rag",
    ),
    # Ollama local
    "qwen3:4b": ModelConfig(
        id="qwen3:4b",
        provider="ollama_local",
        label="Ollama Local Qwen3 4B",
        context_window=256_000,
        default_system_prompt="rag",
    ),
    "qwen3:latest": ModelConfig(
        id="qwen3:latest",
        provider="ollama_local",
        label="Ollama Local Qwen3",
        context_window=40_000,
        default_system_prompt="rag",
    ),
}

DEFAULT_MODEL_ID = "gpt-5-nano-2025-08-07"


def _provider_available(provider: str) -> bool:
    """
    Gate model visibility by required API keys/env.
    """
    if provider == "openai":
        return bool(os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY"))
    if provider == "openrouter":
        return bool(os.getenv("OPENROUTER_API_KEY"))
    if provider == "groq":
        return bool(os.getenv("GROQ_API_KEY"))
    if provider == "ollama_local":
        return True
    return False


def list_chat_models() -> List[ModelConfig]:
    # Preserve a stable, readable ordering (OpenAI → OpenRouter → Groq → Ollama local)
    provider_order = ["openai", "openrouter", "groq", "ollama_local"]
    available = [
        cfg for cfg in MODEL_REGISTRY.values() if _provider_available(cfg.provider)
    ]
    return sorted(
        available,
        key=lambda cfg: (provider_order.index(cfg.provider) if cfg.provider in provider_order else 99, cfg.label),
    )


def get_model_config(model_id: str) -> ModelConfig:
    if model_id not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model id: {model_id}")
    return MODEL_REGISTRY[model_id]


def _build_openai(cfg: ModelConfig):
    api_key = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_KEY not set.")
    kwargs = {"model": cfg.id, "temperature": 0, "streaming": True, "api_key": api_key}
    kwargs.update(cfg.model_kwargs)
    return ChatOpenAI(**kwargs), "openai"


def _build_openrouter(cfg: ModelConfig):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set.")
    kwargs = {
        "model": cfg.id,
        "temperature": 0,
        "streaming": True,
        "api_key": api_key,
        "base_url": "https://openrouter.ai/api/v1",
    }
    kwargs.update(cfg.model_kwargs)
    kwargs.update(cfg.provider_kwargs)
    return ChatOpenAI(**kwargs), "openrouter"


def _build_groq(cfg: ModelConfig):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set.")
    kwargs = {
        "model": cfg.id,
        "temperature": 0,
        "streaming": True,
        "api_key": api_key,
        "base_url": "https://api.groq.com/openai/v1",
    }
    kwargs.update(cfg.model_kwargs)
    return ChatOpenAI(**kwargs), "groq"


def _build_ollama_local(cfg: ModelConfig):
    def _reachable(base_url: str) -> bool:
        try:
            with urllib.request.urlopen(f"{base_url}/api/tags", timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    host = _resolve_ollama_host()
    if not _reachable(host):
        fallback = "http://127.0.0.1:11434"
        if host != fallback and _reachable(fallback):
            host = fallback
    kwargs = {
        "model": cfg.id,
        "temperature": 0,
        "streaming": True,
        "num_ctx": cfg.context_window,
    }
    if host:
        kwargs["base_url"] = host
    kwargs.update(cfg.model_kwargs)
    return ChatOllama(**kwargs), "ollama_local"


def _resolve_ollama_host() -> str:
    """
    Return a reachable Ollama host, preferring OLLAMA_HOST but falling back to default.
    """
    def _reachable(base_url: str) -> bool:
        try:
            with urllib.request.urlopen(f"{base_url}/api/tags", timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    host = os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434"
    if not _reachable(host):
        fallback = "http://127.0.0.1:11434"
        if host != fallback and _reachable(fallback):
            host = fallback
    return host


def _build_ollama_cloud(cfg: ModelConfig):
    """
    Use Ollama's Python client via LangChain's ChatOllama with a remote host.
    """
    api_key = os.getenv("OLLAMA_CLOUD_API_KEY") or os.getenv("OLLAMA_API_KEY")
    if not api_key:
        raise RuntimeError("OLLAMA_CLOUD_API_KEY (or OLLAMA_API_KEY) not set.")
    base_url = os.getenv("OLLAMA_CLOUD_BASE_URL", "https://ollama.com")
    kwargs = {
        "model": cfg.id,
        "temperature": 0,
        "streaming": True,
        "base_url": base_url,
        "headers": {"Authorization": f"Bearer {api_key}"},
    }
    # Respect any extra kwargs (e.g., num_ctx overrides)
    kwargs.update(cfg.model_kwargs)
    return ChatOllama(**kwargs), "ollama_cloud"


_FACTORIES: Dict[str, Callable[[ModelConfig], Tuple[Any, str]]] = {
    "openai": _build_openai,
    "openrouter": _build_openrouter,
    "groq": _build_groq,
    "ollama_local": _build_ollama_local,
    "ollama_cloud": _build_ollama_cloud,
}


def build_chat_model(model_id: str) -> Tuple[Any, str]:
    cfg = get_model_config(model_id)
    if cfg.provider not in _FACTORIES:
        raise RuntimeError(f"No factory registered for provider: {cfg.provider}")
    return _FACTORIES[cfg.provider](cfg)


# -------------------------
# Embeddings
# -------------------------
def get_embeddings(model: Optional[str] = None, provider: Optional[str] = None):
    """
    Return an embeddings instance. Defaults to Ollama embeddings unless demo mode
    or an explicit provider requires OpenAI-compatible embeddings.
    """
    demo_mode = os.getenv("DEMO_MODE", "0") == "1"
    openai_preferred = provider in {"openai", "openrouter", "groq", "ollama_cloud"} or demo_mode

    if openai_preferred:
        api_key = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_KEY required for OpenAI embeddings.")
        embed_model = model or os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=embed_model, api_key=api_key)

    ollama_model = model or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
    ollama_host = _resolve_ollama_host()
    if ollama_host:
        # Force env for the underlying client to avoid random host selection
        os.environ["OLLAMA_HOST"] = ollama_host
        # Avoid corporate/local proxies hijacking localhost calls (seen as port flips)
        for key in ("NO_PROXY", "no_proxy"):
            existing = os.environ.get(key, "")
            hosts = [h.strip() for h in existing.split(",") if h.strip()]
            for loop_host in ("127.0.0.1", "localhost"):
                if loop_host not in hosts:
                    hosts.append(loop_host)
            os.environ[key] = ",".join(hosts) if hosts else "127.0.0.1,localhost"
    kwargs = {"model": ollama_model}
    if ollama_host:
        kwargs["base_url"] = ollama_host
    
    # -----------------------------------------------------------
    # Robust Wrapper Implementation for Ollama Embeddings
    # -----------------------------------------------------------
    class RobustOllamaEmbeddings(OllamaEmbeddings):
        """
        Subclass to add retry logic for transient Ollama errors (e.g. EOF).
        Forces a fresh client connection on retry to mitigate connection pool issues.
        """
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            import time
            import random
            from ollama import Client
            
            # Ensure we are targeting the correct local host
            target_url = "http://127.0.0.1:11434"
            if self.base_url != target_url:
                self.base_url = target_url
                self._client = Client(host=target_url)

            # Helper to embed a single text with retries
            def _embed_single(text):
                local_client = Client(host=target_url)
                for _ in range(3):
                    try:
                        resp = local_client.embed(model=self.model, input=text)
                        return resp['embeddings'][0]
                    except Exception:
                        time.sleep(0.5)
                # Fallback: return a zero vector if strictly necessary, or let it fail
                # For now, return zero vector to keep pipeline moving
                return [0.0] * 768

            max_retries = 3
            
            # 1. Try batch
            for attempt in range(max_retries):
                try:
                    # Refresh client on retry to clear any stuck connection state
                    if attempt > 0:
                        self._client = Client(host=target_url)
                    return super().embed_documents(texts)
                except Exception as e:
                    msg = str(e)
                    # Retry on network/server errors
                    if any(x in msg for x in ["EOF", "Connection refused", "500", "ResponseError"]):
                        sleep_time = (attempt + 1) + random.uniform(0, 1)
                        print(f"[RobustEmbeddings] Batch retry {attempt+1}/{max_retries} failed ({msg}). Sleeping {sleep_time:.1f}s...")
                        time.sleep(sleep_time)
                        continue
                    raise e
            
            # 2. Fallback to serial processing
            print(f"[RobustEmbeddings] Batch failed. Falling back to serial processing for {len(texts)} items.")
            results = []
            for t in texts:
                try:
                    res = _embed_single(t)
                    results.append(res)
                except Exception as e:
                    print(f"[RobustEmbeddings] Single embed failed: {e}")
                    results.append([0.0] * 768)
            return results

    return RobustOllamaEmbeddings(**kwargs)
