"""
Main Streamlit application for Legal Drafting System
Provides a web interface for legal document Q&A using RAG (Retrieval-Augmented Generation)
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
from time_utils import now_ist_stamp
from dotenv import load_dotenv

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

# Import RAG components for document retrieval and processing
from rag import (
    get_vectorstore,
    build_retriever,
    maybe_full_case,
    filtration_retriever,
    assemble_context_from_plan,
    apply_named_case_guard,
    summary_search_pg,
    fuse_cases_by_rrf,
    fetch_top_chunks_for_cases,
    interleave_docs_by_case,
    enforce_case_diversity,
)
MODEL_CONTEXT_WINDOWS = {
    "gpt-5-nano-2025-08-07": 400000,
    "openai/gpt-oss-120b": 131072,
    "openai/gpt-oss-20b": 131072,
    "meta-llama/llama-4-scout": 327700,
    "qwen/qwen3-235b-a22b": 40960,
    "qwen/qwen3-14b": 40960,
    "qwen3:latest": 32768,
}

# Reserve some headroom for the model's completion tokens so we don't exceed total context
RESERVED_COMPLETION_TOKENS = {
    "gpt-5-nano-2025-08-07": 20000,
    "openai/gpt-oss-120b": 8192,
    "openai/gpt-oss-20b": 8192,
    "meta-llama/llama-4-scout": 20000,
    "qwen/qwen3-235b-a22b": 16000,
    "qwen/qwen3-14b": 16000,
    "qwen3:latest": 16000,
}
from st_debug import debug as sdebug
from orchestrator import process_query, QueryPlan


def _init_models():
    """
    Initialize and configure the language models and retrieval settings.
    Creates sidebar UI controls for user configuration.
    
    Returns:
        tuple: (llm, model_name, filtration_mode, k_val, court_name, statutes, query_sort_mode)
    """
    load_dotenv()  # Load environment variables from .env file
    
    # Create sidebar controls for filtration settings
    with st.sidebar.expander("Filtration LLM", expanded=True):
        # Choose how to filter retrieved documents
        filtration_mode = st.selectbox(
            "LLM filtration model mode",
            ["chunk context filtration", "metadata filtration"],
            index=0,
            key="filtration_mode_select",
        )
        
        # Configure retrieval parameters
        auto_topk = st.toggle("Auto Top-K (LLM)", value=True, key="auto_topk", help="Let the Query Processor set retrieval K based on query type")
        if not auto_topk:
            k_val = st.slider("Top-K chunks", min_value=3, max_value=20, value=6, step=1, key="k_chunks")
        else:
            # Keep a placeholder for UI state when auto mode is on
            k_val = st.session_state.get("k_chunks", 6)
        
        query_sort_mode = st.selectbox(
            "Query sorting mode",
            ["Manual (no general law)", "Auto (allow general law)"],
            index=0,
            key="query_sort_mode",
            help="Manual: force retrieval (new/followup). Auto: allow general-law routing for very broad queries.",
        )
        # Diversity controls
        # Auto Min Full Docs (LLM-driven)
        auto_min_docs = st.toggle(
            "Auto Min Full Docs (LLM)",
            value=True,
            key="auto_min_docs",
            help="Let the Query Processor suggest min full cases based on query breadth",
        )
        if not auto_min_docs:
            min_full_cases = st.slider(
                "Min full cases for overview",
                min_value=1,
                max_value=8,
                value=st.session_state.get("min_full_cases", 3),
                step=1,
                key="min_full_cases",
                help="Minimum distinct cases to include as full docs for broad queries",
            )
        else:
            # Preserve prior value for when user switches off auto later
            _ = st.session_state.get("min_full_cases", 3)
            min_full_cases = _
        max_chunks_per_case = st.slider(
            "Max chunks per case",
            min_value=1,
            max_value=6,
            value=2,
            step=1,
            key="max_chunks_per_case",
            help="Upper bound on chunk selections per case to avoid over-concentration",
        )

    # Create sidebar controls for chat model selection
    with st.sidebar.expander("Chat LLM", expanded=True):
        models = [
            "gpt-5-nano-2025-08-07",            # OpenAI (Responses API via LangChain)
            "qwen3:latest",                     # Ollama local
            # OpenRouter models (via OpenAI-compatible API)
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "meta-llama/llama-4-scout",
            "qwen/qwen3-235b-a22b",
            "qwen/qwen3-14b",
        ]
        model = st.selectbox("Model", models, index=0, key="chat_model_select")
    
    # Initialize the chat language model based on user selection
    openrouter_models = {
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "meta-llama/llama-4-scout",
        "qwen/qwen3-235b-a22b",
        "qwen/qwen3-14b",
    }
    provider_name = "openai"
    if model == "gpt-5-nano-2025-08-07":
        api_key = os.getenv("OPENAI_KEY")
        if not api_key:
            st.sidebar.warning("OPENAI_KEY not set; falling back to qwen3:latest")
            llm = ChatOllama(model="qwen3:latest", temperature=0, streaming=True, num_ctx=40000)
            provider_name = "ollama"
        else:
            llm = ChatOpenAI(model="gpt-5-nano-2025-08-07", temperature=0, streaming=True, api_key=api_key)
            provider_name = "openai"
    elif model in openrouter_models:
        or_key = os.getenv("OPENROUTER_API_KEY")
        if not or_key:
            st.sidebar.warning("OPENROUTER_API_KEY not set; falling back to qwen3:latest")
            llm = ChatOllama(model="qwen3:latest", temperature=0, streaming=True, num_ctx=40000)
            provider_name = "ollama"
        else:
            # Use OpenRouter via OpenAI-compatible LangChain client
            chat_kwargs = {
                "model": model,
                "temperature": 0,
                "streaming": True,
                "api_key": or_key,
                "base_url": "https://openrouter.ai/api/v1",
            }
            if model == "meta-llama/llama-4-scout":
                chat_kwargs["model_kwargs"] = {
                    "extra_body": {"provider": {"only": ["deepinfra/fp8"]}}
                }
            llm = ChatOpenAI(**chat_kwargs)
            provider_name = "openrouter"
    else:
        llm = ChatOllama(model="qwen3:latest", temperature=0, streaming=True, num_ctx=40000)
        provider_name = "ollama"
    
    # Parse statute filters from comma-separated text
    # Sidebar filters removed; planner handles filtering via rewrite
    statutes = None
    court_name = "Supreme Court of India"

    return llm, model, filtration_mode, k_val, court_name, statutes, query_sort_mode, min_full_cases, max_chunks_per_case, auto_min_docs, provider_name


def _ensure_session_state():
    """
    Initialize all required session state variables for the Streamlit app.
    These variables persist across user interactions within a session.
    """
    _ensure_user_profile()
    # Chat history and message storage
    if "history" not in st.session_state:
        st.session_state.history = InMemoryChatMessageHistory()
    if "debug_logs" not in st.session_state:
        st.session_state.debug_logs = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = InMemoryChatMessageHistory()
    
    # Context and document state from previous queries
    if "last_context" not in st.session_state:
        st.session_state.last_context = ""
    if "last_docs" not in st.session_state:
        st.session_state.last_docs = []
    if "last_stems" not in st.session_state:
        st.session_state.last_stems = []
    if "last_filters" not in st.session_state:
        st.session_state.last_filters = {}
    if "last_question_rewrite" not in st.session_state:
        st.session_state.last_question_rewrite = ""
    if "rolling_summary" not in st.session_state:
        st.session_state.rolling_summary = ""


def _reset_chat_state():
    """
    Clear all chat-related session state when user clicks "Clear chat & context".
    This resets the conversation history and debug logs.
    """
    st.session_state.messages = []
    # Reset both history stores used in the app
    st.session_state.history = InMemoryChatMessageHistory()
    st.session_state.chat_history = InMemoryChatMessageHistory()
    st.session_state.user_profile = {}
    # Clear any cached context and planner-related artifacts
    st.session_state.last_context = ""
    st.session_state.last_docs = []
    st.session_state.last_stems = []
    st.session_state.last_filters = {}
    st.session_state.last_question_rewrite = ""
    st.session_state.rolling_summary = ""
    # Clear debug
    st.session_state.debug_logs = []
    st.session_state["debug_string"] = ""
    # Intentionally do not log a reset section in the user-visible log


def _append_debug(msg: str):
    """
    Add a debug message to the session state and debug window.
    
    Args:
        msg (str): Debug message to add
    """
    ts_msg = f"{now_ist_stamp()} {msg}"
    st.session_state.debug_logs.append(ts_msg)
    try:
        sdebug(ts_msg)  # Also add to Streamlit debug window
    except Exception:
        pass  # Ignore debug window errors

def _debug_section(title: str, payload, note: str | None = None) -> None:
    try:
        if isinstance(payload, (dict, list)):
            body = __import__("json").dumps(payload, indent=2)
        else:
            body = str(payload)
    except Exception:
        body = str(payload)
    sep = "=" * 28
    if note:
        header = f"\n\n{sep}\n{title}\n{sep}\nNote: {note}\n"
    else:
        header = f"\n\n{sep}\n{title}\n{sep}\n"
    _append_debug(f"{header}{body}\n{'-'*28}")


_NAME_CAPTURE = re.compile(
    r"\b(?:my name is|i am|i['`]?m|this is)\s+([A-Za-z][A-Za-z\s'.-]{0,40})",
    re.IGNORECASE,
)
_NAME_QUERY = re.compile(r"\bwhat(?:'s| is)\s+my\s+name\b", re.IGNORECASE)


def _normalise_name(raw: str) -> str:
    cleaned = (raw or "").strip(" .,!?:;\"'")
    if not cleaned:
        return ""
    parts = cleaned.split()
    return " ".join(part.capitalize() for part in parts)


def _ensure_user_profile():
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = {}


def _update_user_profile(text: str) -> None:
    if not text:
        return
    match = _NAME_CAPTURE.search(text)
    if not match:
        return
    candidate = _normalise_name(match.group(1))
    if not candidate:
        return
    _ensure_user_profile()
    st.session_state.user_profile["preferred_name"] = candidate


def _get_known_user_name() -> Optional[str]:
    _ensure_user_profile()
    name = st.session_state.user_profile.get("preferred_name")
    if name:
        return name
    for message in reversed(st.session_state.messages):
        if message.get("role") != "user":
            continue
        text = message.get("content") or ""
        match = _NAME_CAPTURE.search(text)
        if match:
            candidate = _normalise_name(match.group(1))
            if candidate:
                st.session_state.user_profile["preferred_name"] = candidate
                return candidate
    return None


def _persona_prompt_note() -> str:
    name = _get_known_user_name()
    if not name:
        return ""
    return f"The user introduced themselves as {name}; address them by name when greeting or acknowledging them."


def _maybe_answer_personal_memory_question(user_q: str) -> Optional[str]:
    if not user_q:
        return None
    lower = user_q.strip()
    if not lower:
        return None
    if _NAME_QUERY.search(lower):
        name = _get_known_user_name()
        if name:
            return f"You mentioned earlier that your name is {name}."
        return "I don't think you've told me your name yet."
    return None


def _handle_general_chat(user_input: str, llm) -> str:
    text = (user_input or "").strip()
    lower = text.lower()
    known_name = _get_known_user_name()

    if not text:
        return "Hi there! Let me know when you have a legal question."

    if _NAME_QUERY.search(lower):
        if known_name:
            return f"You mentioned earlier that your name is {known_name}. I'm ready whenever you want to discuss a legal question."
        return "I don't think you've shared your name yet. Tell me your legal question and I'll do my best to help."

    chat_system = (
        "You are a friendly assistant for the Legal Drafting System. "
        "When the user is not asking for legal help, reply briefly (no more than three sentences), "
        "stay positive, and encourage them to share any legal question if appropriate. "
        "You may answer general knowledge queries directly without mentioning legal content. "
        "If the user steers back toward legal matters, gently invite them to provide more details so we can help."
    )

    context_notes = []
    if known_name:
        context_notes.append(f"The user previously introduced themselves as {known_name}.")
    notes_block = ""
    if context_notes:
        notes_block = "Context notes:\n" + "\n".join(f"- {note}" for note in context_notes) + "\n\n"

    prompt = f"{chat_system}\n\n{notes_block}User message: {user_input}\nAssistant:"
    try:
        reply = (_collect_stream_text(llm, prompt) or "").strip()
    except Exception as exc:
        _append_debug(f"[DEBUG][GeneralChat][error] {exc}")
        reply = "I'm glad we're chatting. If you have a legal question, feel free to share it with me."
    return reply or "I'm glad we're chatting. If you have a legal question, feel free to share it with me."


def _fmt_qp(plan: dict) -> str:
    try:
        lines = []
        if plan is None:
            return "(no plan)"
        t = plan.get("type")
        if t:
            lines.append(f"type: {t}")
        rw = plan.get("rewrite")
        if rw:
            lines.append(f"rewritten prompt: {rw}")
        cp = plan.get("case_probe")
        if cp:
            lines.append(f"case_probe: {cp}")
        kc = plan.get("keep_context")
        if kc is not None:
            lines.append(f"keep_context: {bool(kc)}")
        bs = plan.get("bridging_strategy")
        if bs:
            lines.append(f"bridge: {bs}")
        sts = plan.get("statutes") or []
        if sts:
            lines.append("statutes: " + ", ".join(sts))
        rk = plan.get("retrieval_k")
        if rk is not None:
            lines.append(f"retrieval_k: {rk}")
        rs = plan.get("reason")
        if rs:
            lines.append(f"reason: {rs}")
        return "\n".join(lines)
    except Exception:
        return str(plan)


def _fmt_topk(info: dict) -> str:
    if not isinstance(info, dict):
        return str(info)
    ek = info.get("effective_k")
    src = info.get("source")
    return f"effective_k: {ek}\nsource: {src}"


def _fmt_summary_fts(rows: list) -> str:
    try:
        if not rows:
            return "(none)"
        out = []
        for i, r in enumerate(rows, 1):
            # rows may be tuples (fs, rk) or dicts
            if isinstance(r, dict):
                fs = r.get("file_stem")
                rk = r.get("rank")
            else:
                fs, rk = r
            out.append(f"{i}. {fs} (rank={rk:.4f})")
        return "\n".join(out)
    except Exception:
        return str(rows)


def _fmt_fused_cases(cases: list[str]) -> str:
    if not cases:
        return "(none)"
    return "\n".join(f"{i}. {fs}" for i, fs in enumerate(cases, 1))


def _fmt_retrieved_previews(items: list[dict]) -> str:
    try:
        if not items:
            return "(none)"
        return "\n".join(
            f"{i}. {it.get('file_stem')}: {it.get('preview')}"
            for i, it in enumerate(items, 1)
        )
    except Exception:
        return str(items)


def _fmt_filtration_plan(plan: dict) -> str:
    try:
        if not plan:
            return "(none)"
        full = plan.get("selected_full_docs") or []
        chunks = plan.get("selected_chunks") or []
        lines = [
            "full docs: " + (", ".join(full) if full else "(none)"),
            f"selected chunks: {len(chunks)}",
        ]
        if chunks:
            sample = chunks[:4]
            sample_txt = ", ".join(f"{c.get('file_stem')}[{c.get('chunk_index')}]" for c in sample)
            lines.append(f"sample chunks: {sample_txt}")
        # budget from plan is ignored intentionally
        return "\n".join(lines)
    except Exception:
        return str(plan)


def _collect_stream_text(llm, prompt: str) -> str:
    """Run a streaming LLM call and capture the combined text."""
    text_chunks: List[str] = []
    for chunk in llm.stream(prompt):
        content = getattr(chunk, "content", None)
        if content:
            text_chunks.append(content)
    return "".join(text_chunks)


def _case_sources_from_plan(plan) -> List[str]:
    """Extract unique file stems referenced by a filtration plan."""
    if not plan:
        return []
    sources: List[str] = []
    full_docs = getattr(plan, "selected_full_docs", None) or []
    for stem in full_docs:
        if stem and stem not in sources:
            sources.append(stem)
    chunks = getattr(plan, "selected_chunks", None) or []
    for ch in chunks:
        if hasattr(ch, "file_stem"):
            stem = getattr(ch, "file_stem")
        elif isinstance(ch, dict):
            stem = ch.get("file_stem")
        else:
            stem = None
        if stem and stem not in sources:
            sources.append(stem)
    return sources


def _execute_case_probe(
    probe_query: str,
    user_q: str,
    vs,
    court_name: str,
    statutes: Optional[List[str]],
    model_name: str,
    filtration_mode: str,
    max_chunks_per_case: int,
) -> dict:
    """Run a lightweight retrieval pipeline to surface precedents for general-law answers."""
    result = {
        "context": "",
        "docs": [],
        "stems": [],
        "plan": None,
        "detail": "No precedents retrieved",
        "dbg": {},
        "fallback_used": False,
    }

    if not probe_query:
        return result

    probe_k = 8
    probe_min_docs = 2

    def _run_probe(statute_filters=None):
        docs_local = []
        sum_ranked_local = []
        try:
            retriever_local = build_retriever(
                vs,
                statute_filters=statute_filters,
                court_name=court_name,
                k=probe_k,
            )
            docs_local = retriever_local.invoke(probe_query) or []
            _append_debug(
                f"[DEBUG][CaseProbe] vector hits={len(docs_local)} (statute_filters={'None' if statute_filters is None else statute_filters})"
            )
        except Exception as exc_inner:
            _append_debug(f"[DEBUG][CaseProbe][retriever_error] {exc_inner}")
            docs_local = []
        try:
            year_filter_local = None
            try:
                match_local = re.search(r"\b(19|20)\d{2}\b", probe_query)
                if match_local:
                    year_filter_local = int(match_local.group(0))
            except Exception:
                year_filter_local = None
            sum_ranked_local = summary_search_pg(
                probe_query,
                court_name=court_name,
                statutes=statute_filters,
                year=year_filter_local,
                limit=max(40, probe_k * 6),
            )
            _append_debug(
                f"[DEBUG][CaseProbe] summary hits={len(sum_ranked_local) if sum_ranked_local else 0} (statute_filters={'None' if statute_filters is None else statute_filters})"
            )
        except Exception as exc_inner2:
            _append_debug(f"[DEBUG][CaseProbe][summary_error] {exc_inner2}")
            sum_ranked_local = []
        return docs_local, sum_ranked_local

    primary_filters = statutes or None
    docs, sum_ranked = _run_probe(primary_filters)
    fallback_used = False
    if not docs and not sum_ranked and primary_filters is not None:
        _append_debug("[DEBUG][CaseProbe] No hits with statute filters; retrying without filters.")
        docs, sum_ranked = _run_probe(None)
        fallback_used = True

    try:
        fused_cases = fuse_cases_by_rrf(sum_ranked, docs, k=40, top_n=max(12, probe_k * 2)) if (sum_ranked or docs) else []
    except Exception as exc:
        _append_debug(f"[DEBUG][CaseProbe][fuse_error] {exc}")
        fused_cases = []
    if fused_cases:
        _debug_section("Case Probe - Fused Cases", _fmt_fused_cases(fused_cases[:10]), note="Ranked precedents chosen for advisory add-on.")
    else:
        if not docs:
            detail_note = "No precedents retrieved"
            if fallback_used:
                detail_note += " (after filter fallback)"
            result["detail"] = detail_note
            result["fallback_used"] = fallback_used
            return result

    if fused_cases:
        try:
            docs = fetch_top_chunks_for_cases(vs, probe_query, fused_cases, per_case_k=2)
            docs = interleave_docs_by_case(docs, per_case_limit=2, max_total=probe_k)
        except Exception as exc:
            _append_debug(f"[DEBUG][CaseProbe][fetch_error] {exc}")

    if not docs:
        detail_note = "No precedents retrieved"
        if fallback_used:
            detail_note += " (after filter fallback)"
        result["detail"] = detail_note
        result["fallback_used"] = fallback_used
        return result

    stems = [fs for fs in fused_cases if fs] if fused_cases else sorted({(d.metadata or {}).get("file_stem") for d in docs if (d.metadata or {}).get("file_stem")})
    case_metas = _load_case_metadata_for_stems(stems, "narrow")
    _append_debug(f"[DEBUG][CaseProbe] loaded metadata for {len(case_metas)} cases")

    query_context_payload = {
        "breadth": "narrow",
        "fused_case_count": len(stems),
        "retrieval_k": probe_k,
        "desired_min_full_docs": probe_min_docs,
        "planner_min_full_docs": probe_min_docs,
    }
    if fused_cases:
        query_context_payload["fused_ranked_cases"] = fused_cases[:12]

    mode_key = "chunk" if filtration_mode == "chunk context filtration" else "metadata"
    try:
        plan = filtration_retriever(
            probe_query,
            docs,
            mode=mode_key,
            case_metadatas=case_metas,
            desired_min_full_docs=probe_min_docs,
            desired_max_chunks_per_case=min(max_chunks_per_case, 2),
            query_context=query_context_payload,
        )
    except Exception as exc:
        _append_debug(f"[DEBUG][CaseProbe][filtration_error] {exc}")
        result["fallback_used"] = fallback_used
        return result

    try:
        plan = apply_named_case_guard(plan, probe_query, docs)
        plan = enforce_case_diversity(
            plan,
            fused_cases=fused_cases,
            desired_min_full_docs=probe_min_docs,
            desired_max_chunks_per_case=min(max_chunks_per_case, 2),
            breadth="narrow",
            allow_expansion=False,
        )
        _append_debug(f"[DEBUG][CaseProbe][Guard] full_docs={len(plan.selected_full_docs)} chunks={len(plan.selected_chunks)}")
    except Exception as exc:
        _append_debug(f"[DEBUG][CaseProbe][guard_error] {exc}")

    ctx_limit = MODEL_CONTEXT_WINDOWS.get(model_name, 32768)
    headroom = RESERVED_COMPLETION_TOKENS.get(model_name, 8000)
    budget_override = max(4000, min(20000, ctx_limit - headroom))

    try:
        context, _, dbg = assemble_context_from_plan(
            plan,
            probe_query,
            vs,
            txt_dir="processed_data/txt_data",
            budget_tokens=budget_override,
            initial_docs=docs,
        )
    except Exception as exc:
        _append_debug(f"[DEBUG][CaseProbe][assembly_error] {exc}")
        context, dbg = "", {}

    if context:
        result.update(
            {
                "context": context,
                "docs": docs,
                "stems": stems,
                "plan": plan,
                "detail": f"{len(plan.selected_full_docs)} case(s); {len(plan.selected_chunks)} chunk(s){' (fallback on filters)' if fallback_used else ''}",
                "dbg": dbg,
            }
        )

        try:
            _debug_section(
                "Case Probe - Filtration Plan",
                _fmt_filtration_plan(plan.model_dump()),
                note="LLM-selected precedent snippets for advisory reinforcement.",
            )
        except Exception:
            pass
        if isinstance(dbg, dict):
            est_tokens = int(dbg.get("est_tokens", 0) or 0)
        else:
            est_tokens = int(getattr(dbg, "est_tokens", 0) or 0)
        _debug_section(
            "Case Probe - Assembler",
            {"est_tokens": est_tokens, "detail": result["detail"]},
            note="Token footprint for precedent context passed to the case insight LLM.",
        )
    else:
        detail_note = "No precedents retrieved"
        if fallback_used:
            detail_note += " (after filter fallback)"
        result["detail"] = detail_note

    result["fallback_used"] = fallback_used
    return result


def _fmt_assembler(dbg: dict) -> str:
    try:
        if not dbg:
            return "(none)"
        est = dbg.get("est_tokens")
        spans = dbg.get("spans") or []
        lines = [
            f"estimated tokens: {est}",
            f"spans: {len(spans)}",
        ]
        if spans:
            sample = spans[:3]
            sample_txt = ", ".join(f"{s[2]}[{s[0]}–{s[1]}]" if len(s) >= 3 else str(s) for s in sample)
            lines.append(f"sample spans: {sample_txt}")
        return "\n".join(lines)
    except Exception:
        return str(dbg)


def _trim_text(value: Optional[str], limit: int) -> str:
    """
    Trim long metadata fields so filtration prompts stay concise.
    """
    if not value:
        return ""
    txt = str(value).strip()
    if len(txt) <= limit:
        return txt
    return txt[:limit].rstrip() + "…"


def _load_case_metadata_for_stems(stems: List[str], qp_breadth: str) -> List[dict]:
    """
    Load case-level metadata JSONs and flatten key fields for the filtration LLM.
    """
    metas: List[dict] = []
    for rank_idx, fs in enumerate(stems, 1):
        if not fs:
            continue
        mp = Path("processed_data/metadata") / f"{fs}.json"
        if not mp.exists():
            continue
        try:
            data = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            continue
        statutes_raw = data.get("legal_provisions_cited") or data.get("statutes") or []
        if isinstance(statutes_raw, str):
            statutes_list = [statutes_raw]
        elif isinstance(statutes_raw, list):
            statutes_list = [str(s) for s in statutes_raw][:12]
        else:
            statutes_list = []
        metas.append({
            "file_stem": fs,
            "rank": rank_idx,
            "case_number": data.get("case_number"),
            "date_of_judgment": data.get("date_of_judgment"),
            "court_name": data.get("court_name"),
            "summary": _trim_text(data.get("summary"), 700),
            "final_judgment": _trim_text(data.get("final_judgment"), 400),
            "statutes": statutes_list,
            "parties": data.get("parties"),
            "year": data.get("year"),
            "title": data.get("title"),
            "breadth_hint": qp_breadth,
            "metadata": data,
        })
    return metas


_PIPELINE_STAGES: List[Tuple[str, str]] = [
    ("planner", "Rewriting query"),
    ("retrieval", "Retrieving information"),
    ("filtration", "Filtering relevant documents"),
    ("assembly", "Assembling context"),
    ("case_probe", "Reviewing case precedents"),
    ("answer", "Generating answer"),
]

import streamlit as st

class PipelineTracker:
    """
    Lightweight helper to surface pipeline progress in the Streamlit UI.
    """

    def __init__(self, stages: Optional[List[Tuple[str, str]]] = None):
        self.stages = stages or _PIPELINE_STAGES
        self.stage_labels: Dict[str, str] = {key: label for key, label in self.stages}
        self.stage_index: Dict[str, int] = {key: idx for idx, (key, _label) in enumerate(self.stages)}
        self.failed = False
        self.current_stage: Optional[str] = None
        self.total_stages = max(1, len(self.stages))

        # Visual elements
        self.status_box = st.status("Starting pipeline…", state="running", expanded=True)
        with self.status_box:
            self.progress_bar = st.progress(0)

    def _progress_for_stage(self, key: str) -> None:
        idx = self.stage_index.get(key)
        if idx is None:
            return
        percent = int(round(((idx + 1) / self.total_stages) * 100))
        if self.progress_bar is not None:
            self.progress_bar.progress(percent)

    def stage_running(self, key: str, message: Optional[str] = None) -> None:
        if key not in self.stage_labels:
            return
        self.current_stage = key
        label = message or self.stage_labels[key]
        self.status_box.update(label=label, state="running", expanded=True)

    def stage_complete(self, key: str, detail: Optional[str] = None) -> None:
        if key not in self.stage_labels:
            return
        completion_label = detail or f"{self.stage_labels[key]} complete"
        self._progress_for_stage(key)
        self.status_box.update(label=completion_label, state="running", expanded=True)

    def stage_skip(self, key: str, detail: Optional[str] = None) -> None:
        if key not in self.stage_labels:
            return
        skip_label = detail or f"Skipped {self.stage_labels[key]}"
        self._progress_for_stage(key)
        self.status_box.update(label=skip_label, state="running", expanded=True)

    def stage_error(self, key: str, detail: str) -> None:
        if key not in self.stage_labels:
            return
        err_label = f"Error during {self.stage_labels[key]}"
        self._progress_for_stage(key)
        self.failed = True
        self.status_box.update(label=f"{err_label}: {detail}", state="error", expanded=True)

    def finish(self, detail: Optional[str] = None) -> None:
        final_label = detail or ("Pipeline finished with issues" if self.failed else "Pipeline complete")
        final_state = "error" if self.failed else "complete"
        if self.progress_bar is not None:
            self.progress_bar.empty()
            self.progress_bar = None
        self.status_box.update(label=final_label, state=final_state, expanded=False)
        if not self.failed:
            st.toast("Answer Generated", icon="✅", duration="short")
            self.status_box.empty()
            self.status_box = None


def _select_followup_case(
    plan: QueryPlan,
    user_q: str,
    stems: List[str],
    docs: List,
) -> Tuple[Optional[str], int]:
    """Pick which cached case to surface fully for follow-up queries."""
    if not stems:
        return None, 0

    # Planner-provided hints (target_stems) take priority
    target_stems = getattr(plan, "target_stems", None) or []
    for cand in target_stems:
        if cand in stems:
            return cand, 100  # treat planner hint as highest confidence

    # Build a metadata map from cached docs for light matching
    meta_by_stem: Dict[str, dict] = {}
    for d in docs or []:
        meta = getattr(d, "metadata", {}) or {}
        fs = meta.get("file_stem")
        if fs and fs not in meta_by_stem:
            meta_by_stem[fs] = meta

    haystack_tokens_source = " ".join(
        part
        for part in [getattr(plan, "rewrite", "") or "", user_q or ""]
        if part
    ).lower()
    tokens = [tok for tok in re.split(r"\W+", haystack_tokens_source) if tok and len(tok) > 2]
    if not tokens:
        return stems[0], 0

    best_stem = None
    best_score = -1
    for stem in stems:
        meta = meta_by_stem.get(stem)
        if not meta:
            meta_path = Path("processed_data/metadata") / f"{stem}.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
            else:
                meta = {}
        parts = [
            meta.get("case_number") or "",
            meta.get("title") or "",
            meta.get("summary") or "",
            meta.get("final_judgment") or "",
        ]
        parties = meta.get("parties") or {}
        if isinstance(parties, dict):
            parts.extend(parties.values())
        haystack = " ".join(part for part in parts if part).lower()
        if not haystack:
            score = 0
        else:
            score = sum(1 for tok in tokens if tok in haystack)
        if score > best_score:
            best_score = score
            best_stem = stem

    final_stem = best_stem or stems[0]
    return final_stem, best_score


def main():
    """
    Main Streamlit application function.
    Sets up the UI, handles user queries, and orchestrates the RAG pipeline.
    """
    # Configure Streamlit page
    st.set_page_config(page_title="Legal RAG Chat", page_icon="⚖️", layout="wide")
    st.title("⚖️ Legal Drafting Chatbot")
    
    # Initialize session state and models
    _ensure_session_state()
    llm, model_name, filtration_mode, k_val, court_name, statutes, query_sort_mode, min_full_cases, max_chunks_per_case, auto_min_docs, provider_name = _init_models()

    # Connect to vector database and build retriever
    vs = get_vectorstore()


    # Create tabbed interface: Chat and Debug
    chat_tab, debug_tab = st.tabs(["Chat", "Debug"])

    with chat_tab:
        # Clear chat button
        st.button("Clear chat & context", on_click=_reset_chat_state, key="clear_chat", use_container_width=True)

        # Display previous messages
        messages_container = st.container()
        with messages_container:
            for m in st.session_state.messages:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])

        # Chat input at bottom of screen
        user_q = st.chat_input("Ask a legal question...", key="chat_input_main")
        if user_q:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_q})
            with messages_container:
                with st.chat_message("user"):
                    st.markdown(user_q)

            _update_user_profile(user_q)
            personal_memory_reply = _maybe_answer_personal_memory_question(user_q)
            if personal_memory_reply is not None:
                try:
                    st.session_state.chat_history.add_user_message(user_q)
                except Exception:
                    pass
                with messages_container:
                    with st.chat_message("assistant", avatar="⚖️"):
                        st.markdown(personal_memory_reply)
                st.session_state.messages.append({"role": "assistant", "content": personal_memory_reply})
                try:
                    st.session_state.chat_history.add_ai_message(personal_memory_reply)
                except Exception:
                    pass
                _append_debug("[DEBUG][Persona] answered from stored user profile")
                st.stop()

            tracker = PipelineTracker()
            try:
                tracker.stage_running("planner", "Rewriting query…")

                # Step 1: Query Processing - Classify and plan the query
                qp: QueryPlan = process_query(
                    user_q,
                    last_question_rewrite=st.session_state.last_question_rewrite or None,
                    last_stems=st.session_state.last_stems,
                    last_filters=st.session_state.last_filters,
                    last_context_snippet=(st.session_state.last_context or "")[:4000],
                    summary=st.session_state.rolling_summary or None,
                    manual_mode=query_sort_mode.startswith("Manual"),
                )
                if query_sort_mode.startswith("Manual"):
                    _append_debug("[DEBUG][QP] Using manual-only prompt (no general_law)")
                _debug_section("Query Plan", _fmt_qp(qp.model_dump()), note="Planner's classification and rewrite; drives how we retrieve and filter.")

                # If query sorting mode is Manual (no general law), override to retrieval
                if query_sort_mode.startswith("Manual") and getattr(qp, "type", "") == "general_law":
                    qp.type = "new"
                    # Force a crisp retrieval rewrite (short, keywords/statutes/issues)
                    try:
                        base = (qp.rewrite or user_q)
                        # Simple heuristic rewrite: strip long prose, keep key terms
                        import re as _re
                        tokens = [t for t in _re.split(r"\W+", base) if t]
                        # keep up to 20 tokens
                        qp.rewrite = " ".join(tokens[:20])
                    except Exception:
                        pass
                    if not getattr(qp, "retrieval_k", None):
                        qp.retrieval_k = 8
                    _append_debug("[DEBUG][QP] Manual mode: forced 'new' and retrieval-oriented rewrite")

                qp_breadth = getattr(qp, "breadth", "unknown")
                tracker.stage_complete("planner", f"type={getattr(qp, 'type', 'unknown')} breadth={qp_breadth}")

                general_chat_mode = getattr(qp, "type", "") == "general_chat"
                general_law_mode = getattr(qp, "type", "") == "general_law"
                case_probe_query = (getattr(qp, "case_probe", "") or "").strip()
                general_law_retrieval = general_law_mode and bool(case_probe_query)

                # Decide effective Top-K (retrieval depth)
                if general_chat_mode:
                    effective_k = 0
                    k_source = "general_chat"
                elif general_law_mode and not general_law_retrieval:
                    effective_k = 0
                    k_source = "general_law_no_case_probe"
                else:
                    auto_topk_enabled = bool(st.session_state.get("auto_topk", True))
                    if auto_topk_enabled and getattr(qp, "retrieval_k", None):
                        try:
                            effective_k = int(qp.retrieval_k)
                        except Exception:
                            effective_k = 6
                        # Clamp to a reasonable range
                        effective_k = max(3, min(20, effective_k))
                        k_source = "qp.retrieval_k"
                    elif auto_topk_enabled:
                        # Fallback heuristic if planner did not set K
                        effective_k = 6
                        k_source = "auto_heuristic"
                    else:
                        # Manual slider
                        try:
                            effective_k = int(k_val)
                        except Exception:
                            effective_k = 6
                        effective_k = max(3, min(20, effective_k))
                        k_source = "manual_slider"
                _debug_section("Top-K Selection", _fmt_topk({"effective_k": effective_k, "source": k_source}), note="How many items to retrieve; affects breadth vs. depth.")
                # Decide Min Full Docs (either from planner or UI)
                if general_chat_mode:
                    effective_min_docs = 0
                    min_docs_source = "general_chat"
                elif general_law_mode and not general_law_retrieval:
                    effective_min_docs = 0
                    min_docs_source = "general_law_no_case_probe"
                elif auto_min_docs and getattr(qp, "min_full_docs", None):
                    effective_min_docs = max(2, int(qp.min_full_docs))
                    min_docs_source = "qp.min_full_docs"
                elif auto_min_docs:
                    # Heuristic fallback informed by planner breadth
                    if qp_breadth == "broad":
                        effective_min_docs = 5
                    elif qp_breadth == "narrow":
                        effective_min_docs = 3
                    else:
                        effective_min_docs = 2
                    min_docs_source = "auto_heuristic"
                else:
                    effective_min_docs = max(2, int(min_full_cases))
                    min_docs_source = "manual_slider"
                _debug_section("Min Full Docs", {"min_full_docs": effective_min_docs, "source": min_docs_source}, note="Minimum number of full cases to include; tuned to query breadth.")

                # Initialize variables for context assembly
                context = ""
                docs = []
                stems = []
                dbg = {"est_tokens": 0, "spans": []}
                strategy = ""
                case_probe_context = ""
                case_probe_plan = None
                case_probe_sources: List[str] = []
                case_probe_stage_handled = False
                case_probe_detail = ""

                # Step 2: Context Strategy - Choose how to handle the query
                tracker.stage_running("retrieval", "Retrieving information…")
                retrieval_notes: List[str] = []
                if general_chat_mode:
                    strategy = "general_chat"
                    tracker.stage_skip("retrieval", "General chat (no retrieval).")
                    tracker.stage_skip("filtration", "General chat (skipped).")
                    tracker.stage_skip("assembly", "General chat (skipped).")
                    tracker.stage_skip("case_probe", "General chat (no precedents).")
                    case_probe_stage_handled = True
                elif general_law_mode and not general_law_retrieval:
                    # General legal knowledge without precedent retrieval fallback
                    strategy = "general_law"
                    tracker.stage_skip("retrieval", "General-law routing (no retrieval).")
                    tracker.stage_skip("filtration", "General-law routing (skipped).")
                    tracker.stage_skip("assembly", "General-law routing (skipped).")
                    tracker.stage_skip("case_probe", "No case-probe rewrite provided; advisory only.")
                    case_probe_stage_handled = True
                elif qp.type == "followup" and qp.keep_context and st.session_state.last_context:
                    # Reuse context from previous query
                    strategy = "reuse_last_context"
                    context = st.session_state.last_context
                    docs = st.session_state.last_docs or []
                    stems = st.session_state.last_stems or []
                    dbg = {"est_tokens": len(context)//4, "spans": []}
                    reused_details = {
                        "chars": len(context),
                        "est_tokens": len(context) // 4,
                        "cases_cached": stems,
                        "filters_cached": st.session_state.last_filters,
                    }
                    _debug_section(
                        "Context Reuse (followup)",
                        reused_details,
                        note="Leveraged previous answer context; no new retrieval yet.",
                    )
                    # Step 2a: Bridging Strategies - Enhance context based on query plan
                
                    # Strategy: Add more documents by statute filters
                    if getattr(qp, "bridging_strategy", "") == "statute_refill" and qp.statutes:
                        try:
                            retr_q = qp.rewrite or user_q
                            # Build retriever with statute filters
                            retriever_refill = build_retriever(
                                vs,
                                statute_filters=qp.statutes,
                                court_name=court_name,
                                k=effective_k,
                            )
                            refill_docs = retriever_refill.invoke(retr_q)
                            if refill_docs:
                                retrieval_notes.append(f"Statute refill {len(refill_docs)} chunk(s)")
                                # Get case metadata for filtration
                                stems_refill = sorted({(d.metadata or {}).get("file_stem") for d in refill_docs if (d.metadata or {}).get("file_stem")})
                                case_metas_refill = _load_case_metadata_for_stems(stems_refill, qp_breadth)
                                query_ctx_refill = {
                                    "breadth": qp_breadth,
                                    "fused_case_count": len(stems_refill),
                                    "retrieval_k": effective_k,
                                    "desired_min_full_docs": effective_min_docs,
                                    "planner_min_full_docs": int(getattr(qp, "min_full_docs", effective_min_docs) or effective_min_docs),
                                }
                                # Apply filtration to new documents
                                mode_key2 = "chunk" if filtration_mode == "chunk context filtration" else "metadata"
                                plan_refill = filtration_retriever(
                                    retr_q,
                                    refill_docs,
                                    mode=mode_key2,
                                    case_metadatas=case_metas_refill,
                                    desired_min_full_docs=effective_min_docs,
                                    desired_max_chunks_per_case=max_chunks_per_case,
                                    query_context=query_ctx_refill,
                                )
                                plan_refill = enforce_case_diversity(
                                    plan_refill,
                                    fused_cases=stems_refill,
                                    desired_min_full_docs=effective_min_docs,
                                    desired_max_chunks_per_case=max_chunks_per_case,
                                    breadth=qp_breadth,
                                    allow_expansion=bool(getattr(plan_refill, "request_more_cases", False) and qp_breadth == "broad"),
                                )
                                budget_override = 400000 if model_name == "gpt-5-nano-2025-08-07" else plan_refill.context_budget_tokens
                                context_refill, _, dbg_refill = assemble_context_from_plan(
                                    plan_refill,
                                    retr_q,
                                    vs,
                                    txt_dir="processed_data/txt_data",
                                    budget_tokens=budget_override,
                                    initial_docs=refill_docs,
                                )
                                # Merge the new context with existing context
                                if context_refill:
                                    context = (context + "\n\n---\n\n" + context_refill) if context else context_refill
                                    _append_debug("[DEBUG][Bridge] statute_refill merged additional context")
                                    strategy = "reuse_last_context+statute_refill"
                                    merged_info = {
                                        "chars": len(context),
                                        "est_tokens": len(context) // 4,
                                        "merge_components": ["previous_context", "statute_refill"],
                                    }
                                    _debug_section(
                                        "Context After Statute Refill",
                                        merged_info,
                                        note="Cached context plus refill additions passed to assembler/answer.",
                                    )
                        except Exception as be:
                            _append_debug(f"[DEBUG][Bridge] statute_refill error: {be}")
                
                    # Strategy: Load full text of the dominant case from previous query
                    if getattr(qp, "bridging_strategy", "") == "same_case_full" and stems:
                        try:
                            dominant, score = _select_followup_case(qp, user_q, stems, docs)
                            if not dominant:
                                raise ValueError("Unable to resolve dominant case for same_case_full")
                            full_path = Path("processed_data/txt_data") / f"{dominant}.txt"
                            if full_path.exists():
                                raw = full_path.read_text(encoding="utf-8")
                                header = f"[file: {dominant}.txt]\n"
                                context = header + raw
                                strategy = "same_case_full"
                                retrieval_notes.append(f"same_case_full {dominant}")
                                _append_debug(f"[DEBUG][Bridge] same_case_full loaded {dominant}.txt (score={score})")
                                _debug_section(
                                    "Context After Same Case Full",
                                    {
                                        "file_loaded": dominant,
                                        "chars": len(context),
                                        "est_tokens": len(context) // 4,
                                    },
                                    note="Entire dominant case inserted as context for follow-up answer.",
                                )
                            else:
                                _append_debug(f"[DEBUG][Bridge] same_case_full missing file: {full_path}")
                        except Exception as se:
                            _append_debug(f"[DEBUG][Bridge] same_case_full error: {se}")
                    reuse_detail = "; ".join(retrieval_notes) if retrieval_notes else "Reused previous context"
                    tracker.stage_complete("retrieval", reuse_detail)
                    tracker.stage_skip("filtration", "Context reused from previous turn.")
                    tracker.stage_skip("assembly", "Context reused from previous turn.")
                else:
                    # Step 2b: Fresh Retrieval - applies to new queries and general-law precedent lookup
                    retr_q = case_probe_query if general_law_retrieval else (qp.rewrite or user_q)
                    if general_law_retrieval:
                        _append_debug(f"[DEBUG][GeneralLaw] case-probe retrieval query: {retr_q}")
                        tracker.stage_running("case_probe", "Retrieving precedents…")
                    else:
                        _append_debug(f"[DEBUG][Retrieval] using rewrite query: {retr_q}")

                    # Build retriever with query plan filters (vector path)
                    retriever2 = build_retriever(
                        vs,
                        statute_filters=qp.statutes or None,
                        court_name=court_name,
                        k=effective_k,
                    )
                    docs_vector = retriever2.invoke(retr_q)
                
                    # Summary path (Postgres FTS over case summaries)
                    try:
                        # Optional: derive year filter from rewrite simple heuristic
                        year_filter = None
                        try:
                            import re as _re
                            m = _re.search(r"\b(19|20)\d{2}\b", retr_q)
                            if m:
                                year_filter = int(m.group(0))
                        except Exception:
                            year_filter = None
                        sum_ranked = summary_search_pg(
                            retr_q,
                            court_name=court_name,
                            statutes=qp.statutes or None,
                            year=year_filter,
                            limit=max(50, effective_k * 8),
                        )
                    except Exception as se:
                        _append_debug(f"[DEBUG][SummaryFTS] error: {se}")
                        sum_ranked = []
                    # Debug: show top summary cases
                    if sum_ranked:
                        _debug_section("Summary FTS - Top Cases", _fmt_summary_fts([{"file_stem": fs, "rank": rk} for fs, rk in sum_ranked[:10]]), note="Keyword-based matches over case summaries; seeds diverse case selection.")
                
                    # If vector returns nothing, try relaxed court filter for vectors
                    docs = docs_vector
                
                    # Fallback: if no results, relax court filter
                    if not docs:
                        retriever_relaxed = build_retriever(vs, court_name=None, k=effective_k)
                        docs = retriever_relaxed.invoke(retr_q)
                        if not sum_ranked and court_name is not None:
                            try:
                                sum_ranked = summary_search_pg(
                                    retr_q,
                                    court_name=None,
                                    statutes=qp.statutes or None,
                                    year=None,
                                    limit=max(50, effective_k * 8),
                                )
                            except Exception:
                                sum_ranked = []
                
                    # Fuse case-level results from summary path and vector path
                    try:
                        fused_cases = fuse_cases_by_rrf(sum_ranked, docs, k=60, top_n=max(20, effective_k * 3)) if (sum_ranked or docs) else []
                    except Exception as fe:
                        _append_debug(f"[DEBUG][Fusion] error: {fe}")
                        fused_cases = []
                    if fused_cases:
                        _debug_section("Fused Cases (RRF)", _fmt_fused_cases(fused_cases), note="Combined ranking from summaries and chunks; final case list before chunk fetch.")
                
                    # If fusion produced cases, fetch top chunks per case to replace docs
                    if fused_cases:
                        try:
                            docs = fetch_top_chunks_for_cases(vs, retr_q, fused_cases, per_case_k=max(2, min(6, effective_k)))
                            # Interleave to increase diversity in what we show and pass forward
                            docs = interleave_docs_by_case(docs, per_case_limit=2, max_total=effective_k)
                        except Exception as ge:
                            _append_debug(f"[DEBUG][FetchChunks] error: {ge}")
                            # keep original docs
                
                    # Debug: Show what was retrieved
                    _debug_section("Retrieved Previews", _fmt_retrieved_previews([
                        {
                            "file_stem": (d.metadata or {}).get("file_stem"),
                            "preview": (d.page_content or "").strip().replace("\n", " ")[:200],
                        }
                        for d in docs[:12]
                    ]), note="First few chunks shown per case; this is what the filtration LLM sees.")

                    # Extract case file stems and load metadata (prefer fused order if available)
                    if 'fused_cases' in locals() and fused_cases:
                        stems = [fs for fs in fused_cases if fs]
                    else:
                        stems = sorted({(d.metadata or {}).get("file_stem") for d in docs if (d.metadata or {}).get("file_stem")})
                    case_metas = _load_case_metadata_for_stems(stems, qp_breadth)
                    _append_debug(f"[DEBUG][Cases] loaded metadata for {len(case_metas)} cases (ranked, breadth={qp_breadth})")
                    retrieval_detail = f"{len(docs)} chunk(s) across {len(stems)} case(s)"
                    if general_law_retrieval:
                        retrieval_detail += " (general-law precedents)"
                    tracker.stage_complete("retrieval", retrieval_detail)
                    tracker.stage_running("filtration", "Filtering relevant documents…")

                    query_context_payload = {
                        "breadth": qp_breadth,
                        "fused_case_count": len(stems),
                        "retrieval_k": effective_k,
                        "desired_min_full_docs": effective_min_docs,
                        "planner_min_full_docs": int(getattr(qp, "min_full_docs", effective_min_docs) or effective_min_docs),
                    }
                    if 'fused_cases' in locals() and fused_cases:
                        query_context_payload["fused_ranked_cases"] = fused_cases[:20]

                    # Step 3: Filtration - Use LLM to select most relevant chunks and cases
                    mode_key = "chunk" if filtration_mode == "chunk context filtration" else "metadata"
                    plan = filtration_retriever(
                        retr_q,
                        docs,
                        mode=mode_key,
                        case_metadatas=case_metas,
                        desired_min_full_docs=effective_min_docs,
                        desired_max_chunks_per_case=max_chunks_per_case,
                        query_context=query_context_payload,
                    )
                
                    # Step 4: Named Case Guard - Ensure named cases are included
                    plan = apply_named_case_guard(plan, retr_q, docs)
                    # Step 4b: Diversity Guard - Enforce minimum full cases and cap chunks per case
                    try:
                        plan = enforce_case_diversity(
                            plan,
                            fused_cases=fused_cases if 'fused_cases' in locals() else None,
                            desired_min_full_docs=effective_min_docs,
                            desired_max_chunks_per_case=max_chunks_per_case,
                            breadth=qp_breadth,
                            allow_expansion=bool(getattr(plan, "request_more_cases", False) and qp_breadth == "broad"),
                        )
                        _append_debug(f"[DEBUG][Guard] after diversity: full_docs={len(plan.selected_full_docs)} chunks={len(plan.selected_chunks)}")
                    except Exception as ge2:
                        _append_debug(f"[DEBUG][Guard] error: {ge2}")
                    tracker.stage_complete(
                        "filtration",
                        f"{len(plan.selected_full_docs)} full doc(s); {len(plan.selected_chunks)} chunk(s)",
                    )
                    tracker.stage_running("assembly", "Assembling context…")
                
                    # Step 5: Context Assembly - Build final context from plan
                    # Effective budget: use model context window minus reserved completion headroom
                    ctx_limit = MODEL_CONTEXT_WINDOWS.get(model_name, 32768)
                    headroom = RESERVED_COMPLETION_TOKENS.get(model_name, 8000)
                    budget_override = max(4000, ctx_limit - headroom)
                    context, _, dbg = assemble_context_from_plan(
                        plan,
                        retr_q,
                        vs,
                        txt_dir="processed_data/txt_data",
                        budget_tokens=budget_override,
                        initial_docs=docs,
                    )
                    ctx_tokens_est = 0
                    try:
                        ctx_tokens_est = int(dbg.get("est_tokens", 0))
                    except Exception:
                        ctx_tokens_est = 0
                    tracker.stage_complete("assembly", f"context ≈ {ctx_tokens_est} tokens")
                    if qp_breadth == "broad" and len(plan.selected_full_docs) < max(4, effective_min_docs):
                        _append_debug(
                            f"[DEBUG][Coverage] Broad query retained {len(plan.selected_full_docs)} full docs (< target {max(4, effective_min_docs)})."
                        )
                    # Debug: Log filtration plan details
                    try:
                        import json as _json
                        _debug_section("Filtration Plan", _fmt_filtration_plan(plan.model_dump()), note="LLM's selection of full cases and chunks.")
                    except Exception:
                        _append_debug(f"[DEBUG][Filtration] plan: full_docs={plan.selected_full_docs}, sel_chunks={len(plan.selected_chunks)}, budget={plan.context_budget_tokens}")
                
                    # Log reasoning for selected documents and chunks
                    if getattr(plan, "overall_reasoning", None):
                        _append_debug(f"[DEBUG][Filtration] overall_reasoning: {plan.overall_reasoning}")
                    if getattr(plan, "reasoning_full_docs", None):
                        for rd in plan.reasoning_full_docs:
                            try:
                                _append_debug(f"[DEBUG][Filtration] full_doc_reason: file_stem={rd.file_stem} reason={rd.reason}")
                            except Exception:
                                pass
                    if getattr(plan, "reasoning_chunks", None):
                        for rc in plan.reasoning_chunks:
                            try:
                                _append_debug(f"[DEBUG][Filtration] chunk_reason: file_stem={rc.file_stem} idx={rc.chunk_index} reason={rc.reason}")
                            except Exception:
                                pass
                    if getattr(plan, "request_more_cases", False):
                        _append_debug(f"[DEBUG][Filtration] request_more_cases=True reason={getattr(plan, 'expansion_reason', '') or 'not provided'}")
                    if getattr(plan, "notes", None):
                        _append_debug(f"[DEBUG][Filtration] notes: {plan.notes}")
                
                    # Report context vs model limits
                    try:
                        ctx_tokens = dbg.get("est_tokens", 0)
                    except Exception:
                        ctx_tokens = 0
                    limit_tokens = MODEL_CONTEXT_WINDOWS.get(model_name, 32768)
                    fit_note = "within limit" if ctx_tokens <= limit_tokens else "exceeds limit"
                    _debug_section(
                        "Assembler",
                        _fmt_assembler(dbg) + f"\ncontext tokens: {ctx_tokens} / limit: {limit_tokens} ({fit_note})\nEffective budget tokens (messages): {budget_override}\nReserved for completion: {headroom}",
                        note="How the final context was built from selected items; includes token estimate and model limit check.",
                    )
                
                    # Fallback: if filtration produced no context, use all retrieved chunks
                    if not context:
                        context = "\n\n".join(d.page_content for d in docs)
                        _append_debug(f"[DEBUG] Filtration empty; fallback context length: {len(context)}")
                    if general_law_retrieval:
                        strategy = "general_law+rag"
                        case_probe_context = context
                        case_probe_plan = plan
                        case_probe_sources = _case_sources_from_plan(plan) if plan else []
                        case_probe_sources = [s for s in case_probe_sources if s]
                        case_probe_detail = (
                            f"{len(plan.selected_full_docs)} full doc(s); {len(plan.selected_chunks)} chunk(s)"
                            if plan
                            else f"{len(stems)} case(s) retrieved"
                        )
                        tracker.stage_complete("case_probe", case_probe_detail or "Precedents prepared")
                        case_probe_stage_handled = True
                    else:
                        strategy = "fresh_retrieval"

                if general_law_retrieval and not case_probe_stage_handled:
                    tracker.stage_skip("case_probe", case_probe_detail or "No precedents retrieved.")
                    case_probe_stage_handled = True

                if not case_probe_stage_handled:
                    tracker.stage_skip("case_probe", "Not required (retrieval pipeline).")

                # Step 6: Answer Generation - Generate response using LLM
                if 'qp' in locals() and getattr(qp, 'type', '') == 'general_law':
                    # General legal knowledge - no document constraints
                    system_prefix = (
                        "You are the primary advisory voice for an Indian legal assistant. "
                        "Deliver empathetic, actionable guidance grounded in Indian law unless the user specifies another jurisdiction. "
                        "Structure your reply under the heading 'General Guidance'. "
                        "Explain key rights, immediate steps, procedural options, and statutory hooks (e.g., IPC, CrPC, PC Act) relevant to the scenario. "
                        "Keep the focus on universal principles—do not cite specific cases or rely on any precedent context, because a separate module will append case-based insights. "
                        "Flag uncertainties, urge the user to consult a qualified lawyer, and avoid definitive promises about outcomes."
                    )
                    persona_note = _persona_prompt_note()
                    if persona_note:
                        system_prefix += f" {persona_note}"
                    prompt = f"{system_prefix}\n\nQUESTION: {user_q}"
                else:
                    # RAG mode - must use only provided context
                    system_prefix = (
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
                        "  • Delay beyond 15 days beyond initial 30 days is jurisdictionally barred (file 2.txt).\n"
                        "  • Certified copy requirement under Limitation Act §12(3) applies only if application filed (file 2.txt).\n"
                        "Outcome: Appeals dismissed; NCLAT order upheld.\n\n"
                        "Sources: 2.txt\n\n"
                        "========================\n"
                        "### TEMPLATE B — SINGLE CASE DEEP ANALYSIS (In-Depth)\n"
                        "========================\n"
                        "[Case: <Case Name> | File: <N.txt>]\n"
                        "Court / Date / Citation\n\n"
                        "Overview / Holding (2–3 sentences)\n"
                        "A concise statement of the ruling and key principle.\n\n"
                        "Facts (essential only)\n"
                        "• …\n"
                        "• …\n\n"
                        "Issues\n"
                        "• …\n\n"
                        "Held / Disposition\n"
                        "• … (appeal allowed / dismissed / directions / etc.)\n\n"
                        "Reasoning (text-supported)\n"
                        "1) … — short quote if relevant (Case — file N.txt, chunk X)\n"
                        "2) …\n"
                        "3) …\n\n"
                        "Rule / Ratio\n"
                        "• …\n\n"
                        "Statutes / Provisions Cited\n"
                        "• …\n\n"
                        "Limits / Caveats\n"
                        "• …\n\n"
                        "Practical Takeaways\n"
                        "• …\n\n"
                        "Sources: N.txt\n\n"
                        "Example:\n"
                        "[Case: A. John Kennedy etc. v. State of Tamil Nadu & Ors. | File: 1.txt]\n"
                        "SC (24 Mar 2025) — 2025 INSC 443\n\n"
                        "Overview / Holding:\n"
                        "Supreme Court continued its environmental mandamus, ordering a Central Empowered Committee survey "
                        "to restore the Agasthyamalai forest landscape, while deferring rehabilitation issues.\n\n"
                        "Facts:\n"
                        "• Tea estate leases in reserve forest; competing claims between displaced workers and conservation authorities.\n"
                        "• High Court closed PILs without concrete restoration plan (file 1.txt).\n\n"
                        "Issue:\n"
                        "• How to ensure forest restoration and biodiversity protection while handling workers’ rehabilitation claims.\n\n"
                        "Held / Disposition:\n"
                        "• Directed a scientific survey using satellite imagery and geo-mapping within 12 weeks; matter relisted for follow-up; "
                        "rehabilitation issue to be heard separately (file 1.txt).\n\n"
                        "Reasoning:\n"
                        "1) Ecocentric over anthropocentric approach — forest protection is constitutional necessity.\n"
                        "2) Ongoing Godavarman line of cases supports continued judicial oversight.\n\n"
                        "Rule / Ratio:\n"
                        "• Critical tiger habitats demand the highest level of protection; restoration orders may proceed through continuing mandamus (file 1.txt).\n\n"
                        "Statutes Cited:\n"
                        "• Wildlife (Protection) Act, 1972; Forest (Conservation) Act, 1980; Tamil Nadu Forests Act, 1882.\n\n"
                        "Limits:\n"
                        "• No final ruling on individual worker claims in this order.\n\n"
                        "Takeaways:\n"
                        "• Forest restoration given primacy over economic rehabilitation; court retains seisin pending report.\n\n"
                        "Sources: 1.txt\n"
                    )
                    prompt = f"{system_prefix}\n\nQUESTION: {user_q}\n\nCONTEXT:\n{context}"
            
                # Generate response (branch for general-law vs RAG)
                tracker.stage_running("answer", "Generating answer…")
                answer = ""
                if general_chat_mode:
                    try:
                        st.session_state.chat_history.add_user_message(user_q)
                    except Exception:
                        pass
                    answer = _handle_general_chat(user_q, llm)
                    with messages_container:
                        with st.chat_message("assistant", avatar="⚖️"):
                            st.markdown(answer)
                    tracker.stage_complete("answer", "General chat response.")
                elif general_law_mode:
                    try:
                        st.session_state.chat_history.add_user_message(user_q)
                    except Exception:
                        pass

                    try:
                        general_text = _collect_stream_text(llm, prompt) or ""
                    except Exception as e:
                        err_str = str(e)
                        prov = provider_name
                        detail = None
                        try:
                            import json as _json
                            detail = _json.dumps(getattr(e, "__dict__", {}), indent=2)
                        except Exception:
                            detail = None
                        answer = f"[Error] Provider returned error\nprovider={prov}\nmessage={err_str}\n{('details=' + detail) if detail else ''}"
                        _debug_section("Provider Error", {"provider": prov, "error": err_str, "details": detail}, note="Captured at advisory LLM failure; check provider diagnostics.")
                        tracker.stage_error("answer", err_str)
                        with messages_container:
                            with st.chat_message("assistant", avatar="⚖️"):
                                st.markdown(answer)
                    else:
                        general_text = general_text.strip()
                        case_text = ""
                        if case_probe_context:
                            case_system_prefix = (
                                "You are an Indian legal research assistant augmenting an advisory reply. "
                                "Use ONLY the provided case-law context to highlight precedents that complement the general guidance already shared with the user. "
                                "Write a section titled 'Relevant Precedents' with concise takeaways (2-3 sentences or bullets) for up to three cases. "
                                "Cite statutes mentioned in the context when helpful. End with a line of the form 'Sources: <file ids>'."
                            )
                            case_prompt = (
                                f"{case_system_prefix}\n\nUSER QUESTION: {user_q}\n\nRETRIEVAL QUERY: {case_probe_query}\n\nCONTEXT:\n{case_probe_context}"
                            )
                            try:
                                case_text = _collect_stream_text(llm, case_prompt).strip()
                            except Exception as ce:
                                _append_debug(f"[DEBUG][CaseProbe][summary_error] {ce}")
                                case_text = ""
                            if case_text:
                                if case_probe_sources and "Sources:" not in case_text:
                                    case_text = case_text.rstrip() + f"\n\nSources: {', '.join(case_probe_sources)}"
                            elif case_probe_sources:
                                bullets = "\n".join(f"- Refer to {stem}.txt" for stem in case_probe_sources[:3])
                                case_text = f"Relevant Precedents\n{bullets}\n\nSources: {', '.join(case_probe_sources)}"
                        if not case_text and case_probe_stage_handled and case_probe_detail:
                            cleaned_detail = case_probe_detail.strip()
                            if cleaned_detail.endswith("."):
                                cleaned_detail = cleaned_detail[:-1]
                            case_text = f"Relevant Precedents\n- {cleaned_detail}.\n"

                        parts = [part for part in [general_text, case_text] if part]
                        final_answer = "\n\n---\n\n".join(parts).strip()
                        answer = final_answer or general_text or case_text
                        if not answer:
                            answer = "I'm sorry — I couldn't generate an answer right now."

                        def _final_stream():
                            yield answer

                        with messages_container:
                            with st.chat_message("assistant", avatar="⚖️"):
                                final_text = st.write_stream(_final_stream())
                                if isinstance(final_text, str) and not answer:
                                    answer = final_text
                        tracker.stage_complete("answer", "Answer delivered.")
                else:
                    try:
                        def _stream_gen():
                            nonlocal answer
                            st.session_state.chat_history.add_user_message(user_q)
                            result = llm.stream(prompt)
                            for chunk in result:
                                content = getattr(chunk, "content", None)
                                if content:
                                    answer += content
                                    yield content

                        with messages_container:
                            with st.chat_message("assistant", avatar="⚖️"):
                                final_text = st.write_stream(_stream_gen())
                                if isinstance(final_text, str) and not answer:
                                    answer = final_text
                        tracker.stage_complete("answer", "Answer delivered.")
                    except Exception as e:
                        err_str = str(e)
                        prov = provider_name
                        detail = None
                        try:
                            import json as _json
                            detail = _json.dumps(getattr(e, "__dict__", {}), indent=2)
                        except Exception:
                            detail = None
                        answer = f"[Error] Provider returned error\nprovider={prov}\nmessage={err_str}\n{('details=' + detail) if detail else ''}"
                        _debug_section("Provider Error", {"provider": prov, "error": err_str, "details": detail}, note="Captured at stream failure; check provider dashboard if needed.")
                        tracker.stage_error("answer", err_str)

            finally:
                tracker.finish()

            # Step 7: Update Session State - Store results for next query
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # Store assistant message in chat history
            try:
                st.session_state.chat_history.add_ai_message(answer)
            except Exception:
                pass

            # Check if we have many chunks from one case (suggests full case available)
            full = maybe_full_case(docs)
            if full:
                _append_debug(f"[Hint] Many chunks from one case — full judgment available: {full}")

            # Update session state for next query (enables follow-up questions)
            if not general_chat_mode:
                st.session_state.last_context = context
                st.session_state.last_docs = docs
                st.session_state.last_stems = stems
                st.session_state.last_filters = {"court_name": court_name, "statutes": getattr(qp, "statutes", None) or statutes}
                st.session_state.last_question_rewrite = getattr(qp, "rewrite", None) or user_q
            _append_debug(f"[DEBUG][Router] strategy={strategy}")

    # Debug Tab - Show detailed logs
    with debug_tab:
        if st.button("Clear logs", use_container_width=True):
            st.session_state.debug_logs = []
            st.session_state["debug_string"] = ""
        
        logs_text = "\n".join(st.session_state.debug_logs) if st.session_state.get("debug_logs") else ""
        filt = st.text_input("Filter log (case-insensitive)", value="", help="Show only lines containing this text")
        if filt:
            try:
                lines = logs_text.splitlines()
                match = filt.lower()
                lines = [ln for ln in lines if match in ln.lower()]
                view_text = "\n".join(lines)
            except Exception:
                view_text = logs_text
        else:
            view_text = logs_text
        st.text_area("Debug log", value=view_text or "(log empty)", height=700, label_visibility="collapsed")
        st.download_button("Download log", data=logs_text, file_name="debug.log", mime="text/plain", use_container_width=True)


if __name__ == "__main__":
    main()
