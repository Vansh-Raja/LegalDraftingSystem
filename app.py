"""
Main Streamlit application for Legal Drafting System
Provides a web interface for legal document Q&A using RAG (Retrieval-Augmented Generation)
"""

import os
import json
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

            tracker = PipelineTracker()
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

            # Decide effective Top-K (retrieval depth)
            auto_topk_enabled = bool(st.session_state.get("auto_topk", True))
            if auto_topk_enabled and getattr(qp, "retrieval_k", None):
                try:
                    effective_k = int(qp.retrieval_k)
                except Exception:
                    effective_k = 16 if getattr(qp, "type", "") == "general_law" else 6
                # Clamp to a reasonable range
                effective_k = max(3, min(20, effective_k))
                k_source = "qp.retrieval_k"
            elif auto_topk_enabled:
                # Fallback heuristic if planner did not set K
                effective_k = 16 if getattr(qp, "type", "") == "general_law" else 6
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
            if getattr(qp, "type", "") == "general_law":
                effective_min_docs = 0
                min_docs_source = "not_applicable"
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

            # Step 2: Context Strategy - Choose how to handle the query
            tracker.stage_running("retrieval", "Retrieving information…")
            retrieval_notes: List[str] = []
            if qp.type == "general_law":
                # General legal knowledge - no document retrieval needed
                strategy = "general_law"
                # context remains empty
                tracker.stage_skip("retrieval", "General-law routing (no retrieval).")
                tracker.stage_skip("filtration", "General-law routing (skipped).")
                tracker.stage_skip("assembly", "General-law routing (skipped).")
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
                        dominant = stems[0]  # Most frequent case from last query
                        full_path = Path("processed_data/txt_data") / f"{dominant}.txt"
                        if full_path.exists():
                            raw = full_path.read_text(encoding="utf-8")
                            header = f"[file: {dominant}.txt]\n"
                            context = header + raw
                            strategy = "same_case_full"
                            _append_debug(f"[DEBUG][Bridge] same_case_full loaded {dominant}.txt")
                            _debug_section(
                                "Context After Same Case Full",
                                {
                                    "file_loaded": dominant,
                                    "chars": len(context),
                                    "est_tokens": len(context) // 4,
                                },
                                note="Entire dominant case inserted as context for follow-up answer.",
                            )
                    except Exception as se:
                        _append_debug(f"[DEBUG][Bridge] same_case_full error: {se}")
                reuse_detail = "; ".join(retrieval_notes) if retrieval_notes else "Reused previous context"
                tracker.stage_complete("retrieval", reuse_detail)
                tracker.stage_skip("filtration", "Context reused from previous turn.")
                tracker.stage_skip("assembly", "Context reused from previous turn.")
            else:
                # Step 2b: Fresh Retrieval - New query requires document search
                retr_q = qp.rewrite or user_q
                
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
                strategy = "fresh_retrieval"

            # Step 6: Answer Generation - Generate response using LLM
            if 'qp' in locals() and getattr(qp, 'type', '') == 'general_law':
                # General legal knowledge - no document constraints
                system_prefix = (
                    "You are a **legal expert**. Answer general legal questions clearly and concisely. "
                    "Default to Indian law if jurisdiction is not specified. Provide statute names/sections when relevant (e.g., IPC s.302, CrPC s.482), "
                    "and note jurisdictional variations if applicable. Avoid case-specific claims unless cases are explicitly provided. "
                    "Do NOT invent case citations. If the question requires specific documents, ask for them."
                )
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
            
            # Generate streaming response
            tracker.stage_running("answer", "Generating answer…")
            answer = ""
            try:
                def _stream_gen():
                    nonlocal answer
                    # Add user message to chat history
                    st.session_state.chat_history.add_user_message(user_q)
                    # Ensure OpenRouter models also get the same prompt (already built above)
                    result = llm.stream(prompt)
                    for chunk in result:
                        # Extract content from chunk (may be AIMessage)
                        content = getattr(chunk, "content", None)
                        if content:
                            answer += content
                            yield content
                
                # Display streaming response
                with messages_container:
                    with st.chat_message("assistant", avatar="⚖️"):
                        final_text = st.write_stream(_stream_gen())
                        if isinstance(final_text, str) and not answer:
                            answer = final_text
                tracker.stage_complete("answer", "Answer delivered.")
            except Exception as e:
                # Try to extract provider error details if present
                err_str = str(e)
                prov = provider_name
                # Common fields that sometimes appear on OpenRouter-like errors
                detail = None
                try:
                    # If error is JSON-ish
                    import json as _json
                    detail = _json.dumps(getattr(e, "__dict__", {}), indent=2)
                except Exception:
                    detail = None
                answer = f"[Error] Provider returned error\nprovider={prov}\nmessage={err_str}\n{('details=' + detail) if detail else ''}"
                _debug_section("Provider Error", {"provider": prov, "error": err_str, "details": detail}, note="Captured at stream failure; check provider dashboard if needed.")
                tracker.stage_error("answer", err_str)

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
            st.session_state.last_context = context
            st.session_state.last_docs = docs
            st.session_state.last_stems = stems
            st.session_state.last_filters = {"court_name": court_name, "statutes": statutes}
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
