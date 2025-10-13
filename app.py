import os
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from rag import (
    get_vectorstore,
    build_retriever,
    maybe_full_case,
    filtration_retriever,
    assemble_context_from_plan,
    apply_named_case_guard,
)
from st_debug import debug as sdebug


def _init_models():
    load_dotenv()
    models = ["gpt-5-nano-2025-08-07", "qwen3:latest", ]
    st.sidebar.header("Settings")
    model = st.sidebar.selectbox("Model", models, index=0)
    # Initialize LLM
    if model == "gpt-5-nano-2025-08-07":
        api_key = os.getenv("OPENAI_KEY")
        if not api_key:
            st.sidebar.warning("OPENAI_KEY not set; falling back to qwen3:latest")
            llm = ChatOllama(model="qwen3:latest", temperature=0, streaming=False, num_ctx=40000)
        else:
            llm = ChatOpenAI(model="gpt-5-nano-2025-08-07", temperature=0, streaming=False, api_key=api_key)
    else:
        llm = ChatOllama(model="qwen3:latest", temperature=0, streaming=False, num_ctx=40000)
    return llm, model


def _ensure_session_state():
    if "history" not in st.session_state:
        st.session_state.history = InMemoryChatMessageHistory()
    if "debug_logs" not in st.session_state:
        st.session_state.debug_logs = []
    if "messages" not in st.session_state:
        st.session_state.messages = []


def _reset_chat_state():
    st.session_state.messages = []
    st.session_state.history = InMemoryChatMessageHistory()
    st.session_state.debug_logs = []
    st.session_state["debug_string"] = ""


def _append_debug(msg: str):
    st.session_state.debug_logs.append(msg)
    try:
        sdebug(msg)
    except Exception:
        pass


def main():
    st.set_page_config(page_title="Legal RAG Chat", page_icon="⚖️", layout="wide")
    st.title("⚖️ Legal Drafting Chatbot")
    _ensure_session_state()
    llm, model_name = _init_models()

    vs = get_vectorstore()
    retriever = build_retriever(vs, court_name="Supreme Court of India", k=6)

    # Split UI into Chat and Debug tabs
    chat_tab, debug_tab = st.tabs(["Chat", "Debug"])

    with chat_tab:
        # Chat UI controls
        cols = st.columns([1, 1, 6])
        with cols[0]:
            st.button("Clear chat & context", on_click=_reset_chat_state, key="clear_chat")

        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        user_q = st.chat_input("Ask a legal question...")
        if user_q:
            st.session_state.messages.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)

            # Retrieve
            docs = retriever.invoke(user_q)
            if not docs:
                retriever_relaxed = build_retriever(vs, court_name=None, k=6)
                docs = retriever_relaxed.invoke(user_q)

            # Diagnostics
            _append_debug("[DEBUG] Retrieved docs:")
            for i, d in enumerate(docs[:6], 1):
                fs = d.metadata.get("file_stem")
                preview = (d.page_content or "").strip().replace("\n", " ")[:160]
                _append_debug(f"  {i}. file_stem={fs} ... {preview}")

            # Filtration-based context
            plan = filtration_retriever(user_q, docs)
            # Named-case guard to force include target case if query names it
            plan = apply_named_case_guard(plan, user_q, docs)
            context, _, dbg = assemble_context_from_plan(plan, user_q, vs, txt_dir="processed_data/txt_data", budget_tokens=plan.context_budget_tokens, initial_docs=docs)
            try:
                import json as _json
                _append_debug(f"[DEBUG][Filtration] plan JSON: {_json.dumps(plan.model_dump(), indent=2)}")
            except Exception:
                _append_debug(f"[DEBUG][Filtration] plan: full_docs={plan.selected_full_docs}, sel_chunks={len(plan.selected_chunks)}, budget={plan.context_budget_tokens}")
            # Also print any provided reasoning fields
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
            _append_debug(f"[DEBUG][Filtration] assembler: est_tokens~{dbg.get('est_tokens')}, spans={dbg.get('spans')[:5]}")
            if not context:
                # Fallback simple context if filtration empty
                context = "\n\n".join(d.page_content for d in docs)
                _append_debug(f"[DEBUG] Filtration empty; fallback context length: {len(context)}")
            # No context preview to avoid noise in logs

            # Answer (non-streaming for UI simplicity)
            system_prefix = (
                "You are a legal RAG assistant. Use only the provided context."
            )
            prompt = f"{system_prefix}\n\nQUESTION: {user_q}\n\nCONTEXT:\n{context}"
            try:
                resp = llm.invoke(prompt)
                answer = getattr(resp, "content", None) or str(resp)
            except Exception as e:
                answer = f"[Error] {e}"

            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)

            full = maybe_full_case(docs)
            if full:
                _append_debug(f"[Hint] Many chunks from one case — full judgment available: {full}")

    with debug_tab:
        dcols = st.columns([1, 5])
        with dcols[0]:
            if st.button("Clear logs"):
                st.session_state.debug_logs = []
                st.session_state["debug_string"] = ""
        logs_text = "\n".join(st.session_state.debug_logs) if st.session_state.get("debug_logs") else ""
        st.text_area("Debug logs", value=logs_text, height=320, label_visibility="collapsed")


if __name__ == "__main__":
    main()


