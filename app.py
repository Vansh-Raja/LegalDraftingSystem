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
    # Sidebar split: Filtration vs Chat LLM settings
    with st.sidebar.expander("Filtration LLM", expanded=True):
        filtration_mode = st.selectbox(
            "LLM filtration model mode",
            ["chunk context filtration", "metadata filtration"],
            index=0,
            key="filtration_mode_select",
        )
        # Retrieval knobs
        k_val = st.slider("Top-K chunks", min_value=3, max_value=12, value=6, step=1, key="k_chunks")
        court_choice = st.selectbox(
            "Court filter",
            ["Supreme Court of India", "All courts (no filter)"] ,
            index=0,
            key="court_filter",
        )
        statutes_text = st.text_input("Statute filters (comma-separated)", value="", key="statute_filters_text")

    with st.sidebar.expander("Chat LLM", expanded=True):
        models = ["gpt-5-nano-2025-08-07", "qwen3:latest", ]
        model = st.selectbox("Model", models, index=0, key="chat_model_select")
    # Initialize LLM
    if model == "gpt-5-nano-2025-08-07":
        api_key = os.getenv("OPENAI_KEY")
        if not api_key:
            st.sidebar.warning("OPENAI_KEY not set; falling back to qwen3:latest")
            llm = ChatOllama(model="qwen3:latest", temperature=0, streaming=True, num_ctx=40000)
        else:
            llm = ChatOpenAI(model="gpt-5-nano-2025-08-07", temperature=0, streaming=True, api_key=api_key)
    else:
        llm = ChatOllama(model="qwen3:latest", temperature=0, streaming=True, num_ctx=40000)
    statutes = [s.strip() for s in statutes_text.split(",") if s.strip()] or None
    court_name = None if court_choice.startswith("All") else "Supreme Court of India"
    return llm, model, filtration_mode, k_val, court_name, statutes


def _ensure_session_state():
    if "history" not in st.session_state:
        st.session_state.history = InMemoryChatMessageHistory()
    if "debug_logs" not in st.session_state:
        st.session_state.debug_logs = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = InMemoryChatMessageHistory()


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
    llm, model_name, filtration_mode, k_val, court_name, statutes = _init_models()

    vs = get_vectorstore()
    retriever = build_retriever(vs, court_name=court_name, k=k_val)

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

        user_q = st.chat_input("Ask a legal question...", key="chat_input_main")
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

            # Build case metadata for retrieved stems
            stems = sorted({(d.metadata or {}).get("file_stem") for d in docs if (d.metadata or {}).get("file_stem")})
            case_metas = []
            for fs in stems:
                try:
                    if not fs:
                        continue
                    mp = Path("processed_data/metadata") / f"{fs}.json"
                    if mp.exists():
                        case_metas.append({"file_stem": fs, "metadata": __import__("json").loads(mp.read_text(encoding="utf-8"))})
                except Exception:
                    pass

            # Filtration-based context
            mode_key = "chunk" if filtration_mode == "chunk context filtration" else "metadata"
            plan = filtration_retriever(user_q, docs, mode=mode_key, case_metadatas=case_metas)
            # Named-case guard to force include target case if query names it
            plan = apply_named_case_guard(plan, user_q, docs)
            # Increase context budget for GPT-5 Nano (supports very large context windows)
            budget_override = 400000 if model_name == "gpt-5-nano-2025-08-07" else plan.context_budget_tokens
            context, _, dbg = assemble_context_from_plan(
                plan,
                user_q,
                vs,
                txt_dir="processed_data/txt_data",
                budget_tokens=budget_override,
                initial_docs=docs,
            )
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
            _append_debug(f"[DEBUG][Filtration] assembler: est_tokens~{dbg.get('est_tokens')}, spans={dbg.get('spans')[:5]}, budget_used={budget_override}")
            if not context:
                # Fallback simple context if filtration empty
                context = "\n\n".join(d.page_content for d in docs)
                _append_debug(f"[DEBUG] Filtration empty; fallback context length: {len(context)}")
            # No context preview to avoid noise in logs

            # Answer (streaming) with memory
            system_prefix = (
                "You are a **legal expert helper** in a RAG system. You will be given a user question and context assembled from retrieved chunks/full cases. Use **only** the provided context; do **not** use external knowledge or invent facts. After you generate the answer, **self-verify**: recheck that the user's question is most accurately matched by the selected context; if not, qualify your answer or state that information is insufficient.\n\n"
                "Guidelines:\n\n"
                "1. If the user’s question names a **specific case** (by parties, court, date, or case number), **prioritize** that case and **limit** use of other cases unless strictly needed.\n"
                "2. If the question is about a **legal topic, statute, or section**, you may **synthesize** across multiple relevant documents.\n"
                "3. Your answer should be **precise, well-reasoned, and evidence-based**. You may include **short quotes (≤ 2 sentences)** from the context, with citations (case name + chunk metadata).\n"
                "4. After drafting the answer, **self-check**: verify every factual claim has supporting text in the context. If any claim is not verified, **remove or qualify** it.\n"
                "5. If the context is **insufficient**, respond: ‘I’m sorry — I don’t know based on the provided documents.’\n"
                "6. At the end, list the **sources used** (case name + chunk metadata).\n\n"
                "Finally, add a single line ‘Sources: N.txt, …’ by extracting file headers like [file: N.txt] present in the context."
            )
            prompt = f"{system_prefix}\n\nQUESTION: {user_q}\n\nCONTEXT:\n{context}"
            answer = ""
            try:
                def _stream_gen():
                    nonlocal answer
                    # LangChain chat models expose .stream on instances
                    # Use RunnableWithMessageHistory equivalent: we manually maintain history
                    st.session_state.chat_history.add_user_message(user_q)
                    result = llm.stream(prompt)
                    for chunk in result:
                        # chunk may be an AIMessage with .content
                        content = getattr(chunk, "content", None)
                        if content:
                            answer += content
                            yield content
                with st.chat_message("assistant", avatar="⚖️"):
                    final_text = st.write_stream(_stream_gen())
                    if isinstance(final_text, str) and not answer:
                        answer = final_text
            except Exception as e:
                answer = f"[Error] {e}"

            st.session_state.messages.append({"role": "assistant", "content": answer})
            # store assistant message in memory
            try:
                st.session_state.chat_history.add_ai_message(answer)
            except Exception:
                pass

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


