"""
Main Streamlit application for Legal Drafting System
Provides a web interface for legal document Q&A using RAG (Retrieval-Augmented Generation)
"""

import os
from pathlib import Path
from typing import List

import streamlit as st
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
)
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
        auto_topk = st.toggle("Auto Top-K (LLM)", value=True, key="auto_topk", help="Let the Query Processor set retrieval K based on question type")
        if not auto_topk:
            k_val = st.slider("Top-K chunks", min_value=3, max_value=20, value=6, step=1, key="k_chunks")
        else:
            # Keep a placeholder for UI state when auto mode is on
            k_val = st.session_state.get("k_chunks", 6)
        court_choice = st.selectbox(
            "Court filter",
            ["Supreme Court of India", "All courts (no filter)"] ,
            index=0,
            key="court_filter",
        )
        statutes_text = st.text_input("Statute filters (comma-separated)", value="", key="statute_filters_text")
        query_sort_mode = st.selectbox(
            "Query sorting mode",
            ["Manual (no general law)", "Auto (allow general law)"],
            index=0,
            key="query_sort_mode",
            help="Manual: force retrieval (new/followup). Auto: allow general-law routing for very broad queries.",
        )

    # Create sidebar controls for chat model selection
    with st.sidebar.expander("Chat LLM", expanded=True):
        models = [
            "gpt-5-nano-2025-08-07",            # OpenAI (Responses API via LangChain)
            "qwen3:latest",                     # Ollama local
            # OpenRouter models (via OpenAI-compatible API)
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b:free",
            "qwen/qwen3-235b-a22b:free",
            "qwen/qwen3-14b",
        ]
        model = st.selectbox("Model", models, index=0, key="chat_model_select")
    
    # Initialize the chat language model based on user selection
    openrouter_models = {
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b:free",
        "qwen/qwen3-235b-a22b:free",
        "qwen/qwen3-14b",
    }
    if model == "gpt-5-nano-2025-08-07":
        api_key = os.getenv("OPENAI_KEY")
        if not api_key:
            st.sidebar.warning("OPENAI_KEY not set; falling back to qwen3:latest")
            llm = ChatOllama(model="qwen3:latest", temperature=0, streaming=True, num_ctx=40000)
        else:
            llm = ChatOpenAI(model="gpt-5-nano-2025-08-07", temperature=0, streaming=True, api_key=api_key)
    elif model in openrouter_models:
        or_key = os.getenv("OPENROUTER_API_KEY")
        if not or_key:
            st.sidebar.warning("OPENROUTER_API_KEY not set; falling back to qwen3:latest")
            llm = ChatOllama(model="qwen3:latest", temperature=0, streaming=True, num_ctx=40000)
        else:
            # Use OpenRouter via OpenAI-compatible LangChain client
            llm = ChatOpenAI(
                model=model,
                temperature=0,
                streaming=True,
                api_key=or_key,
                base_url="https://openrouter.ai/api/v1",
            )
    else:
        llm = ChatOllama(model="qwen3:latest", temperature=0, streaming=True, num_ctx=40000)
    
    # Parse statute filters from comma-separated text
    statutes = [s.strip() for s in statutes_text.split(",") if s.strip()] or None
    court_name = None if court_choice.startswith("All") else "Supreme Court of India"
    
    return llm, model, filtration_mode, k_val, court_name, statutes, query_sort_mode


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
    st.session_state.history = InMemoryChatMessageHistory()
    st.session_state.debug_logs = []
    st.session_state["debug_string"] = ""


def _append_debug(msg: str):
    """
    Add a debug message to the session state and debug window.
    
    Args:
        msg (str): Debug message to add
    """
    st.session_state.debug_logs.append(msg)
    try:
        sdebug(msg)  # Also add to Streamlit debug window
    except Exception:
        pass  # Ignore debug window errors


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
    llm, model_name, filtration_mode, k_val, court_name, statutes, query_sort_mode = _init_models()

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

            # Step 1: Query Processing - Classify and plan the query
            qp: QueryPlan = process_query(
                user_q,
                last_question_rewrite=st.session_state.last_question_rewrite or None,
                last_stems=st.session_state.last_stems,
                last_filters=st.session_state.last_filters,
                last_context_snippet=(st.session_state.last_context or "")[:4000],
                summary=st.session_state.rolling_summary or None,
            )
            _append_debug(f"[DEBUG][QP] plan: {qp.model_dump_json(indent=2)}")

            # If query sorting mode is Manual (no general law), override to retrieval
            if query_sort_mode.startswith("Manual") and getattr(qp, "type", "") == "general_law":
                qp.type = "new"
                # keep rewrite; force retrieval_k to moderate if missing
                if not getattr(qp, "retrieval_k", None):
                    qp.retrieval_k = 6
                _append_debug("[DEBUG][QP] Manual mode: general_law overridden to new (retrieval)")

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
            _append_debug(f"[DEBUG][TopK] effective_k={effective_k} source={k_source}")

            # Initialize variables for context assembly
            context = ""
            docs = []
            stems = []
            dbg = {"est_tokens": 0, "spans": []}
            strategy = ""

            # Step 2: Context Strategy - Choose how to handle the query
            if qp.type == "general_law":
                # General legal knowledge - no document retrieval needed
                strategy = "general_law"
                # context remains empty
            elif qp.type == "followup" and qp.keep_context and st.session_state.last_context:
                # Reuse context from previous query
                strategy = "reuse_last_context"
                context = st.session_state.last_context
                docs = st.session_state.last_docs or []
                stems = st.session_state.last_stems or []
                dbg = {"est_tokens": len(context)//4, "spans": []}
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
                            # Get case metadata for filtration
                            stems_refill = sorted({(d.metadata or {}).get("file_stem") for d in refill_docs if (d.metadata or {}).get("file_stem")})
                            case_metas_refill = []
                            for fs in stems_refill:
                                try:
                                    if not fs:
                                        continue
                                    mp = Path("processed_data/metadata") / f"{fs}.json"
                                    if mp.exists():
                                        case_metas_refill.append({"file_stem": fs, "metadata": __import__("json").loads(mp.read_text(encoding="utf-8"))})
                                except Exception:
                                    pass
                            
                            # Apply filtration to new documents
                            mode_key2 = "chunk" if filtration_mode == "chunk context filtration" else "metadata"
                            plan_refill = filtration_retriever(retr_q, refill_docs, mode=mode_key2, case_metadatas=case_metas_refill)
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
                    except Exception as se:
                        _append_debug(f"[DEBUG][Bridge] same_case_full error: {se}")
            else:
                # Step 2b: Fresh Retrieval - New query requires document search
                retr_q = qp.rewrite or user_q
                
                # Build retriever with query plan filters
                retriever2 = build_retriever(
                    vs,
                    statute_filters=qp.statutes or None,
                    court_name=court_name,
                    k=effective_k,
                )
                docs = retriever2.invoke(retr_q)
                
                # Fallback: if no results, relax court filter
                if not docs:
                    retriever_relaxed = build_retriever(vs, court_name=None, k=effective_k)
                    docs = retriever_relaxed.invoke(retr_q)
                
                # Debug: Show what was retrieved
                _append_debug("[DEBUG] Retrieved docs:")
                for i, d in enumerate(docs[:6], 1):
                    fs = d.metadata.get("file_stem")
                    preview = (d.page_content or "").strip().replace("\n", " ")[:160]
                    _append_debug(f"  {i}. file_stem={fs} ... {preview}")

                # Extract case file stems and load metadata
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

                # Step 3: Filtration - Use LLM to select most relevant chunks and cases
                mode_key = "chunk" if filtration_mode == "chunk context filtration" else "metadata"
                plan = filtration_retriever(retr_q, docs, mode=mode_key, case_metadatas=case_metas)
                
                # Step 4: Named Case Guard - Ensure named cases are included
                plan = apply_named_case_guard(plan, retr_q, docs)
                
                # Step 5: Context Assembly - Build final context from plan
                budget_override = 400000 if model_name == "gpt-5-nano-2025-08-07" else plan.context_budget_tokens
                context, _, dbg = assemble_context_from_plan(
                    plan,
                    retr_q,
                    vs,
                    txt_dir="processed_data/txt_data",
                    budget_tokens=budget_override,
                    initial_docs=docs,
                )
                # Debug: Log filtration plan details
                try:
                    import json as _json
                    _append_debug(f"[DEBUG][Filtration] plan JSON: {_json.dumps(plan.model_dump(), indent=2)}")
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
                
                _append_debug(f"[DEBUG][Filtration] assembler: est_tokens~{dbg.get('est_tokens')}, spans={dbg.get('spans')[:5]}")
                
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
                    "You are a **legal expert helper** in a RAG system. You will be given a user question and context assembled from retrieved chunks/full cases. Use **only** the provided context; do **not** use external knowledge or invent facts.\n\n"
                    "Guidelines:\n\n"
                    "1. If the user's question names a **specific case** (by parties, court, date, or case number), **prioritize** that case and **limit** use of other cases unless strictly needed.\n"
                    "2. If the question is about a **legal topic, statute, or section**, you may **synthesize** across multiple relevant documents.\n"
                    "3. Your answer should be **precise, well-reasoned, and evidence-based**. You may include **short quotes (≤ 2 sentences)** from the context, with citations (case name + chunk metadata).\n"
                    "4. Verify silently that every factual claim has support in the context; do not output a separate self-check section. If something lacks support, remove or qualify it.\n"
                    "5. If the context is **insufficient**, respond: 'I'm sorry — I don't know based on the provided documents.'\n"
                    "6. At the end, list the **sources used** (case name + chunk metadata).\n\n"
                    "Finally, add a single line 'Sources: N.txt, …' by extracting file headers like [file: N.txt] present in the context."
                )
                prompt = f"{system_prefix}\n\nQUESTION: {user_q}\n\nCONTEXT:\n{context}"
            
            # Generate streaming response
            answer = ""
            try:
                def _stream_gen():
                    nonlocal answer
                    # Add user message to chat history
                    st.session_state.chat_history.add_user_message(user_q)
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
            except Exception as e:
                answer = f"[Error] {e}"

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
        dcols = st.columns([1, 5])
        with dcols[0]:
            if st.button("Clear logs"):
                st.session_state.debug_logs = []
                st.session_state["debug_string"] = ""
        logs_text = "\n".join(st.session_state.debug_logs) if st.session_state.get("debug_logs") else ""
        st.text_area("Debug logs", value=logs_text, height=320, label_visibility="collapsed")


if __name__ == "__main__":
    main()


