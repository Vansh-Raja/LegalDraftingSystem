"""
CLI chat interface (deprecated in favor of the Streamlit app `app.py`).

This module retains a console-based experience mainly for testing or headless
environments. Most users should run the Streamlit UI for a better workflow:

    streamlit run app.py

The logic mirrors the retrieval/filtration/context assembly pipeline used in
the app, but without the rich UI and session management provided by Streamlit.
"""

from typing import List
from pathlib import Path
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
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


def chat_with_memory() -> None:
    """
    Start an interactive CLI chat session with simple in-memory history.

    Note: This is considered near-deprecated. Prefer the Streamlit app.
    """
    # In-memory chat histories (per session)
    store: dict[str, InMemoryChatMessageHistory] = {}
    session_id = "default"

    def get_history(sess_id: str):
        hist = store.get(sess_id)
        if hist is None:
            hist = InMemoryChatMessageHistory()
            store[sess_id] = hist
        return hist

    load_dotenv()
    # Runtime model selection (simple prompt; Streamlit offers a nicer UI)
    print("Select model: [1] Ollama qwen3:latest (default), [2] OpenAI gpt-5-nano-2025-08-07 > ", end="")
    _choice = input().strip()
    if _choice == "2" or _choice.lower() == "openai":
        api_key = os.getenv("OPENAI_KEY")
        if not api_key:
            print("OPENAI_KEY not set in environment. Falling back to Ollama qwen3:latest.")
            llm = ChatOllama(model="qwen3:latest", temperature=0, streaming=True, num_ctx=40000)
        else:
            llm = ChatOpenAI(model="gpt-5-nano-2025-08-07", temperature=0, streaming=True, api_key=api_key)
    else:
        llm = ChatOllama(model="qwen3:latest", temperature=0, streaming=True, num_ctx=40000)
    vs = get_vectorstore()

    # Prompt template: system instructions + history + user question
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are a **legal expert helper** in a RAG system. You will be given a user question and context assembled from retrieved chunks/full cases. Use **only** the provided context; do **not** use external knowledge or invent facts. After you generate your answer, **self-verify**: recheck that the user's question is most accurately matched by the selected context; if not, qualify your answer or state that information is insufficient.\n\n"
                "The context always begins with a `Case Metadata Overview`. Read that section first, confirm which cases are in scope, and use it as an anchor before examining the detailed excerpts that follow.\n\n"
                "Guidelines:\n\n"
                "1. If the user’s question names a **specific case** (by parties, court, date, or case number), **prioritize** that case and **limit** use of other cases unless strictly needed.\n"
                "2. If the question is about a **legal topic, statute, or section**, you may **synthesize** across multiple relevant documents.\n"
                "3. Your answer should be **precise, well-reasoned, and evidence-based**. You may include **short quotes (≤ 2 sentences)** from the context, with citations (case name + chunk identifier or metadata).\n"
                "4. After drafting the answer, **self-check**: verify that every factual claim in your answer has supporting text in the context. If any claim is not verified, **remove or qualify** it.\n"
                "5. If the context is **insufficient** to give a confident answer, respond:\n"
                "   > “I’m sorry — I don’t know based on the provided documents.”\n"
                "6. At the end, list the **sources used** (case name + chunk metadata) that you relied on in answering.\n\n"
                "**QUESTION:** {question}\n"
                "**CONTEXT:**\n"
                "{context}"
            ),
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    # Base LLM chain (answer string)
    base_chain = (
        RunnablePassthrough()
        | prompt
        | llm
        | StrOutputParser()
    )

    # Wrap with message history
    chain_with_history = RunnableWithMessageHistory(
        base_chain,
        get_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    def make_chain(statutes: List[str] | None = None):
        """
        Create a callable that performs retrieval, filtration, and answering
        for a single question, optionally constrained by statute filters.
        """
        # Try strict court filter first; if no hits, fallback without court filter
        retriever = build_retriever(vs, statute_filters=statutes, court_name="Supreme Court of India", k=6)

        def invoke_with_context(user_q: str):
            """Retrieve docs, build a context, and stream an answer."""
            docs = retriever.invoke(user_q)
            if not docs:
                # Fallback: relax court filter
                retriever_relaxed = build_retriever(vs, statute_filters=statutes, court_name=None, k=6)
                docs = retriever_relaxed.invoke(user_q)
            # Debug: show retrieval diagnostics
            print("\n[DEBUG] Retrieved docs:")
            for i, d in enumerate(docs[:6], 1):
                fs = d.metadata.get("file_stem")
                preview = (d.page_content or "").strip().replace("\n", " ")[:160]
                print(f"  {i}. file_stem={fs} ... {preview}")
            # Filtration retriever and context assembly
            plan = filtration_retriever(user_q, docs)
            plan = apply_named_case_guard(plan, user_q, docs)
            context, _, dbg = assemble_context_from_plan(plan, user_q, vs, txt_dir="processed_data/txt_data", budget_tokens=plan.context_budget_tokens, initial_docs=docs)
            try:
                import json as _json
                print(f"[DEBUG][Filtration] plan JSON: {_json.dumps(plan.model_dump(), indent=2)}")
            except Exception:
                print(f"[DEBUG][Filtration] plan: selected_full_docs={plan.selected_full_docs}, selected_chunks={len(plan.selected_chunks)}, budget={plan.context_budget_tokens}")
            # print reasoning if present
            if getattr(plan, "overall_reasoning", None):
                print(f"[DEBUG][Filtration] overall_reasoning: {plan.overall_reasoning}")
            if getattr(plan, "reasoning_full_docs", None):
                for rd in plan.reasoning_full_docs:
                    try:
                        print(f"[DEBUG][Filtration] full_doc_reason: file_stem={rd.file_stem} reason={rd.reason}")
                    except Exception:
                        pass
            if getattr(plan, "reasoning_chunks", None):
                for rc in plan.reasoning_chunks:
                    try:
                        print(f"[DEBUG][Filtration] chunk_reason: file_stem={rc.file_stem} idx={rc.chunk_index} reason={rc.reason}")
                    except Exception:
                        pass
            print(f"[DEBUG][Filtration] assembler: est_tokens~{dbg.get('est_tokens')}, spans={dbg.get('spans')[:5]}")
            included_dbg = dbg.get("included_full_docs") or []
            if included_dbg:
                print("[DEBUG][Filtration] assembler_full_docs:")
                for info in included_dbg:
                    file_name = info.get("file")
                    mode = info.get("mode")
                    included_flag = info.get("included")
                    chars_used = info.get("chars_used")
                    avail = info.get("available_chars")
                    print(f"  - {file_name}: mode={mode}, included={included_flag}, chars_used={chars_used}, available_chars={avail}")
            metadata_dbg = dbg.get("metadata_cases") or []
            if metadata_dbg:
                print("[DEBUG][Filtration] metadata_cases:")
                for case_info in metadata_dbg:
                    print(
                        f"  - {case_info.get('file_stem')}: "
                        f"case_number={case_info.get('case_number')}, "
                        f"court={case_info.get('court_name')}, "
                        f"date={case_info.get('date_of_judgment')}, "
                        f"has_summary={case_info.get('has_summary')}"
                    )
            if not context:
                print("[DEBUG][Filtration] Empty context; using dominant-case fallback.")
            else:
                preview = (context[:800] + "...") if len(context) > 800 else context
                print(f"[DEBUG][Context Preview] {preview}")

            # Full-case fallback gate: if context is empty, window the dominant case
            from collections import Counter
            stems = [d.metadata.get("file_stem") for d in docs if d.metadata.get("file_stem")]
            # context may be already set by planner; only fallback if empty
            if stems:
                # Choose dominant case: prefer match against query tokens; else simple majority
                dominant = None
                ratio = 0.0
                try:
                    qnorm = user_q.lower()
                    tokens = [t for t in qnorm.replace(" v. ", " vs ").split() if t.isalpha() and len(t) > 2]
                    party_hints = set(tokens)
                    best_fs = None
                    best_hits = -1
                    for d in docs[:6]:
                        txt = (d.page_content or "").lower()
                        hits = sum(1 for t in party_hints if t in txt)
                        if hits > best_hits:
                            best_hits = hits
                            best_fs = d.metadata.get("file_stem")
                    if best_fs and best_hits > 0:
                        dominant = best_fs
                        print(f"[DEBUG] Dominant by title-match: file_stem={dominant}, hits={best_hits}")
                except Exception as e:
                    print(f"[DEBUG] Title-match selection error: {e}")

                if dominant is None:
                    cnt_map = Counter(stems)
                    dominant, cnt = cnt_map.most_common(1)[0]
                    ratio = cnt / max(1, len(docs))
                    print(f"[DEBUG] Dominant by majority: file_stem={dominant}, ratio={ratio:.2f}")
                else:
                    ratio = 1.0  # force full-case when explicit title match

                if not context and ratio >= 0.5:
                    # Build focused context from raw full file within a safe budget
                    full_path = Path("processed_data/txt_data") / f"{dominant}.txt"
                    try:
                        if full_path.exists():
                            raw_text = full_path.read_text(encoding="utf-8")
                            budget = 160000  # ~40k tokens; align with expanded num_ctx
                            raw_lc = raw_text.lower()
                            qtokens = [t for t in (user_q.lower().replace(" v. ", " vs "))
                                       .split() if t.isalpha() and len(t) > 2]
                            # find match positions for query tokens
                            positions = []
                            for t in set(qtokens):
                                start = 0
                                hits = 0
                                while True:
                                    pos = raw_lc.find(t, start)
                                    if pos == -1 or hits >= 5:  # cap matches per token
                                        break
                                    positions.append(pos)
                                    start = pos + max(1, len(t))
                                    hits += 1
                            positions = sorted(set(positions))
                            windows = []
                            win_half = max(2000, budget // 8)  # 2-3k around each hit
                            for p in positions:
                                s = max(0, p - win_half)
                                e = min(len(raw_text), p + win_half)
                                windows.append((s, e))
                            # merge overlapping windows
                            merged = []
                            for s, e in sorted(windows):
                                if not merged or s > merged[-1][1] + 50:
                                    merged.append([s, e])
                                else:
                                    merged[-1][1] = max(merged[-1][1], e)
                            # if no hits, use head+tail
                            if not merged:
                                head = raw_text[: budget // 2]
                                tail = raw_text[-budget // 2 :]
                                used = head + "\n\n...\n\n" + tail
                                selected_spans = [(0, len(head)), (len(raw_text) - len(tail), len(raw_text))]
                            else:
                                parts = []
                                total = 0
                                selected_spans = []
                                for s, e in merged:
                                    if total + (e - s) > budget:
                                        if budget - total <= 0:
                                            break
                                        e = s + (budget - total)
                                    parts.append(raw_text[s:e])
                                    selected_spans.append((s, e))
                                    total += (e - s)
                                used = "\n\n...\n\n".join(parts)
                            context = used
                            est_tokens = int(len(context) / 4)
                            print(f"[DEBUG] Full-case RAW file: {full_path.name}, length={len(raw_text)}, used_chars={len(context)}, est_tokens~{est_tokens}")
                            print(f"[DEBUG] Context windows: {selected_spans[:5]}{' (truncated)' if len(selected_spans) > 5 else ''}")
                        else:
                            print(f"[DEBUG] Full-case RAW file missing: {full_path}")
                            context = None
                    except Exception as e:
                        print(f"[DEBUG] Full-case RAW read error: {e}")
                        context = None
            if not context:
                context = "\n\n".join(d.page_content for d in docs)
                est_tokens = int(len(context) / 4)
                print(f"[DEBUG] Context length: {len(context)} chars, chunks: {len(docs)}, est_tokens~{est_tokens}")
            # Stream the response using the chain with message history
            stream = chain_with_history.stream(
                {"question": user_q, "context": context},
                config={"configurable": {"session_id": session_id}},
            )
            answer_parts = []
            for chunk in stream:
                if isinstance(chunk, str):
                    print(chunk, end="", flush=True)
                    answer_parts.append(chunk)
            print()
            answer = "".join(answer_parts)
            print("[DEBUG] Answer length:", len(answer) if isinstance(answer, str) else "n/a")
            return answer, docs

        return invoke_with_context

    print("[Deprecated] CLI Legal Drafting Chatbot with memory is live. Type exit to quit.")
    while True:
        user_q = input("\nYou: ").strip()
        if user_q.lower() in ("exit", "quit"):
            break

        # Optional: simple statute extraction placeholder (can be expanded)
        statutes = None
        chain = make_chain(statutes)
        answer, docs = chain(user_q)

        print("\nBot:", answer)
        print("\nSources:")
        for d in docs:
            fs = d.metadata.get("file_stem")
            print(f"- Case: {fs}, metadata: {d.metadata}")

        full = maybe_full_case(docs)
        if full:
            print("[Hint] Many chunks from one case — full judgment available:", full)


if __name__ == "__main__":
    chat_with_memory()
