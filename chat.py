from typing import List
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_ollama import ChatOllama

from rag import (
    get_vectorstore,
    build_retriever,
    maybe_full_case,
)


def chat_with_memory() -> None:
    # In-memory chat histories (per session)
    store: dict[str, InMemoryChatMessageHistory] = {}
    session_id = "default"

    def get_history(sess_id: str):
        hist = store.get(sess_id)
        if hist is None:
            hist = InMemoryChatMessageHistory()
            store[sess_id] = hist
        return hist

    llm = ChatOllama(model="qwen3:latest", temperature=0, streaming=True)
    vs = get_vectorstore()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are a **legal RAG assistant**. Use **only** the retrieved documents provided with the user question. You must **not** rely on external knowledge or invent facts.\n\n"
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
        # Start without strict court filter to ensure we retrieve
        retriever = build_retriever(vs, statute_filters=statutes, court_name=None, k=6)

        def invoke_with_context(user_q: str):
            docs = retriever.invoke(user_q)
            # Debug: show retrieval diagnostics
            print("\n[DEBUG] Retrieved docs:")
            for i, d in enumerate(docs[:6], 1):
                fs = d.metadata.get("file_stem")
                preview = (d.page_content or "").strip().replace("\n", " ")[:160]
                print(f"  {i}. file_stem={fs} ... {preview}")
            context = "\n\n".join(d.page_content for d in docs)
            print(f"[DEBUG] Context length: {len(context)} chars, chunks: {len(docs)}")
            # Feed question + context to the LLM; history is handled by the wrapper
            # Stream the response
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

    print("Legal Drafting Chatbot with memory is live. Type exit to quit.")
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


