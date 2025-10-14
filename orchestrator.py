"""
Query processing and planning module.
Classifies user queries and creates execution plans for the RAG system.
"""

import os
import json
import re
from typing import List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI


class QueryPlan(BaseModel):
    """
    Data model for query execution plans.
    
    Attributes:
        type (str): Query type - "followup", "new", or "general_law"
        rewrite (str): Rewritten query for better retrieval
        keep_context (bool): Whether to reuse previous context
        bridging_strategy (str): How to bridge with previous query - "none", "adjacent", "statute_refill", "same_case_full"
        target_stems (List[str]): Specific case file stems to target
        statutes (List[str]): Statute filters to apply
        retrieval_k (Optional[int]): Number of documents to retrieve
        reason (Optional[str]): Explanation of the plan
    """
    type: str  # "followup" | "new" | "general_law"
    rewrite: str
    keep_context: bool = False
    bridging_strategy: str = "none"  # none | adjacent | statute_refill | same_case_full
    target_stems: List[str] = []
    statutes: List[str] = []
    retrieval_k: Optional[int] = None
    reason: Optional[str] = None


# Common prefixes that indicate general legal knowledge questions
_GENERIC_PREFIXES = (
    "what is",
    "what are",
    "how is",
    "how are",
    "explain",
    "outline",
    "give an overview",
)


def _detect_signals(text: str) -> dict:
    """
    Analyze text to detect legal query patterns and signals.
    
    Args:
        text (str): User query text to analyze
        
    Returns:
        dict: Dictionary with detected signals (case markers, statute markers, generic patterns)
    """
    t = (text or "").lower()
    
    # Detect case-specific markers (party names, case numbers, court references)
    has_case_marker = bool(re.search(r"\s(v\.|vs\.|vs)\s|insc|slp|criminal\s+appeal|writ\s+petition|scc", t))
    
    # Detect statute/legal provision markers
    has_statute_marker = bool(re.search(r"\bipc\b|\bcrpc\b|evidence act|article\s+\d+|section\s+\d+", t))
    
    # Check if query starts with generic question patterns
    generic_like = t.startswith(_GENERIC_PREFIXES)
    
    # Additional light-weight specificity checks
    token_count = len([w for w in re.split(r"\s+", t) if w])
    has_digit = any(ch.isdigit() for ch in t)
    
    return {
        "has_case_marker": has_case_marker,
        "has_statute_marker": has_statute_marker,
        "generic": generic_like,
        "token_count": token_count,
        "has_digit": has_digit,
    }


def _heuristic_plan(user_q: str) -> "QueryPlan":
    """
    Create a basic query plan using simple heuristics (fallback when LLM fails).
    
    Args:
        user_q (str): User query text
        
    Returns:
        QueryPlan: Basic plan based on detected patterns
    """
    sig = _detect_signals(user_q)
    
    # Only the very broadest queries should be routed to general_law:
    # - starts with a generic prefix
    # - AND has no case/statute markers
    # - AND is short and non-specific (few tokens, no digits)
    if (
        sig["generic"]
        and not sig["has_case_marker"]
        and not sig["has_statute_marker"]
        and sig.get("token_count", 0) <= 8
        and not sig.get("has_digit", False)
    ):
        # Broad/general → suggest larger K
        return QueryPlan(
            type="general_law",
            rewrite=user_q,
            keep_context=False,
            bridging_strategy="none",
            target_stems=[],
            statutes=[],
            retrieval_k=16,
            reason="Heuristic: generic GK without case/statute markers (broader K)",
        )
    
    # Default to new query type
    # Otherwise prefer retrieval path (new/followup) even if somewhat generic
    return QueryPlan(type="new", rewrite=user_q, keep_context=False, bridging_strategy="none", retrieval_k=6)


def process_query(
    user_q: str,
    last_question_rewrite: Optional[str] = None,
    last_stems: Optional[List[str]] = None,
    last_filters: Optional[dict] = None,
    last_context_snippet: Optional[str] = None,
    summary: Optional[str] = None,
) -> QueryPlan:
    """
    Classify and create execution plan for user query using LLM.
    
    This is the main query processing function that:
    1. Analyzes the user query in context of previous conversation
    2. Determines if it's a follow-up, new query, or general law question
    3. Creates a plan for how to handle retrieval and context assembly
    
    Args:
        user_q (str): Current user query
        last_question_rewrite (Optional[str]): Previous query rewrite
        last_stems (Optional[List[str]]): Previous case file stems
        last_filters (Optional[dict]): Previous filters applied
        last_context_snippet (Optional[str]): Previous context snippet
        summary (Optional[str]): Conversation summary
        
    Returns:
        QueryPlan: Execution plan for the query
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_KEY")
    client = OpenAI(api_key=api_key)

    # System prompt for query classification and planning
    system_msg = (
        "You are a query-processor for a legal RAG assistant. Return STRICT JSON with fields: "
        "{type, rewrite, keep_context, bridging_strategy, target_stems, statutes, retrieval_k, reason}.\n"
        "Rules: type is one of followup | new | general_law.\n"
        "ROUTING: Only route to general_law if the question is extremely broad and non-specific (short, no case/statute markers).\n"
        "If there is ANY specificity (numbers, dates, party names, sections, court names, or concrete scenario), choose new or followup for retrieval.\n"
        "If followup, decide keep_context (true if the current context already contains the case/material needed). "
        "If the user appears to ask for similar cases or statutes beyond current context, set bridging_strategy=statute_refill or adjacent; if they want the same case full, set same_case_full. "
        "Always produce a helpful standalone rewrite for retrieval; expand acronyms and include entities (parties, court, date, case numbers) if known.\n\n"
        "TOP-K SELECTION: When appropriate, set retrieval_k as follows (use judgment; integers only):\n"
        "- General/very broad questions (no specific case/statute): retrieval_k ~ 12-20\n"
        "- Typical topic queries: retrieval_k ~ 8-12\n"
        "- Case-specific or tightly-focused follow-ups: retrieval_k ~ 4-6\n"
        "If uncertain, pick 8.\n\n"
        "GENERAL-LAW CLASSIFICATION GUIDELINE: If the question is high-level (e.g., 'What laws apply to murder cases?', 'What is res judicata?', 'How is bail decided?'), and it does not reference a specific case name, number, court, date, or document already in context, classify it as general_law. In that case, keep_context=false and bridging_strategy='none'.\n\n"
        "EXAMPLES (label -> JSON):\n"
        "Q: 'What laws are used in a murder case?' -> {\"type\": \"general_law\", \"rewrite\": \"What statutes and charges typically apply to homicide/murder cases in India (IPC sections)?\", \"keep_context\": false, \"bridging_strategy\": \"none\", \"target_stems\": [], \"statutes\": [\"IPC s.302\", \"IPC s.304\"], \"retrieval_k\": 16, \"reason\": \"General law query without specific case\"}\n"
        "Q: 'In John Kennedy vs State of Tamil Nadu, what was the final order?' -> {\"type\": \"new\", \"rewrite\": \"Final order in A. John Kennedy vs State of Tamil Nadu, 2025 INSC 443 (Supreme Court of India)\", \"keep_context\": false, \"bridging_strategy\": \"same_case_full\", \"target_stems\": [\"1\"], \"statutes\": [], \"retrieval_k\": 6, \"reason\": \"Specific case named\"}\n"
        "Q: 'Also list similar cases where IPC 302 was applied' (after a case turn) -> {\"type\": \"followup\", \"rewrite\": \"Supreme Court decisions applying IPC Section 302 similar to <last case>\", \"keep_context\": true, \"bridging_strategy\": \"statute_refill\", \"target_stems\": [], \"statutes\": [\"IPC s.302\"], \"retrieval_k\": 8, \"reason\": \"Follow-up requesting similar cases by statute\"}"
    )

    # Prepare payload with context information
    payload = {
        "user_question": user_q,
        "last_question_rewrite": last_question_rewrite or "",
        "last_stems": last_stems or [],
        "last_filters": last_filters or {},
        "last_context_snippet": (last_context_snippet or "")[:4000],
        "rolling_summary": (summary or "")[:2000],
        "schema": {
            "type": "followup|new|general_law",
            "rewrite": "string",
            "keep_context": True,
            "bridging_strategy": "none|adjacent|statute_refill|same_case_full",
            "target_stems": ["7", "21"],
            "statutes": ["IPC s.302"],
            "retrieval_k": 12,
            "reason": "short explanation",
        },
    }

    # Try OpenAI's structured output API first
    try:
        parsed_plan = client.responses.parse(
            model="gpt-5-nano-2025-08-07",
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": system_msg}]},
                {"role": "user", "content": [{"type": "input_text", "text": json.dumps(payload)}]},
            ],
            text_format=QueryPlan,
        )
        
        # Handle different SDK response formats
        try:
            if isinstance(parsed_plan, QueryPlan):
                print("[DEBUG][QP][parse_api] returned QueryPlan instance")
                return parsed_plan
            # Try common attributes for parsed output
            maybe = getattr(parsed_plan, "output_parsed", None) or getattr(parsed_plan, "parsed", None)
            if isinstance(maybe, QueryPlan):
                print("[DEBUG][QP][parse_api] returned output_parsed QueryPlan")
                return maybe
        except Exception as ix:
            print(f"[DEBUG][QP][parse_api_introspection_error] {ix}")
        
        # Debug: log unexpected response format
        try:
            print(f"[DEBUG][QP][parse_api_unexpected_type] {type(parsed_plan)}")
            for attr in ("output", "output_text", "output_parsed", "response", "status_code", "model_dump_json"):
                val = getattr(parsed_plan, attr, None)
                if callable(val):
                    try:
                        val = val()
                    except Exception:
                        pass
                if val is not None:
                    head = str(val)
                    if isinstance(head, str):
                        head = head[:400]
                    print(f"[DEBUG][QP][parse_api_unexpected_attr] {attr}={head}")
        except Exception as dx:
            print(f"[DEBUG][QP][parse_api_dump_error] {dx}")
        print("[DEBUG][QP][parse_api] unexpected return; skipping JSON mode fallback per request")
    except Exception as e:
        print(f"[DEBUG][QP][parse_api_error] {e}")

    # Fallback to heuristic planning if LLM fails
    plan = _heuristic_plan(user_q)
    print(f"[DEBUG][QP][heuristic_fallback] type={plan.type} rewrite={plan.rewrite}")
    return plan


