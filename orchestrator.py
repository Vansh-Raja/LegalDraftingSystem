"""
Query processing and planning module.
Classifies user queries and creates execution plans for the RAG system.
"""

import json
import re
import os
from typing import List, Optional
from pydantic import BaseModel
from time_utils import now_ist_stamp
from dotenv import load_dotenv
from openai import OpenAI


def _log_debug(msg: str) -> None:
    print(f"{now_ist_stamp()} {msg}")


class QueryPlan(BaseModel):
    """
    Data model for query execution plans.
    
    Attributes:
        type (str): Query type - "followup", "new", "general_law", or "general_chat"
        rewrite (str): Rewritten query for better retrieval
        keep_context (bool): Whether to reuse previous context
        bridging_strategy (str): How to bridge with previous query - "none", "adjacent", "statute_refill", "same_case_full"
        target_stems (List[str]): Specific case file stems to target
        statutes (List[str]): Statute filters to apply
        retrieval_k (Optional[int]): Number of documents to retrieve
        reason (Optional[str]): Explanation of the plan
        case_probe (Optional[str]): Secondary retrieval query for general-law advisory augmentation
    """
    type: str  # "followup" | "new" | "general_law"
    rewrite: str
    keep_context: bool = False
    bridging_strategy: str = "none"  # none | adjacent | statute_refill | same_case_full
    target_stems: List[str] = []
    statutes: List[str] = []
    retrieval_k: Optional[int] = None
    min_full_docs: Optional[int] = None
    case_probe: Optional[str] = None
    reason: Optional[str] = None
    breadth: str = "unknown"  # specific | narrow | broad | unknown


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
        "raw": t,
    }


_BROAD_KEYWORDS = (
    "cases",
    "judgments",
    "summaries",
    "summary",
    "overview",
    "trend",
    "trends",
    "statistics",
    "analysis",
    "compare",
    "comparison",
    "list",
    "catalog",
    "catalogue",
    "recent",
    "latest",
    "across",
    "multiple",
    "various",
    "collection",
    "survey",
    "digest",
    "roundup",
)


def _classify_breadth(user_q: str, sig: dict, plan: Optional["QueryPlan"] = None) -> str:
    """
    Classify query breadth (specific, narrow, broad) using heuristics and plan hints.
    """
    if plan and getattr(plan, "breadth", None) and plan.breadth not in {"", "unknown"}:
        return plan.breadth
    if plan and plan.type == "general_law":
        return "broad"
    if plan and plan.type == "general_chat":
        return "chat"
    if plan and plan.target_stems:
        return "specific"

    text_lower = sig.get("raw") or (user_q or "").lower()

    if sig.get("has_case_marker"):
        return "specific"
    if " vs " in text_lower or " v. " in text_lower:
        return "specific"

    broad_keyword_hit = any(word in text_lower for word in _BROAD_KEYWORDS)

    if sig.get("has_statute_marker") and not broad_keyword_hit:
        return "narrow"
    if broad_keyword_hit:
        return "broad"
    if any(year in text_lower for year in ("2022", "2023", "2024", "2025", "2026")) and "case" in text_lower and " vs " not in text_lower:
        return "broad"
    if sig.get("token_count", 0) >= 20 and not sig.get("has_statute_marker"):
        return "broad"
    return "narrow"


def _postprocess_plan(plan: "QueryPlan", user_q: str, sig: Optional[dict] = None) -> "QueryPlan":
    if plan is None:
        return plan
    signals = sig or _detect_signals(user_q)
    breadth = _classify_breadth(user_q, signals, plan)

    if plan.type == "general_chat":
        update_payload = {
            "breadth": "chat",
            "min_full_docs": 0,
            "retrieval_k": 0,
            "case_probe": plan.case_probe or "",
            "statutes": plan.statutes or [],
            "target_stems": plan.target_stems or [],
            "keep_context": False,
            "bridging_strategy": "none",
        }
        if plan.rewrite is None:
            update_payload["rewrite"] = ""
        return plan.copy(update=update_payload)

    min_docs = plan.min_full_docs
    if min_docs is None:
        min_docs = 4 if plan.type == "general_law" or breadth == "broad" else (3 if breadth == "narrow" else 2)
    retr_k = plan.retrieval_k
    if retr_k is None:
        if plan.type == "general_law":
            retr_k = 16
        elif breadth == "broad":
            retr_k = 12
        else:
            retr_k = 8 if breadth == "narrow" else 6
    try:
        retr_k = max(3, min(20, int(retr_k)))
    except Exception:
        retr_k = 8
    return plan.copy(update={"breadth": breadth, "min_full_docs": min_docs, "retrieval_k": retr_k})


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
        plan = QueryPlan(
            type="general_law",
            rewrite=user_q,
            keep_context=False,
            bridging_strategy="none",
            target_stems=[],
            statutes=[],
            retrieval_k=12,
            min_full_docs=3,
            case_probe=user_q,
            reason="Heuristic: generic guidance query; pair advisory reply with precedent retrieval",
        )
        return _postprocess_plan(plan, user_q, sig)
    
    # Default to new query type
    # Otherwise prefer retrieval path (new/followup) even if somewhat generic
    plan = QueryPlan(type="new", rewrite=user_q, keep_context=False, bridging_strategy="none", retrieval_k=6)
    return _postprocess_plan(plan, user_q, sig)


def process_query(
    user_q: str,
    last_question_rewrite: Optional[str] = None,
    last_stems: Optional[List[str]] = None,
    last_filters: Optional[dict] = None,
    last_context_snippet: Optional[str] = None,
    summary: Optional[str] = None,
    manual_mode: bool = False,
    llm=None,
    model_name: Optional[str] = None,
    provider_name: Optional[str] = None,
) -> QueryPlan:
    """
    Classify and create execution plan for user query using the selected LLM.
    Falls back to heuristic planning if the provided LLM fails or is absent.
    """
    if llm is None:
        _log_debug("[DEBUG][QP] No LLM provided; using heuristic planner")
        plan = _heuristic_plan(user_q)
        _log_debug(f"[DEBUG][QP][heuristic_only] type={plan.type} rewrite={plan.rewrite}")
        return plan

    from models import get_prompt
    system_msg = get_prompt("planner_manual" if manual_mode else "planner_auto")
    mode_label = "manual" if manual_mode else "auto"
    _log_debug(f"[DEBUG][QP][mode] using {mode_label} planner prompt via model={model_name} provider={provider_name}")

    # System prompts
    system_msg_auto = (
        "You are a query-processor for a legal RAG assistant. Return STRICT JSON with fields: "
        "{type, rewrite, case_probe, keep_context, bridging_strategy, target_stems, statutes, retrieval_k, min_full_docs, breadth, reason}.\n"
        "Rules: type is one of followup | new | general_law | general_chat.\n"
        "ROUTING: Use general_law when the user is asking for concepts, rights, procedures, or personal/hypothetical guidance without pointing to a specific docketed case or prior answer.\n"
        "CASE RETRIEVAL RULE: Any request to list, find, summarise, compare, or check for cases/judgments—even if phrased broadly or without citations—must be routed to new (or followup if it clearly references earlier results).\n"
        "If the question names a particular case, judgment, docket, citation, court/date, or explicitly references earlier assistant context, choose new or followup for retrieval instead.\n"
        "If followup, decide keep_context (true if the current context already contains the case/material needed). "
        "If the user appears to ask for similar cases or statutes beyond current context, set bridging_strategy=statute_refill or adjacent; if they want the same case full, set same_case_full. "
        "Always produce a helpful standalone rewrite for retrieval; expand acronyms and include entities (parties, court, date, case numbers) if known. "
        "Keep the rewrite concise and keyword-rich (≤ 20 tokens). "
        "For general_law queries, lightly tidy the user's phrasing but preserve their perspective and scenario (do NOT invent new facts or over-narrow the question). "
        "For new or followup queries, continue to produce tight, keyword-rich rewrites with explicit parties, courts, dates, or statutes when available. "
        "When type=general_law, ALSO populate case_probe with a focused retrieval query for precedent search (include statute numbers, offence names, timeframes, geography if available). "
        "Set case_probe to an empty string for new, followup, or general_chat plans.\n\n"
        "TOP-K SELECTION: When appropriate, set retrieval_k as follows (use judgment; integers only):\n"
        "MIN FULL DOCS: Suggest min_full_docs (integer) ~ proportional to retrieval_k and breadth of query.\n"
        "- Broad/overview queries: min_full_docs ~ 3-6 (at least 2).\n"
        "- Case-specific queries: min_full_docs ~ 2-3.\n"
        "Except for general_chat, never return less than 2.\n\n"
        "BREADTH: Set breadth to \"broad\" (survey/overview across many cases), \"narrow\" (focused topic/statute requiring a handful of cases), or \"specific\" (single case or highly targeted follow-up).\n\n"
        "- General/very broad questions (no specific case/statute): retrieval_k ~ 12-20\n"
        "- Typical topic queries: retrieval_k ~ 8-12\n"
        "- Case-specific or tightly-focused follow-ups: retrieval_k ~ 4-6\n"
        "If uncertain, pick 8.\n\n"
        "GENERAL-LAW CLASSIFICATION GUIDELINE: If the question is high-level (e.g., 'What laws apply to murder cases?', 'What is res judicata?', 'How is bail decided?'), and it does not reference a specific case name, number, court, date, or document already in context, classify it as general_law. "
        "Mentioning statutes or offence sections alone does not force new; keep it under general_law if no particular case is identified. "
        "First-person or conversational hypotheticals without explicit case identifiers are still general_law. "
        "However, if the user asks to list, summarise, compare, or check for cases/judgments (even without naming them) by topic, statute, timeframe, geography, or parties (e.g., 'any rental cases from 2025'), classify it as new because retrieval of case documents is required. "
        "In that case, keep_context=false and bridging_strategy='none'.\n\n"
        "GENERAL-CHAT CLASSIFICATION GUIDELINE: If the user is only greeting you, introducing themselves, expressing thanks, or asking personal/identity questions unrelated to legal matters (e.g., 'hi, I am Vansh', 'remember my name?'), classify it as general_chat. "
        "For general_chat, set rewrite=\"\", case_probe=\"\", keep_context=false, retrieval_k=0, min_full_docs=0, and breadth='chat'. Provide a short reason noting it is non-legal conversation.\n\n"
        "EXPANDED EXAMPLES (label -> JSON):\n"
        "Q: 'What is anticipatory bail and how can I apply for it?' -> {\"type\": \"general_law\", \"rewrite\": \"Anticipatory bail meaning and application steps in India\", \"case_probe\": \"Supreme Court anticipatory bail jurisprudence CrPC 438 arrest\", \"keep_context\": false, \"bridging_strategy\": \"none\", \"target_stems\": [], \"statutes\": [\"CrPC s.438\"], \"retrieval_k\": 16, \"min_full_docs\": 4, \"breadth\": \"broad\", \"reason\": \"Refined legal concept inquiry; no case reference\"}\n"
        "Q: 'Can someone get bail if charged under Section 302 of the IPC?' -> {\"type\": \"general_law\", \"rewrite\": \"Bail eligibility when accused under IPC Section 302\", \"case_probe\": \"Supreme Court murder IPC 302 bail precedents\", \"keep_context\": false, \"bridging_strategy\": \"none\", \"target_stems\": [], \"statutes\": [\"IPC s.302\"], \"retrieval_k\": 16, \"min_full_docs\": 4, \"breadth\": \"broad\", \"reason\": \"Statutory question about criminal law\"}\n"
        "Q: 'I was just detained by the police — what should I do right now?' -> {\"type\": \"general_law\", \"rewrite\": \"Immediate legal steps when detained by police in India\", \"case_probe\": \"Supreme Court rights of arrested person CrPC 41 50 guidance\", \"keep_context\": false, \"bridging_strategy\": \"none\", \"target_stems\": [], \"statutes\": [], \"retrieval_k\": 16, \"min_full_docs\": 4, \"breadth\": \"broad\", \"reason\": \"Personal yet general procedural advice\"}\n"
        "Q: 'If my neighbour keeps harassing me online, what actions can I take?' -> {\"type\": \"general_law\", \"rewrite\": \"Legal remedies for ongoing online harassment by a neighbour\", \"case_probe\": \"Supreme Court cyber harassment remedies IT Act 2000 IPC 354D\", \"keep_context\": false, \"bridging_strategy\": \"none\", \"target_stems\": [], \"statutes\": [], \"retrieval_k\": 16, \"min_full_docs\": 4, \"breadth\": \"broad\", \"reason\": \"Seeks remedies without case references\"}\n"
        "Q: 'How can a company appeal against an order of the NCLT?' -> {\"type\": \"general_law\", \"rewrite\": \"Procedure for company appeals against NCLT orders\", \"case_probe\": \"Supreme Court NCLT appeal procedure IBC section 61 limitation\", \"keep_context\": false, \"bridging_strategy\": \"none\", \"target_stems\": [], \"statutes\": [\"IBC s.61\"], \"retrieval_k\": 16, \"min_full_docs\": 4, \"breadth\": \"broad\", \"reason\": \"General corporate procedure question\"}\n"
        "Q: 'What's the difference between a cognizable and a non-cognizable offence?' -> {\"type\": \"general_law\", \"rewrite\": \"Difference between cognizable and non-cognizable offences in India\", \"case_probe\": \"Supreme Court cognizable non cognizable offence distinction precedents\", \"keep_context\": false, \"bridging_strategy\": \"none\", \"target_stems\": [], \"statutes\": [], \"retrieval_k\": 12, \"min_full_docs\": 3, \"breadth\": \"broad\", \"reason\": \"Basic conceptual query\"}\n"
        "Q: 'Do I need a lawyer to register an FIR or can I go alone?' -> {\"type\": \"general_law\", \"rewrite\": \"Whether a lawyer is needed to file an FIR in India\", \"case_probe\": \"Supreme Court FIR registration rights without lawyer guidance\", \"keep_context\": false, \"bridging_strategy\": \"none\", \"target_stems\": [], \"statutes\": [], \"retrieval_k\": 12, \"min_full_docs\": 3, \"breadth\": \"broad\", \"reason\": \"Everyday rights question\"}\n"
        "Q: 'What is the punishment for bribery under Indian law?' -> {\"type\": \"general_law\", \"rewrite\": \"Punishment for bribery offences under Indian law\", \"case_probe\": \"Supreme Court Prevention of Corruption Act sentencing 2025\", \"keep_context\": false, \"bridging_strategy\": \"none\", \"target_stems\": [], \"statutes\": [\"PC Act\"], \"retrieval_k\": 12, \"min_full_docs\": 3, \"breadth\": \"broad\", \"reason\": \"Topic-based legal query\"}\n"
        "Q: 'My company is shutting down — how do I make sure employees get paid legally?' -> {\"type\": \"general_law\", \"rewrite\": \"Legal compliance for employee payouts during company shutdown\", \"case_probe\": \"Supreme Court retrenchment severance compliance labour law 2025\", \"keep_context\": false, \"bridging_strategy\": \"none\", \"target_stems\": [], \"statutes\": [], \"retrieval_k\": 16, \"min_full_docs\": 4, \"breadth\": \"broad\", \"reason\": \"Procedural compliance question\"}\n"
        "Q: 'If I accidentally sign a contract under pressure, is it still valid?' -> {\"type\": \"general_law\", \"rewrite\": \"Validity of contracts signed under pressure in India\", \"case_probe\": \"Supreme Court undue influence coercion contract validity precedents\", \"keep_context\": false, \"bridging_strategy\": \"none\", \"target_stems\": [], \"statutes\": [], \"retrieval_k\": 12, \"min_full_docs\": 3, \"breadth\": \"broad\", \"reason\": \"Hypothetical contract law scenario\"}\n"
        "Q: 'Hi, I am Vansh.' -> {\"type\": \"general_chat\", \"rewrite\": \"\", \"case_probe\": \"\", \"keep_context\": false, \"bridging_strategy\": \"none\", \"target_stems\": [], \"statutes\": [], \"retrieval_k\": 0, \"min_full_docs\": 0, \"breadth\": \"chat\", \"reason\": \"Greeting / self-introduction\"}\n"
        "Q: 'Thanks for your help!' -> {\"type\": \"general_chat\", \"rewrite\": \"\", \"case_probe\": \"\", \"keep_context\": false, \"bridging_strategy\": \"none\", \"target_stems\": [], \"statutes\": [], \"retrieval_k\": 0, \"min_full_docs\": 0, \"breadth\": \"chat\", \"reason\": \"Appreciation / no legal content\"}\n"
        "Q: 'Tell me a joke about lawyers.' -> {\"type\": \"general_chat\", \"rewrite\": \"\", \"case_probe\": \"\", \"keep_context\": false, \"bridging_strategy\": \"none\", \"target_stems\": [], \"statutes\": [], \"retrieval_k\": 0, \"min_full_docs\": 0, \"breadth\": \"chat\", \"reason\": \"Entertainment request\"}\n"
        "Q: 'What is my name?' -> {\"type\": \"general_chat\", \"rewrite\": \"\", \"case_probe\": \"\", \"keep_context\": false, \"bridging_strategy\": \"none\", \"target_stems\": [], \"statutes\": [], \"retrieval_k\": 0, \"min_full_docs\": 0, \"breadth\": \"chat\", \"reason\": \"Personal memory check\"}\n"
        "Q: 'What country is New York in?' -> {\"type\": \"general_chat\", \"rewrite\": \"\", \"case_probe\": \"\", \"keep_context\": false, \"bridging_strategy\": \"none\", \"target_stems\": [], \"statutes\": [], \"retrieval_k\": 0, \"min_full_docs\": 0, \"breadth\": \"chat\", \"reason\": \"General knowledge question\"}\n"
        "Q: 'Can you remind me of my name?' -> {\"type\": \"general_chat\", \"rewrite\": \"\", \"case_probe\": \"\", \"keep_context\": false, \"bridging_strategy\": \"none\", \"target_stems\": [], \"statutes\": [], \"retrieval_k\": 0, \"min_full_docs\": 0, \"breadth\": \"chat\", \"reason\": \"Personal memory check\"}\n"
        "Q: 'How are you doing today?' -> {\"type\": \"general_chat\", \"rewrite\": \"\", \"case_probe\": \"\", \"keep_context\": false, \"bridging_strategy\": \"none\", \"target_stems\": [], \"statutes\": [], \"retrieval_k\": 0, \"min_full_docs\": 0, \"breadth\": \"chat\", \"reason\": \"Small talk\"}\n"
        "Q: 'Do you remember me from yesterday?' -> {\"type\": \"general_chat\", \"rewrite\": \"\", \"case_probe\": \"\", \"keep_context\": false, \"bridging_strategy\": \"none\", \"target_stems\": [], \"statutes\": [], \"retrieval_k\": 0, \"min_full_docs\": 0, \"breadth\": \"chat\", \"reason\": \"Memory prompt\"}\n"
        "Q: 'What is the weather like in Delhi today?' -> {\"type\": \"general_chat\", \"rewrite\": \"\", \"case_probe\": \"\", \"keep_context\": false, \"bridging_strategy\": \"none\", \"target_stems\": [], \"statutes\": [], \"retrieval_k\": 0, \"min_full_docs\": 0, \"breadth\": \"chat\", \"reason\": \"Non-legal factual query\"}\n"
        "Q: 'Sing me a motivational quote.' -> {\"type\": \"general_chat\", \"rewrite\": \"\", \"case_probe\": \"\", \"keep_context\": false, \"bridging_strategy\": \"none\", \"target_stems\": [], \"statutes\": [], \"retrieval_k\": 0, \"min_full_docs\": 0, \"breadth\": \"chat\", \"reason\": \"Motivational request\"}\n"
        "Q: 'What time is it?' -> {\"type\": \"general_chat\", \"rewrite\": \"\", \"case_probe\": \"\", \"keep_context\": false, \"bridging_strategy\": \"none\", \"target_stems\": [], \"statutes\": [], \"retrieval_k\": 0, \"min_full_docs\": 0, \"breadth\": \"chat\", \"reason\": \"General utility question\"}\n"
        "Q: 'Summarize the Supreme Court judgment in Abdul Nassar vs State of Kerala (2025).' -> {\"type\": \"new\", \"rewrite\": \"Supreme Court judgment summary for Abdul Nassar vs State of Kerala decided 2025\", \"keep_context\": false, \"bridging_strategy\": \"same_case_full\", \"target_stems\": [], \"statutes\": [], \"retrieval_k\": 6, \"min_full_docs\": 2, \"breadth\": \"specific\", \"reason\": \"Explicit single-case summary\"}\n"
        "Q: 'Summarize all Supreme Court cases of bribery in 2025.' -> {\"type\": \"new\", \"rewrite\": \"Supreme Court bribery judgments from calendar year 2025\", \"keep_context\": false, \"bridging_strategy\": \"adjacent\", \"target_stems\": [], \"statutes\": [], \"retrieval_k\": 16, \"min_full_docs\": 4, \"breadth\": \"broad\", \"reason\": \"Broad list-style retrieval\"}\n"
        "Q: 'Summarize all cases that happened against the State of Maharashtra in March 2025.' -> {\"type\": \"new\", \"rewrite\": \"Cases against State of Maharashtra decided March 2025\", \"keep_context\": false, \"bridging_strategy\": \"adjacent\", \"target_stems\": [], \"statutes\": [], \"retrieval_k\": 16, \"min_full_docs\": 4, \"breadth\": \"broad\", \"reason\": \"Time and party constrained search\"}\n"
        "Q: 'There was a case about forest encroachments in Tamil Nadu — what did the Court say?' -> {\"type\": \"new\", \"rewrite\": \"Supreme Court decision on Tamil Nadu forest encroachment A. John Kennedy vs State of Tamil Nadu\", \"keep_context\": false, \"bridging_strategy\": \"same_case_full\", \"target_stems\": [], \"statutes\": [], \"retrieval_k\": 6, \"min_full_docs\": 2, \"breadth\": \"specific\", \"reason\": \"Implicit reference to a single case\"}\n"
        "Q: 'Find all Supreme Court cases involving cybercrime or online harassment this year.' -> {\"type\": \"new\", \"rewrite\": \"Supreme Court cases on cybercrime or online harassment in 2025\", \"keep_context\": false, \"bridging_strategy\": \"adjacent\", \"target_stems\": [], \"statutes\": [], \"retrieval_k\": 16, \"min_full_docs\": 4, \"breadth\": \"broad\", \"reason\": \"Broad thematic retrieval\"}\n"
        "Q: 'Was there any Supreme Court judgment about delay in IBC appeals recently?' -> {\"type\": \"new\", \"rewrite\": \"Recent Supreme Court rulings on delay condonation for IBC Section 61 appeals\", \"keep_context\": false, \"bridging_strategy\": \"adjacent\", \"target_stems\": [], \"statutes\": [\"IBC s.61\"], \"retrieval_k\": 12, \"min_full_docs\": 3, \"breadth\": \"narrow\", \"reason\": \"Indirect request for new case\"}\n"
        "Q: 'Show me cases where the High Court order was overturned by the Supreme Court in 2025.' -> {\"type\": \"new\", \"rewrite\": \"Supreme Court 2025 matters reversing High Court decisions\", \"keep_context\": false, \"bridging_strategy\": \"adjacent\", \"target_stems\": [], \"statutes\": [], \"retrieval_k\": 16, \"min_full_docs\": 4, \"breadth\": \"broad\", \"reason\": \"Analytical cross-case query\"}\n"
        "Q: 'I think there was some ruling about forest rights and tiger reserves — can you find it?' -> {\"type\": \"new\", \"rewrite\": \"Supreme Court ruling on forest rights and tiger reserve management\", \"keep_context\": false, \"bridging_strategy\": \"adjacent\", \"target_stems\": [], \"statutes\": [], \"retrieval_k\": 12, \"min_full_docs\": 3, \"breadth\": \"narrow\", \"reason\": \"Partial recall triggering new search\"}\n"
        "Q: 'Was there any corruption-related case decided in February 2025?' -> {\"type\": \"new\", \"rewrite\": \"Supreme Court corruption judgments February 2025\", \"keep_context\": false, \"bridging_strategy\": \"adjacent\", \"target_stems\": [], \"statutes\": [], \"retrieval_k\": 16, \"min_full_docs\": 4, \"breadth\": \"broad\", \"reason\": \"Date-specific retrieval\"}\n"
        "Q: 'Give me a list of all pending Supreme Court cases related to electoral bonds.' -> {\"type\": \"new\", \"rewrite\": \"Pending Supreme Court matters on electoral bonds\", \"keep_context\": false, \"bridging_strategy\": \"adjacent\", \"target_stems\": [], \"statutes\": [], \"retrieval_k\": 16, \"min_full_docs\": 4, \"breadth\": \"broad\", \"reason\": \"Open-ended search for cases\"}\n"
        "Q: 'Can you summarise any rental/property cases from 2025?' -> {\"type\": \"new\", \"rewrite\": \"Supreme Court rental or property law cases decided in 2025 with summaries\", \"keep_context\": false, \"bridging_strategy\": \"adjacent\", \"target_stems\": [], \"statutes\": [], \"retrieval_k\": 16, \"min_full_docs\": 4, \"breadth\": \"broad\", \"reason\": \"Topic-based case summaries for a specific year\"}\n"
        "Q: 'Tell me more about the Abdul Wahid case you mentioned earlier.' -> {\"type\": \"followup\", \"rewrite\": \"Further details on Abdul Wahid case discussed earlier\", \"keep_context\": true, \"bridging_strategy\": \"same_case_full\", \"target_stems\": [], \"statutes\": [], \"retrieval_k\": 6, \"min_full_docs\": 2, \"breadth\": \"specific\", \"reason\": \"Follow-up on prior case\"}\n"
        "Q: 'Can you show me the final part of that Kerala case again?' -> {\"type\": \"followup\", \"rewrite\": \"Retrieve final portion of earlier Kerala Supreme Court case\", \"keep_context\": true, \"bridging_strategy\": \"same_case_full\", \"target_stems\": [], \"statutes\": [], \"retrieval_k\": 4, \"min_full_docs\": 2, \"breadth\": \"specific\", \"reason\": \"References existing context\"}\n"
        "Q: 'What did the Court finally decide in the forest judgment you told me about?' -> {\"type\": \"followup\", \"rewrite\": \"Outcome of previously discussed forest encroachment judgment\", \"keep_context\": true, \"bridging_strategy\": \"same_case_full\", \"target_stems\": [], \"statutes\": [], \"retrieval_k\": 4, \"min_full_docs\": 2, \"breadth\": \"specific\", \"reason\": \"Depends on prior discussion\"}\n"
        "Q: 'In that bribery case list, was there any one where the High Court’s order was overturned?' -> {\"type\": \"followup\", \"rewrite\": \"Check prior bribery case list for Supreme Court reversals of High Courts\", \"keep_context\": true, \"bridging_strategy\": \"adjacent\", \"target_stems\": [], \"statutes\": [], \"retrieval_k\": 8, \"min_full_docs\": 3, \"breadth\": \"narrow\", \"reason\": \"Builds on earlier list\"}\n"
        "Q: 'Compare the reasoning between the two cases you just summarized.' -> {\"type\": \"followup\", \"rewrite\": \"Compare reasoning of the two cases summarized in last turn\", \"keep_context\": true, \"bridging_strategy\": \"same_case_full\", \"target_stems\": [], \"statutes\": [], \"retrieval_k\": 4, \"min_full_docs\": 2, \"breadth\": \"specific\", \"reason\": \"Explicit continuation\"}\n"
        "Q: 'Wasn’t there another similar case about forest encroachment you mentioned?' -> {\"type\": \"followup\", \"rewrite\": \"Locate similar forest encroachment case referenced earlier\", \"keep_context\": true, \"bridging_strategy\": \"adjacent\", \"target_stems\": [], \"statutes\": [], \"retrieval_k\": 6, \"min_full_docs\": 2, \"breadth\": \"narrow\", \"reason\": \"Refers back to prior mention\"}\n"
        "Q: 'Open that earlier judgment about the NCLAT limitation issue again.' -> {\"type\": \"followup\", \"rewrite\": \"Reopen previously cited NCLAT limitation judgment\", \"keep_context\": true, \"bridging_strategy\": \"same_case_full\", \"target_stems\": [], \"statutes\": [\"IBC s.61\"], \"retrieval_k\": 4, \"min_full_docs\": 2, \"breadth\": \"specific\", \"reason\": \"Revisit an earlier case\"}\n"
        "Q: 'Can you highlight the paragraph on \"reasonable doubt\" from the last case?' -> {\"type\": \"followup\", \"rewrite\": \"Fetch passage on reasonable doubt from last discussed case\", \"keep_context\": true, \"bridging_strategy\": \"same_case_full\", \"target_stems\": [], \"statutes\": [], \"retrieval_k\": 4, \"min_full_docs\": 2, \"breadth\": \"specific\", \"reason\": \"Detail from current context\"}\n"
        "Q: 'Okay, now show me what the Supreme Court said next in that same case.' -> {\"type\": \"followup\", \"rewrite\": \"Continue supplying next section of the same Supreme Court case\", \"keep_context\": true, \"bridging_strategy\": \"same_case_full\", \"target_stems\": [], \"statutes\": [], \"retrieval_k\": 4, \"min_full_docs\": 2, \"breadth\": \"specific\", \"reason\": \"Continuation request\"}\n"
        "Q: 'So the earlier case you showed — was it upheld or reversed on appeal?' -> {\"type\": \"followup\", \"rewrite\": \"Confirm whether the earlier case outcome was upheld or reversed on appeal\", \"keep_context\": true, \"bridging_strategy\": \"same_case_full\", \"target_stems\": [], \"statutes\": [], \"retrieval_k\": 4, \"min_full_docs\": 2, \"breadth\": \"specific\", \"reason\": \"Asks about prior case disposition\"}"
    )
    system_msg_manual = (
        "You are a query-processor for a legal RAG assistant. Return STRICT JSON with fields: "
        "{type, rewrite, case_probe, keep_context, bridging_strategy, target_stems, statutes, retrieval_k, min_full_docs, breadth, reason}.\n"
        "Rules: type is one of followup | new | general_chat.\n"
        "ROUTING: Do NOT route to general_law. If there is ANY specificity (numbers, dates, party names, sections, court names, or concrete scenario), choose new or followup for retrieval. "
        "If the user is simply greeting, thanking, making small talk, telling jokes, or asking non-legal trivia/general knowledge, classify it as general_chat.\n"
        "If followup, decide keep_context (true if the current context already contains the case/material needed). "
        "If the user appears to ask for similar cases or statutes beyond current context, set bridging_strategy=statute_refill or adjacent; if they want the same case full, set same_case_full. "
        "Always produce a helpful standalone rewrite for retrieval; expand acronyms and include entities (parties, court, date, case numbers) if known. "
        "Keep the rewrite concise and keyword-rich (≤ 20 tokens). "
        "Always set case_probe to an empty string in manual mode (including general_chat).\n\n"
        "TOP-K SELECTION: When appropriate, set retrieval_k as follows (use judgment; integers only):\n"
        "MIN FULL DOCS: Suggest min_full_docs (integer) ~ proportional to retrieval_k and breadth of query.\n"
        "- Broad/overview queries: min_full_docs ~ 3-6 (at least 2).\n"
        "- Case-specific queries: min_full_docs ~ 2-3.\n"
        "Except for general_chat, never return less than 2.\n\n"
        "BREADTH: Set breadth to \"broad\", \"narrow\", or \"specific\" as defined above.\n\n"
        "- General/very broad questions (no specific case/statute): retrieval_k ~ 12-20\n"
        "- Typical topic queries: retrieval_k ~ 8-12\n"
        "- Case-specific or tightly-focused follow-ups: retrieval_k ~ 4-6\n"
        "If uncertain, pick 8.\n\n"
        "EXAMPLES (label -> JSON):\n"
        "Q: 'In John Kennedy vs State of Tamil Nadu, what was the final order?' -> {\"type\": \"new\", \"rewrite\": \"Final order in A. John Kennedy vs State of Tamil Nadu, 2025 INSC 443 (Supreme Court of India)\", \"keep_context\": false, \"bridging_strategy\": \"same_case_full\", \"target_stems\": [\"1\"], \"statutes\": [], \"retrieval_k\": 6, \"min_full_docs\": 2, \"breadth\": \"specific\", \"reason\": \"Specific case named\"}\n"
        "Q: 'Also list similar cases where IPC 302 was applied' (after a case turn) -> {\"type\": \"followup\", \"rewrite\": \"Supreme Court decisions applying IPC Section 302 similar to <last case>\", \"keep_context\": true, \"bridging_strategy\": \"statute_refill\", \"target_stems\": [], \"statutes\": [\"IPC s.302\"], \"retrieval_k\": 8, \"min_full_docs\": 3, \"breadth\": \"narrow\", \"reason\": \"Follow-up requesting similar cases by statute\"}\n"
        "Q: 'Thanks!' -> {\"type\": \"general_chat\", \"rewrite\": \"\", \"case_probe\": \"\", \"keep_context\": false, \"bridging_strategy\": \"none\", \"target_stems\": [], \"statutes\": [], \"retrieval_k\": 0, \"min_full_docs\": 0, \"breadth\": \"chat\", \"reason\": \"Non-legal appreciation\"}\n"
        "Q: 'What is my name?' -> {\"type\": \"general_chat\", \"rewrite\": \"\", \"case_probe\": \"\", \"keep_context\": false, \"bridging_strategy\": \"none\", \"target_stems\": [], \"statutes\": [], \"retrieval_k\": 0, \"min_full_docs\": 0, \"breadth\": \"chat\", \"reason\": \"Personal memory check\"}\n"
        "Q: 'Can you tell me a fun fact?' -> {\"type\": \"general_chat\", \"rewrite\": \"\", \"case_probe\": \"\", \"keep_context\": false, \"bridging_strategy\": \"none\", \"target_stems\": [], \"statutes\": [], \"retrieval_k\": 0, \"min_full_docs\": 0, \"breadth\": \"chat\", \"reason\": \"General trivia request\"}"
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
            "type": "followup|new|general_law|general_chat",
            "rewrite": "string",
            "case_probe": "string|null",
            "keep_context": True,
            "bridging_strategy": "none|adjacent|statute_refill|same_case_full",
            "target_stems": ["7", "21"],
            "statutes": ["IPC s.302"],
            "retrieval_k": 12,
            "min_full_docs": 3,
            "breadth": "specific|narrow|broad",
            "reason": "short explanation",
        },
    }

    prompt = (
        f"{system_msg}\n\nReturn ONLY JSON per the schema. Do not wrap in code fences.\nPAYLOAD:\n{json.dumps(payload)}"
    )

    try:
        resp = llm.invoke(prompt)
        content = getattr(resp, "content", None) or str(resp)
        if isinstance(content, list):
            content = " ".join(str(x) for x in content)
        parsed_json = None
        try:
            parsed_json = json.loads(content)
        except Exception:
            import re as _re
            m = _re.search(r"\\{.*\\}", content, flags=_re.DOTALL)
            if m:
                parsed_json = json.loads(m.group(0))
        if parsed_json:
            plan = QueryPlan(**parsed_json)
            return _postprocess_plan(plan, user_q)
    except Exception as exc:
        _log_debug(f"[DEBUG][QP][llm_error][model={model_name}|provider={provider_name}] {exc}")

    plan = _heuristic_plan(user_q)
    _log_debug(f"[DEBUG][QP][heuristic_fallback] type={plan.type} rewrite={plan.rewrite}")
    return plan


class PetitionPlan(BaseModel):
    """
    Petition-specific planning output.
    """
    retrieval_query: str
    statutes: List[str] = []
    retrieval_k: int = 10
    min_full_docs: int = 3
    fighting_points: List[str] = []
    issues: List[str] = []
    posture: str = ""
    reason: str | None = None


def _heuristic_petition_plan(petition_text: str, user_context: Optional[str] = None) -> PetitionPlan:
    """
    Fallback heuristic when LLM plan fails: extract rough statutes/keywords and set K/min docs.
    """
    import re as _re
    txt = (petition_text or "")[:8000].lower()
    # crude statute extraction
    statutes: List[str] = []
    for m in _re.findall(r"\b(?:ipc|crpc|pc\s*act|evidence\s+act|prevention\s+of\s+corruption)\b\s*[^,;.\n]*", txt):
        s = m.strip()
        if s and s not in statutes:
            statutes.append(s)
        if len(statutes) >= 6:
            break
    # derive topic keywords
    kws: List[str] = []
    if "sanction" in txt:
        kws.append("sanction validity")
    if "trap" in txt or "phenolphthalein" in txt:
        kws.append("trap procedure flaws")
    if "bribe" in txt or "illegal gratification" in txt:
        kws.append("bribery pc act")
    if not kws:
        kws.append("procedural defects")
    if user_context:
        u = user_context.lower()
        if "sanction" in u and "sanction validity" not in kws:
            kws.append("sanction validity")
        if "trap" in u and "trap procedure flaws" not in kws:
            kws.append("trap procedure flaws")
    rq = " ".join(kws[:3])
    fps: List[str] = []
    if "sanction" in txt or (user_context or "").lower().find("sanction") != -1:
        fps.append("Prosecution sanction invalid or non-application of mind under PC Act s.19")
    if "trap" in txt:
        fps.append("Trap procedure tainted; independent corroboration lacking")
    if "demand" in txt and "bribe" in txt:
        fps.append("Demand of illegal gratification not proved beyond reasonable doubt")
    if not fps:
        fps.append("Petition lacks foundational facts and procedural compliance")
    return PetitionPlan(
        retrieval_query=rq,
        statutes=statutes[:6],
        retrieval_k=12,
        min_full_docs=4,
        fighting_points=fps[:8],
        issues=["procedural"],
        posture="respondent rebuttal",
        reason="Heuristic extraction from petition text",
    )


def process_petition_plan(
    petition_text: str,
    user_context: Optional[str] = None,
    llm=None,
    model_name: Optional[str] = None,
    provider_name: Optional[str] = None,
) -> PetitionPlan:
    """
    Build a petition-aware retrieval plan (separate from chat planner).
    Uses the selected LLM; falls back to heuristics on failure or when LLM is missing.
    """
    if llm is None:
        _log_debug("[DEBUG][PetitionPlan] No LLM provided; using heuristic petition planner")
        return _heuristic_petition_plan(petition_text, user_context)

    from models import get_prompt
    system_msg = get_prompt("petition_planner")
    _log_debug(f"[DEBUG][PetitionPlan] using model={model_name} provider={provider_name}")

    pet = (petition_text or "")[:12000]
    ctx = (user_context or "")[:2000]
    payload = {
        "petition": pet,
        "user_context": ctx,
        "schema": {
            "retrieval_query": "string",
            "statutes": ["PC Act s.19"],
            "retrieval_k": 12,
            "min_full_docs": 4,
            "fighting_points": ["bullet"],
            "issues": ["label"],
            "posture": "respondent rebuttal",
            "reason": "short explanation",
        },
    }

    prompt = (
        f"{system_msg}\n\nReturn ONLY JSON per the schema. Do not wrap in code fences.\nPAYLOAD:\n{json.dumps(payload)}"
    )

    try:
        resp = llm.invoke(prompt)
        content = getattr(resp, "content", None) or str(resp)
        if isinstance(content, list):
            content = " ".join(str(x) for x in content)
        parsed_json = None
        try:
            parsed_json = json.loads(content)
        except Exception:
            import re as _re
            m = _re.search(r"\{.*\}", content, flags=_re.DOTALL)
            if m:
                parsed_json = json.loads(m.group(0))
        if parsed_json:
            return PetitionPlan(**parsed_json)
    except Exception as e:
        _log_debug(f"[DEBUG][PetitionPlan][llm_error][model={model_name}|provider={provider_name}] {e}")
    return _heuristic_petition_plan(petition_text, user_context)


def _extract_text_from_response(resp) -> str:
    """Best-effort extraction of text from OpenAI Responses API output."""
    try:
        txt = getattr(resp, "output_text", None)
        if isinstance(txt, str) and txt.strip():
            return txt.strip()
    except Exception:
        pass
    try:
        outputs = getattr(resp, "output", None) or []
        for out in outputs:
            content = getattr(out, "content", None) or []
            for item in content:
                if getattr(item, "type", None) in {"output_text", "text"}:
                    text_val = getattr(item, "text", "")
                    if text_val:
                        return str(text_val).strip()
    except Exception:
        pass
    return ""


def generate_petition_context(petition_text: str, llm=None, model_name: str | None = None) -> str:
    """
    Generate a compact petition context using the currently selected chat model.
    """
    text = (petition_text or "").strip()
    if not text:
        raise ValueError("petition_text is empty")

    from models import get_prompt

    system_msg = get_prompt("petition_context")
    prompt = f"{system_msg}\n\nPETITION:\n{text}"

    if llm is None:
        raise RuntimeError("LLM is required for petition context generation (no OpenAI fallback).")

    try:
        resp = llm.invoke(prompt)
        out = getattr(resp, "content", None) or str(resp)
        out_str = out if isinstance(out, str) else str(out)
        cleaned = out_str.strip()
        if cleaned:
            return cleaned
    except Exception as exc:
        _log_debug(f"[DEBUG][PetitionContext][llm_error][model={model_name}] {exc}")
        raise

    raise RuntimeError("Received empty context from petition LLM")
