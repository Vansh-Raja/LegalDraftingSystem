"""
Drafting pipeline utilities for generating response drafts to uploaded petitions.

MVP goals:
- Extract text from uploaded files (PDF/DOCX/TXT)
- Run a lightweight profiler to infer document type, key issues, statutes, and 3 recommended queries
- Reuse existing RAG retrieval utilities to fetch context for those queries
- Assemble a simple draft (LLM-backed when configured; heuristic fallback otherwise)
"""

from __future__ import annotations

import io
import json
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pathlib import Path

from rag import (
    get_vectorstore,
    summary_search_pg,
    fuse_cases_by_rrf,
    fetch_top_chunks_for_cases,
    interleave_docs_by_case,
    filtration_retriever,
    assemble_context_from_plan,
    build_retriever,
    enforce_case_diversity,
)
from orchestrator import process_query
from langchain_core.documents import Document

# Optional DOCX support
try:
    import docx  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    docx = None

from functions import extract_text_from_pdf as _extract_pdf
from ai import _ensure_json_dict as _ensure_json_dict
from ai import _truncate_text as _truncate_text
from ai import _extract_text_from_responses as _extract_resp
from dotenv import load_dotenv
import os
from openai import OpenAI


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


# Mirror chat pipeline model limits for drafting workflows
MODEL_CONTEXT_WINDOWS: Dict[str, int] = {
    "gpt-5-nano-2025-08-07": 400_000,
}

RESERVED_COMPLETION_TOKENS: Dict[str, int] = {
    "gpt-5-nano-2025-08-07": 20_000,
}


def drafting_context_budget_tokens(model_name: str = "gpt-5-nano-2025-08-07") -> int:
    window = MODEL_CONTEXT_WINDOWS.get(model_name, 32768)
    reserve = RESERVED_COMPLETION_TOKENS.get(model_name, 2000)
    return max(1000, window - reserve)


def drafting_completion_tokens(model_name: str = "gpt-5-nano-2025-08-07") -> int:
    return max(500, RESERVED_COMPLETION_TOKENS.get(model_name, 2000))


@dataclass
class PetitionProfile:
    document_type: str
    key_issues: List[str]
    statutes: List[str]
    requested_relief: str
    opponent_type: str
    parties: Dict[str, str]
    notable_facts: List[str]
    domain_tags: List[str]
    recommended_queries: List[Dict[str, Any]]


def _extract_text_from_docx_bytes(raw: bytes) -> str:
    if docx is None:
        return ""
    f = io.BytesIO(raw)
    d = docx.Document(f)
    parts = [p.text for p in d.paragraphs if p.text]
    return "\n".join(parts).strip()


def extract_text_from_upload(file_name: str, mime_type: str | None, raw_bytes: bytes) -> str:
    name_lower = (file_name or "").lower()
    mt = (mime_type or "").lower()
    if name_lower.endswith(".txt") or mt == "text/plain":
        try:
            return raw_bytes.decode("utf-8", errors="ignore")
        except Exception as exc:
            raise ValueError(f"Failed to decode text file: {exc}") from exc
    if name_lower.endswith(".pdf") or mt == "application/pdf":
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tf:
            tf.write(raw_bytes)
            tf.flush()
            try:
                return _extract_pdf(tf.name)
            except Exception as exc:
                logger.exception("PDF extraction failed for %s", file_name)
                raise ValueError(f"Failed to extract PDF text: {exc}") from exc
    if name_lower.endswith(".docx") or mt.endswith("officedocument.wordprocessingml.document"):
        if docx is None:
            raise ValueError("python-docx package is not installed; cannot read DOCX files")
        try:
            return _extract_text_from_docx_bytes(raw_bytes)
        except Exception as exc:
            logger.exception("DOCX extraction failed for %s", file_name)
            raise ValueError(f"Failed to extract DOCX text: {exc}") from exc
    try:
        return raw_bytes.decode("utf-8", errors="ignore")
    except Exception as exc:
        logger.exception("Fallback decode failed for %s", file_name)
        raise ValueError(f"Could not decode uploaded file: {exc}") from exc


def _simple_detect_statutes(text: str) -> List[str]:
    t = (text or "").lower()
    found: List[str] = []
    for key, label in [
        (" crpc ", "CrPC"),
        (" ipc ", "IPC"),
        (" evidence act", "Evidence Act"),
        (" cpc ", "CPC"),
        (" constitution", "Constitution"),
        (" article 226", "Constitution Art. 226"),
        (" article 32", "Constitution Art. 32"),
        (" s. 482", "CrPC S.482"),
        (" section 482", "CrPC S.482"),
    ]:
        if key in f" {t} ":
            found.append(label)
    return sorted(list(dict.fromkeys(found)))


def heuristic_profile(petition_text: str, user_context: str | None = None) -> PetitionProfile:
    """
    Lightweight profiler when LLM is unavailable. Extracts bare minimum fields
    and synthesizes three recommended queries tailored for reply drafting.
    """
    text = (petition_text or "").strip()
    statutes = _simple_detect_statutes(text)
    # Crude document type guess
    tl = text.lower()
    if " anticipatory bail" in tl or " s. 438" in tl or " section 438" in tl:
        doc_type = "anticipatory_bail"
    elif " bail " in f" {tl} ":
        doc_type = "bail"
    elif " quash" in tl or " 482" in tl:
        doc_type = "quash"
    elif " writ" in tl or " article 226" in tl or " article 32" in tl:
        doc_type = "writ"
    else:
        doc_type = "reply"

    # Minimal issues
    issues = []
    if "jurisdiction" in tl:
        issues.append("jurisdiction")
    if "delay" in tl or "laches" in tl:
        issues.append("delay/laches")
    if "alternative remedy" in tl or "alternate remedy" in tl:
        issues.append("alternate remedy")
    if "suppression" in tl:
        issues.append("suppression/clean hands")
    issues = issues[:4] or ["thresholds", "substantive grounds"]

    # Relief guess
    if "stay" in tl or "interim" in tl:
        relief = "interim/stay"
    elif doc_type in {"bail", "anticipatory_bail"}:
        relief = doc_type
    elif doc_type == "quash":
        relief = "quash"
    else:
        relief = "dismiss petition / deny relief"

    # Opponent type (very rough)
    opponent = "state" if any(k in tl for k in ["state of", "union of india", "police", "department"]) else "private"

    # Parties placeholder
    parties = {"petitioner": "", "respondent": ""}

    facts = []

    rec_qs: List[Dict[str, Any]] = []
    domain_tags: List[str] = []

    buzzword_pack: Optional[Dict[str, Any]] = None

    if doc_type in {"bail", "anticipatory_bail", "quash", "criminal_revision"}:
        domain_tags = ["criminal", "pc_act"]
        maintainability_pack = {
            "intent": "maintainability",
            "broad": "(\"Article 136\" OR \"Article 226\") AND (\"alternate remedy\" OR \"suppression\" OR \"laches\") AND (\"criminal proceedings\" OR \"investigation\")",
            "alternates": [
                "(\"quash\" AND (\"delay\" OR \"clean hands\"))",
                "(\"anticipatory bail\" AND \"maintainability\")",
            ],
            "must_include": ["Article 226"],
            "should_include": ["alternate remedy", "suppression", "laches", "maintainability"],
            "exclude": ["consumer forum", "arbitration"],
            "filters_hint": {"prefer_courts": ["Supreme Court"], "timeframe": "2000-present", "statute_family": ["Constitution"]},
            "keywords": ["maintainability", "clean hands", "delay"],
        }
        statute_pack = {
            "intent": "statute_standard",
            "broad": "(\"Prevention of Corruption Act\" OR \"PC Act\") AND (\"Section 19\" OR \"s 19\") AND (\"sanction\" OR \"failure of justice\")",
            "alternates": [
                "(\"Section 7\" OR \"Section 13\") AND (\"demand and acceptance\" OR \"illegal gratification\")",
                "(\"sanction\" AND \"independent application of mind\")",
            ],
            "must_include": ["Prevention of Corruption Act", "Section 19"],
            "should_include": ["sanction", "competent authority", "failure of justice"],
            "exclude": [],
            "filters_hint": {"prefer_courts": ["Supreme Court"], "timeframe": "none", "statute_family": ["PC Act"]},
            "keywords": ["sanction", "Section 19", "competent authority"],
        }
        fact_pack = {
            "intent": "fact_pattern",
            "broad": "(\"demand and acceptance\" AND (\"illegal gratification\" OR \"bribe\")) AND (\"trap proceedings\" OR \"phenolphthalein\")",
            "alternates": [
                "(\"trap\" AND \"independent witnesses\")",
                "(\"sanction delay\" OR \"sanction irregularity\")",
            ],
            "must_include": ["demand and acceptance"],
            "should_include": ["trap laying officer", "independent witness", "phenolphthalein"],
            "exclude": ["PMLA", "NDPS"],
            "filters_hint": {"prefer_courts": ["Supreme Court"], "timeframe": "2000-present", "statute_family": ["PC Act", "Evidence Act"]},
            "keywords": ["trap", "phenolphthalein", "tainted currency"],
        }
        buzzword_pack = {
            "intent": "buzzwords",
            "broad": "(\"illegal gratification\" OR \"bribery\" OR \"sanction order\" OR \"CBI raid\")",
            "alternates": ["(\"Section 7\" OR \"Section 13\") AND \"PC Act\"", "\"failure of justice\" AND sanction"],
            "must_include": ["bribery"],
            "should_include": ["PC Act", "sanction", "trap"],
            "exclude": [],
            "filters_hint": {"prefer_courts": ["Supreme Court"], "timeframe": "2000-present", "statute_family": ["PC Act"]},
            "keywords": ["bribe", "sanction", "competent authority"],
        }
        rec_qs = [maintainability_pack, statute_pack, fact_pack]
    else:
        domain_tags = ["general_writ"]
        rec_qs = [
            {
                "intent": "maintainability",
                "broad": "(\"Article 226\" OR \"Art 226\") AND (\"alternate remedy\" OR \"suppression\" OR \"laches\")",
                "alternates": [
                    "(\"writ petition\" AND \"delay\")",
                    "(\"clean hands\" AND \"jurisdiction\")",
                ],
                "must_include": ["Article 226"],
                "should_include": ["alternate remedy", "suppression", "laches"],
                "exclude": ["consumer forum", "arbitration"],
                "filters_hint": {"prefer_courts": ["Supreme Court"], "timeframe": "2000-present", "statute_family": ["Constitution"]},
                "keywords": ["maintainability", "clean hands", "delay"],
            },
            {
                "intent": "statute_standard",
                "broad": "(\"Article 32\" OR \"Article 226\") AND (\"standard of review\" OR \"judicial review\")",
                "alternates": [
                    "(\"service law\" AND \"writ jurisdiction\")",
                    "(\"injunction\" AND \"Order 39\")",
                ],
                "must_include": ["Article"],
                "should_include": ["judicial review", "scope", "standard"],
                "exclude": [],
                "filters_hint": {"prefer_courts": ["Supreme Court"], "timeframe": "2000-present", "statute_family": ["Constitution", "CPC"]},
                "keywords": ["judicial review", "scope of interference"],
            },
            {
                "intent": "fact_pattern",
                "broad": "(\"fundamental rights\" AND \"violation\") AND (\"interim relief\" OR \"stay\")",
                "alternates": [
                    "(\"service matter\" AND \"suspension\")",
                    "(\"statutory appeal\" AND \"remedy\")",
                ],
                "must_include": ["fundamental rights"],
                "should_include": ["interim relief", "stay", "writ"],
                "exclude": ["arbitration", "consumer"],
                "filters_hint": {"prefer_courts": ["Supreme Court"], "timeframe": "2000-present", "statute_family": ["Constitution"]},
                "keywords": ["interim relief", "stay", "fundamental rights"],
            },
        ]
        buzzword_pack = {
            "intent": "buzzwords",
            "broad": "(\"bribery\" OR \"illegal gratification\" OR \"sanction\" OR \"trap\")",
            "alternates": ["(\"Prevention of Corruption Act\" AND \"Section 19\")", "\"CBI\" AND \"sanction order\""],
            "must_include": [],
            "should_include": ["sanction", "bribery", "CBI"],
            "exclude": [],
            "filters_hint": {"prefer_courts": ["Supreme Court"], "timeframe": "2000-present", "statute_family": ["PC Act", "CrPC", "Constitution"]},
            "keywords": ["bribe", "sanction", "trap", "note-sheet"],
        }

    packs = rec_qs[:3]
    if buzzword_pack:
        packs.append(buzzword_pack)

    return PetitionProfile(
        document_type=doc_type,
        key_issues=issues,
        statutes=statutes,
        requested_relief=relief,
        opponent_type=opponent,
        parties=parties,
        notable_facts=facts,
        domain_tags=domain_tags or ["general"],
        recommended_queries=packs,
    )


def llm_profile(petition_text: str, user_context: Optional[str] = None, max_chars: int = 12000) -> PetitionProfile:
    """
    LLM-based profiler using OpenAI (gpt-5-nano) to extract a drafting profile
    and produce three recommended RAG queries for Indian reply drafting.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_KEY")
    if not api_key:
        return heuristic_profile(petition_text, user_context)
    client = OpenAI(api_key=api_key)

    system_msg = """You are an Indian legal drafting assistant generating a RESPONDENT profile for Supreme Court drafting.
Return STRICT JSON ONLY with keys: {"document_type","key_issues","statutes","requested_relief","opponent_type","parties","notable_facts","domain_tags","recommended_queries"}.

Guidance:
- document_type: choose from [writ, bail, anticipatory_bail, quash, injunction, service, tax, land, criminal_revision, reply].
- key_issues: 2-4 concise issues prioritising threshold bars (alternate remedy, delay/laches, suppression/clean hands, sanction defects, jurisdiction) plus at most two merits defences.
- statutes: list canonical citations (e.g., Prevention of Corruption Act 1988 s.19, CrPC 482, IPC 302, CPC O.39, Constitution Art.226/32).
- requested_relief: brief description.
- opponent_type: state | authority | private.
- parties: object with "petitioner" and "respondent" (empty strings if unknown).
- notable_facts: 2-5 neutral bullets covering chronology, sanction status, investigative irregularities, delays, prior litigation.
- domain_tags: list of 2-4 high-level domain keywords (e.g., ["pc_act","sanction","trap"], or ["criminal_murder"], ["land_dispute"], ["service_law"], etc.).
- recommended_queries: EXACTLY three objects (maintainability, statute_standard, fact_pattern) that maximise recall. Each object MUST contain:
  {
    "intent": "maintainability | statute_standard | fact_pattern",
    "broad": "BOOLEAN query with OR/AND synonyms, no party names/dates",
    "alternates": ["broader OR variant query 1", "adjacent angle query 2"],
    "must_include": ["anchor tokens"],
    "should_include": ["useful synonyms"],
    "exclude": ["noise terms"],
    "filters_hint": {"prefer_courts": ["Supreme Court"], "timeframe": "none | 2000-present", "statute_family": ["PC Act","CrPC","IPC", ...]},
    "keywords": ["additional buzzwords for fallback"]
  }
Rules for queries: no party names/FIR numbers/cities; heavy use of OR synonyms; include statute abbreviations and full forms; emphasise sanction/bribe/trap when relevant; ensure must_include anchors keep the query on-topic; exclude unrelated domains.
"""
    example_json = (
        "{\n"
        "  \"document_type\": \"writ\",\n"
        "  \"key_issues\": [\"alternate remedy\", \"delay/laches\"],\n"
        "  \"statutes\": [\"Prevention of Corruption Act 1988 s.19\", \"CrPC 482\"],\n"
        "  \"requested_relief\": \"quash sanction and FIR\",\n"
        "  \"opponent_type\": \"state\",\n"
        "  \"parties\": {\"petitioner\": \"Senior Engineer\", \"respondent\": \"State of Kerala\"},\n"
        "  \"notable_facts\": [\"Sanction order dated 12-02-2024 lacks independent mind\", \"Trap arranged on complaint without preliminary inquiry\"],\n"
        "  \"domain_tags\": [\"pc_act\", \"sanction\", \"trap\"],\n"
        "  \"recommended_queries\": [\n"
        "    {\n"
        "      \"intent\": \"maintainability\",\n"
        "      \"broad\": \"(\\\"Article 226\\\" OR \\\"Art 226\\\") AND (\\\"alternate remedy\\\" OR \\\"suppression\\\" OR \\\"laches\\\") AND (\\\"criminal investigation\\\" OR \\\"FIR\\\")\",\n"
        "      \"alternates\": [\"(\\\"quash FIR\\\" AND (\\\"delay\\\" OR \\\"clean hands\\\"))\", \"(\\\"PC Act\\\" AND \\\"writ maintainability\\\")\"],\n"
        "      \"must_include\": [\"Article 226\"],\n"
        "      \"should_include\": [\"alternate remedy\", \"suppression\", \"laches\"],\n"
        "      \"exclude\": [\"consumer forum\", \"arbitration\"],\n"
        "      \"filters_hint\": {\"prefer_courts\": [\"Supreme Court\"], \"timeframe\": \"2000-present\", \"statute_family\": [\"Constitution\"]},\n"
        "      \"keywords\": [\"maintainability\", \"clean hands\"]\n"
        "    },\n"
        "    {\n"
        "      \"intent\": \"statute_standard\",\n"
        "      \"broad\": \"(\\\"Prevention of Corruption Act\\\" OR \\\"PC Act\\\") AND (\\\"Section 19\\\" OR \\\"s 19\\\") AND (\\\"sanction\\\" OR \\\"independent application of mind\\\")\",\n"
        "      \"alternates\": [\"(\\\"sanction\\\" AND \\\"failure of justice\\\")\", \"(\\\"competent authority\\\" AND \\\"PC Act\\\")\"],\n"
        "      \"must_include\": [\"Section 19\"],\n"
        "      \"should_include\": [\"sanction\", \"independent mind\", \"failure of justice\"],\n"
        "      \"exclude\": [],\n"
        "      \"filters_hint\": {\"prefer_courts\": [\"Supreme Court\"], \"timeframe\": \"none\", \"statute_family\": [\"PC Act\"]},\n"
        "      \"keywords\": [\"sanction order\", \"competent authority\"]\n"
        "    },\n"
        "    {\n"
        "      \"intent\": \"fact_pattern\",\n"
        "      \"broad\": \"(\\\"demand and acceptance\\\" AND (\\\"illegal gratification\\\" OR \\\"bribe\\\")) AND (\\\"trap proceedings\\\" OR \\\"phenolphthalein\\\")\",\n"
        "      \"alternates\": [\"(\\\"trap\\\" AND \\\"independent witnesses\\\")\", \"(\\\"PC Act\\\" AND \\\"hostile complainant\\\")\"],\n"
        "      \"must_include\": [\"trap\"],\n"
        "      \"should_include\": [\"phenolphthalein\", \"tainted currency\", \"demand\"],\n"
        "      \"exclude\": [\"NDPS\", \"PMLA\"],\n"
        "      \"filters_hint\": {\"prefer_courts\": [\"Supreme Court\"], \"timeframe\": \"2000-present\", \"statute_family\": [\"PC Act\", \"Evidence Act\"]},\n"
        "      \"keywords\": [\"trap laying officer\", \"independent witness\"]\n"
        "    }\n"
        "  ]\n"
        "}"
    )
    text = _truncate_text(petition_text or "", max_chars=max_chars)
    ctx = (user_context or "").strip()
    user_prompt = (
        "Analyse the petition and client context. Produce the JSON profile and the three queries described in the system message."
        f"\n\nClient context: {ctx or 'None provided.'}"
        f"\n\nPetition text (truncated):\n{text}\n\nJSON:"
    )
    try:
        # Prefer Responses API for structured output
        resp = client.responses.create(
            model="gpt-5-nano-2025-08-07",
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": system_msg}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
            ],
            text={"format": {"type": "json_object"}},
            max_output_tokens=900,
        )
        content = _extract_resp(resp)
        data = _ensure_json_dict(content)
    except Exception:
        # Fallback to Chat Completions
        resp = client.chat.completions.create(
            model="gpt-5-nano-2025-08-07",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt + "\nExample JSON:\n" + example_json},
            ],
            max_tokens=900,
        )
        content = (resp.choices[0].message.content or "")
        data = _ensure_json_dict(content)

    # Coerce into PetitionProfile
    def _as_list(x):
        if not x:
            return []
        if isinstance(x, list):
            return [str(v) for v in x]
        return [str(x)]

    base = heuristic_profile(petition_text, user_context)
    key_issues = [s.strip() for s in _as_list(data.get("key_issues")) if str(s).strip()]
    statutes = [s.strip() for s in _as_list(data.get("statutes")) if str(s).strip()]
    notable = [s.strip() for s in _as_list(data.get("notable_facts")) if str(s).strip()]

    def _normalise_query_pack(item: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict):
            return None
        intent = str(item.get("intent", "")).strip().lower()
        if intent not in {"maintainability", "statute_standard", "fact_pattern"}:
            return None
        pack: Dict[str, Any] = {
            "intent": intent,
            "broad": str(item.get("broad", "")).strip(),
            "alternates": [str(v).strip() for v in item.get("alternates", []) if str(v).strip()],
            "must_include": [str(v).strip() for v in item.get("must_include", []) if str(v).strip()],
            "should_include": [str(v).strip() for v in item.get("should_include", []) if str(v).strip()],
            "exclude": [str(v).strip() for v in item.get("exclude", []) if str(v).strip()],
            "filters_hint": item.get("filters_hint") or {},
            "keywords": [str(v).strip() for v in item.get("keywords", []) if str(v).strip()],
        }
        if not pack["broad"]:
            return None
        if not isinstance(pack["filters_hint"], dict):
            pack["filters_hint"] = {}
        return pack

    rec_raw = data.get("recommended_queries")
    rec_queries: List[Dict[str, Any]] = []
    if isinstance(rec_raw, list):
        for item in rec_raw:
            pack = _normalise_query_pack(item)
            if pack:
                rec_queries.append(pack)

    if not key_issues:
        key_issues = base.key_issues
    if not statutes:
        statutes = base.statutes
    if not notable:
        notable = base.notable_facts
    if len(rec_queries) < 3:
        rec_queries = base.recommended_queries

    domain_tags = [s.strip() for s in _as_list(data.get("domain_tags")) if str(s).strip()]
    if not domain_tags:
        domain_tags = base.domain_tags

    parties = data.get("parties") or base.parties
    return PetitionProfile(
        document_type=str(data.get("document_type", base.document_type) or base.document_type),
        key_issues=key_issues[:6],
        statutes=statutes[:12],
        requested_relief=str(data.get("requested_relief", base.requested_relief) or base.requested_relief),
        opponent_type=str(data.get("opponent_type", base.opponent_type) or base.opponent_type),
        parties=parties,
        notable_facts=notable[:8] or base.notable_facts,
        domain_tags=domain_tags[:6],
        recommended_queries=rec_queries[:3],
    )


def run_retrieval_for_queries(
    queries: List[str],
    per_case_k: int = 2,
    top_cases: int = 20,
    court_name: Optional[str] = "Supreme Court of India",
) -> Tuple[str, List[str]]:
    """
    Reuse existing RAG utilities to fetch chunks for each query and merge
    into a single context string. Returns (context_text, case_file_stems).
    """
    vs = get_vectorstore()
    merged_docs: List[Document] = []
    seen_pairs: set[Tuple[str, int]] = set()
    all_cases: List[str] = []

    for q in queries:
        # summary search (case-level)
        try:
            sum_ranked = summary_search_pg(q, court_name=court_name, statutes=None, year=None, limit=max(50, top_cases * 2))
        except Exception:
            sum_ranked = []
        # quick vector per-case chunks via fused cases
        try:
            # Simple vector pull to seed docs for fusion
            docs_vector = vs.similarity_search(q, k=max(8, per_case_k * 3), filter={"court_name": court_name} if court_name else None)
        except Exception:
            docs_vector = []

        try:
            fused = fuse_cases_by_rrf(sum_ranked, docs_vector, k=60, top_n=top_cases)
        except Exception:
            fused = [fs for fs, _rk in sum_ranked[:top_cases]]

        # Fetch top chunks per fused case
        if fused:
            try:
                docs = fetch_top_chunks_for_cases(vs, q, fused, per_case_k=per_case_k)
                docs = interleave_docs_by_case(docs, per_case_limit=per_case_k, max_total=None)
            except Exception:
                docs = docs_vector
        else:
            docs = docs_vector

        # Deduplicate and merge
        for d in docs:
            fs = (d.metadata or {}).get("file_stem")
            ci = (d.metadata or {}).get("chunk_index")
            if fs is None or ci is None:
                continue
            pair = (fs, int(ci))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            merged_docs.append(d)
            all_cases.append(fs)

    # Build a simple merged context
    parts: List[str] = []
    for d in merged_docs:
        fs = (d.metadata or {}).get("file_stem")
        header = f"[Case file: {fs}.txt]\n" if fs else ""
        parts.append(header + (d.page_content or ""))
    context = "\n\n---\n\n".join(parts)
    # Keep case order
    ordered_cases = list(dict.fromkeys(all_cases))
    return context, ordered_cases


def _load_case_metadata_for_stems(stems: List[str], breadth: str = "narrow") -> List[dict]:
    metas: List[dict] = []
    for rank_idx, fs in enumerate(stems, 1):
        if not fs:
            continue
        mp = Path("processed_data/metadata") / f"{fs}.json"
        if not mp.exists():
            continue
        try:
            import json as _json
            data = _json.loads(mp.read_text(encoding="utf-8"))
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
            "summary": str(data.get("summary", ""))[:700],
            "final_judgment": str(data.get("final_judgment", ""))[:400],
            "statutes": statutes_list,
            "parties": data.get("parties"),
            "year": data.get("year"),
            "title": data.get("title"),
            "breadth_hint": breadth,
            "metadata": data,
        })
    return metas



def run_retrieval_for_queries_full(
    queries: List[Dict[str, Any]],
    domain_tags: List[str],
    budget_tokens: int,
    per_case_k: int = 2,
    top_cases: int = 20,
    court_name: Optional[str] = "Supreme Court of India",
) -> Tuple[Dict[str, Any], Dict[str, Dict[Any, str]], List[str], str]:
    """Run retrieval for structured query packs and return metadata for the selector."""
    vs = get_vectorstore()
    debug_lines: List[str] = []

    selection_payload: Dict[str, Any] = {
        "budget_tokens": int(budget_tokens),
        "target_budget_tokens": int(math.floor(budget_tokens * 0.8)),
        "domain_tags": domain_tags,
        "queries": [],
        "cases": [],
    }
    cases_map: Dict[str, Dict[str, Any]] = {}
    full_doc_texts: Dict[str, str] = {}
    chunk_texts: Dict[Tuple[str, int], str] = {}
    digest_parts: List[str] = []

    def _attempt(query_text: str, label: str, k: int, court: Optional[str], stats_filter: Optional[List[str]]):
        attempt_entry: Dict[str, Any] = {"label": label, "query": query_text}
        try:
            sr = summary_search_pg(
                query_text,
                court_name=court,
                statutes=stats_filter,
                year=None,
                limit=max(50, k * 8),
            )
        except Exception:
            sr = []
        attempt_entry["summary_hits"] = len(sr)
        retriever = build_retriever(
            vs,
            statute_filters=stats_filter,
            court_name=court,
            k=k,
        )
        try:
            dv = retriever.invoke(query_text)
        except Exception:
            dv = []
        attempt_entry["vector_chunks"] = len(dv)
        return attempt_entry, sr, dv

    for idx, pack in enumerate(queries, start=1):
        if not isinstance(pack, dict):
            continue
        intent = str(pack.get("intent", "")).strip().lower()
        base_query = str(pack.get("broad", "")).strip()
        if not base_query:
            debug_lines.append(f"[DRAFT][Q{idx}] skipped empty query for intent={intent}")
            continue

        qp = process_query(
            base_query,
            last_question_rewrite=None,
            last_stems=None,
            last_filters=None,
            last_context_snippet=None,
            summary=None,
            manual_mode=True,
        )
        planner_rewrite = (getattr(qp, "rewrite", None) or base_query).strip()
        debug_lines.append(
            f"[DRAFT][Q{idx}] intent={intent or 'unknown'} planner rewrite='{planner_rewrite}' type={getattr(qp, 'type', 'unknown')}"
        )

        retrieval_k = getattr(qp, "retrieval_k", None)
        try:
            effective_k = max(3, min(20, int(retrieval_k))) if retrieval_k else 8
        except Exception:
            effective_k = 8
        min_full_docs = getattr(qp, "min_full_docs", None)
        try:
            desired_min_full_docs = max(2, int(min_full_docs)) if min_full_docs else 3
        except Exception:
            desired_min_full_docs = 3
        breadth = getattr(qp, "breadth", "narrow") or "narrow"
        statute_filters = getattr(qp, "statutes", None) or None

        attempt_history: List[Dict[str, Any]] = []
        final_court = court_name
        final_statutes = statute_filters
        attempt_entry, sum_ranked, docs_vector = _attempt(planner_rewrite, "primary", effective_k, final_court, final_statutes)
        attempt_history.append(attempt_entry)

        if not sum_ranked and not docs_vector:
            for alt in pack.get("alternates", []) or []:
                alt_q = str(alt or "").strip()
                if not alt_q:
                    continue
                alt_entry, alt_sum, alt_docs = _attempt(alt_q, "alternate", effective_k, final_court, final_statutes)
                attempt_history.append(alt_entry)
                if alt_sum or alt_docs:
                    planner_rewrite = alt_q
                    sum_ranked = alt_sum
                    docs_vector = alt_docs
                    final_court = court_name
                    final_statutes = statute_filters
                    break

        if not sum_ranked and not docs_vector:
            fallback_tokens = [
                str(t).strip()
                for t in ((pack.get("must_include") or []) + (pack.get("keywords") or []) + (pack.get("should_include") or []) + list(domain_tags))
                if isinstance(t, str) and str(t).strip()
            ]
            fallback_tokens = list(dict.fromkeys(fallback_tokens))[:6]
            if fallback_tokens:
                wrapped = [tok if " " not in tok else f'"{tok}"' for tok in fallback_tokens]
                fallback_query = wrapped[0] if len(wrapped) == 1 else "(" + " OR ".join(wrapped) + ")"
                fb_entry, fb_sum, fb_docs = _attempt(fallback_query, "fallback-tight", effective_k, final_court, final_statutes)
                attempt_history.append(fb_entry)
                if fb_sum or fb_docs:
                    planner_rewrite = fallback_query
                    sum_ranked = fb_sum
                    docs_vector = fb_docs
                else:
                    fbw_entry, fbw_sum, fbw_docs = _attempt(fallback_query, "fallback-wide", effective_k, None, None)
                    attempt_history.append(fbw_entry)
                    if fbw_sum or fbw_docs:
                        planner_rewrite = fallback_query
                        sum_ranked = fbw_sum
                        docs_vector = fbw_docs
                        final_court = None
                        final_statutes = None

        debug_lines.append(f"[DRAFT][Q{idx}] summary hits={len(sum_ranked)}")
        debug_lines.append(f"[DRAFT][Q{idx}] vector chunks={len(docs_vector)}")

        query_entry = {
            "intent": intent,
            "original_query": base_query,
            "planner_rewrite": planner_rewrite,
            "planner_type": getattr(qp, "type", "unknown"),
            "retrieval_k": effective_k,
            "min_full_docs": desired_min_full_docs,
            "breadth": breadth,
            "statute_filters": statute_filters,
            "alternates": pack.get("alternates", []),
            "must_include": pack.get("must_include", []),
            "should_include": pack.get("should_include", []),
            "exclude": pack.get("exclude", []),
            "keywords": pack.get("keywords", []),
            "attempt_history": attempt_history[:4],
        }

        if not sum_ranked and not docs_vector:
            debug_lines.append(f"[DRAFT][Q{idx}] no retrieval results even after fallbacks")
            selection_payload["queries"].append(query_entry)
            continue

        selection_payload["queries"].append(
            {
                **query_entry,
                "used_query": planner_rewrite,
                "used_filters": {"court_name": final_court, "statute_filters": final_statutes},
            }
        )

        rank_map: Dict[str, int] = {}
        if sum_ranked:
            for pos, (fs, _score) in enumerate(sum_ranked, start=1):
                if fs:
                    rank_map[fs] = pos
        else:
            for pos, doc in enumerate(docs_vector, start=1):
                fs = (doc.metadata or {}).get("file_stem")
                if fs:
                    rank_map.setdefault(fs, pos + 1000)

        chunk_doc_map: Dict[Tuple[str, int], Document] = {}
        for doc in docs_vector:
            fs = (doc.metadata or {}).get("file_stem")
            ci = (doc.metadata or {}).get("chunk_index")
            if fs is None or ci is None:
                continue
            chunk_doc_map[(fs, int(ci))] = doc

        if docs_vector:
            docs = docs_vector
        else:
            docs = []

        if rank_map:
            stems = list(rank_map.keys())
        else:
            stems = []

        if stems:
            try:
                docs = fetch_top_chunks_for_cases(vs, planner_rewrite, stems, per_case_k=per_case_k)
                docs = interleave_docs_by_case(docs, per_case_limit=per_case_k, max_total=None)
                for doc in docs:
                    fs = (doc.metadata or {}).get("file_stem")
                    ci = (doc.metadata or {}).get("chunk_index")
                    if fs is None or ci is None:
                        continue
                    chunk_doc_map.setdefault((fs, int(ci)), doc)
            except Exception:
                pass

        case_metas = _load_case_metadata_for_stems(stems, breadth=breadth)

        query_ctx = {
            "breadth": breadth,
            "fused_case_count": len(stems),
            "retrieval_k": effective_k,
            "planner_min_full_docs": desired_min_full_docs,
        }

        plan = filtration_retriever(
            planner_rewrite,
            docs,
            mode="chunk",
            case_metadatas=case_metas,
            desired_min_full_docs=desired_min_full_docs,
            desired_max_chunks_per_case=2,
            query_context=query_ctx,
            purpose="draft",
        )
        plan = enforce_case_diversity(
            plan,
            fused_cases=stems,
            desired_min_full_docs=desired_min_full_docs,
            desired_max_chunks_per_case=2,
            breadth=breadth,
            allow_expansion=bool(getattr(plan, "request_more_cases", False) and breadth == "broad"),
        )
        debug_lines.append(f"[DRAFT][Q{idx}] plan full_docs={len(plan.selected_full_docs)} chunks={len(plan.selected_chunks)}")

        for meta in case_metas:
            fs = meta.get("file_stem")
            if not fs:
                continue
            entry = cases_map.setdefault(
                fs,
                {
                    "file_stem": fs,
                    "case_number": meta.get("case_number"),
                    "title": meta.get("title"),
                    "court": meta.get("court_name"),
                    "date": meta.get("date_of_judgment"),
                    "statutes": meta.get("statutes", []),
                    "summary": (meta.get("summary") or "")[:400],
                    "query_intents": set(),
                    "full_doc_options": [],
                    "chunk_options": [],
                    "rank": None,
                },
            )
            entry["query_intents"].add(intent)
            rank_val = rank_map.get(fs)
            if rank_val is not None:
                if entry["rank"] is None or rank_val < entry["rank"]:
                    entry["rank"] = rank_val

        for fs in getattr(plan, "selected_full_docs", []):
            entry = cases_map.get(fs)
            if not entry:
                continue
            if fs not in full_doc_texts:
                txt_path = Path("processed_data/txt_data") / f"{fs}.txt"
                if txt_path.exists():
                    try:
                        full_doc_texts[fs] = txt_path.read_text(encoding="utf-8")
                    except Exception:
                        full_doc_texts[fs] = ""
                else:
                    full_doc_texts[fs] = ""
            text_blob = full_doc_texts.get(fs, "")
            token_est = max(1, len(text_blob) // 4) if text_blob else 0
            reason = next(
                (
                    getattr(reason, "reason", None)
                    for reason in getattr(plan, "reasoning_full_docs", [])
                    if getattr(reason, "file_stem", None) == fs
                ),
                None,
            )
            entry["full_doc_options"].append(
                {
                    "intent": intent,
                    "estimated_tokens": token_est,
                    "planner_reason": reason,
                }
            )

        for chunk_ref in getattr(plan, "selected_chunks", []):
            fs = getattr(chunk_ref, "file_stem", None)
            ci = getattr(chunk_ref, "chunk_index", None)
            if fs is None or ci is None:
                continue
            entry = cases_map.get(fs)
            if not entry:
                continue
            doc = chunk_doc_map.get((fs, int(ci)))
            text_blob = (doc.page_content or "") if doc else ""
            chunk_texts.setdefault((fs, int(ci)), text_blob)
            token_est = max(1, len(text_blob) // 4) if text_blob else 0
            reason = next(
                (
                    getattr(reason, "reason", None)
                    for reason in getattr(plan, "reasoning_chunks", [])
                    if getattr(reason, "file_stem", None) == fs and getattr(reason, "chunk_index", None) == ci
                ),
                None,
            )
            entry["chunk_options"].append(
                {
                    "intent": intent,
                    "chunk_index": int(ci),
                    "estimated_tokens": token_est,
                    "planner_reason": reason,
                }
            )

        for meta in case_metas:
            fs = meta.get("file_stem")
            if not fs or any(fs in line for line in digest_parts):
                continue
            title = meta.get("title") or meta.get("case_number") or f"File {fs}.txt"
            court = meta.get("court_name") or ""
            date = meta.get("date_of_judgment") or ""
            summary = (meta.get("summary") or "").strip()
            if summary:
                summary = summary[:300]
            line = f"{title} ({court} {date}) – file {fs}.txt"
            if summary:
                line += f": {summary}"
            digest_parts.append(line)

    trimmed_cases: List[Dict[str, Any]] = []
    max_cases = min(
        15,
        len(cases_map),
    )
    for entry in sorted(
        cases_map.values(),
        key=lambda e: (e.get("rank") if e.get("rank") is not None else 10**9),
    ):
        entry["query_intents"] = sorted(set(entry["query_intents"]))
        full_opts = entry.get("full_doc_options", [])[:1]
        chunk_opts = entry.get("chunk_options", [])[:2]
        trimmed_cases.append(
            {
                "file_stem": entry["file_stem"],
                "rank": entry.get("rank"),
                "summary": entry.get("summary"),
                "statutes": entry.get("statutes", []),
                "query_intents": entry["query_intents"],
                "full_doc_options": full_opts,
                "chunk_options": chunk_opts,
            }
        )
        if len(trimmed_cases) >= max_cases:
            break
    selection_payload["cases"] = trimmed_cases

    research_digest = "\n".join(digest_parts[:30])
    text_store = {"full_docs": full_doc_texts, "chunks": chunk_texts}

    return selection_payload, text_store, debug_lines, research_digest

def select_draft_context(
    selection_payload: Dict[str, Any],
    debug_lines: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Use an LLM to choose the final mix of full documents and chunks."""
    lines = debug_lines if debug_lines is not None else []
    load_dotenv()
    api_key = os.getenv("OPENAI_KEY")
    if not api_key:
        lines.append("[DRAFT][select] OPENAI_KEY missing; falling back to heuristic selection")
        return _heuristic_selection(selection_payload), lines

    client = OpenAI(api_key=api_key)
    payload = json.dumps(selection_payload, ensure_ascii=False)
    lines.append(f"[DRAFT][select] payload size={len(payload)} bytes cases={len(selection_payload.get('cases', []))}")
    system_msg = """You are a senior associate preparing a Supreme Court draft. You will receive metadata about candidate cases/chunks and the token budget.
Select the best mix of authorities for drafting. Output STRICT JSON with keys: {\"selected_full_docs\", \"selected_chunks\", \"notes\", \"estimated_tokens\"}.
Rules:
- Work within target_budget_tokens (soft) and budget_tokens (hard).
- Use domain_tags (if provided) to keep focus on the petition's subject matter.
- Prefer canonical Supreme Court precedents aligned with each query intent (maintainability, statute standard, fact pattern).
- For each file, decide between full document or targeted chunks; avoid duplication.
- selected_full_docs: list of file_stem strings.
- selected_chunks: list of {\"file_stem\": str, \"chunk_indices\": [int, ...]}.
- Include brief notes summarising rationale.

Example output:
{
  \"selected_full_docs\": [\"1128\"],
  \"selected_chunks\": [{\"file_stem\": \"720\", \"chunk_indices\": [1, 4]}],
  \"estimated_tokens\": 230000,
  \"notes\": \"Maintainability anchored in 1128; PC Act trap addressed via 720 chunks.\"
}
"""
    user_msg = f"Payload:\n{payload}\n\nReturn JSON only."
    try:
        resp = client.responses.create(
            model="gpt-5-nano-2025-08-07",
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": system_msg}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_msg}]},
            ],
            text={"format": {"type": "json_object"}},
            max_output_tokens=1200,
        )
        content = _extract_resp(resp)
        selection = _ensure_json_dict(content)
        lines.append(f"[DRAFT][select] LLM selection raw tokens={len(content)}")
    except Exception as exc:
        lines.append(f"[DRAFT][select] Responses API error: {exc}")
        return {}, lines

    if not selection or not isinstance(selection, dict):
        lines.append("[DRAFT][select] invalid LLM output")
        return {}, lines

    return selection, lines


def _heuristic_selection(selection_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Fallback selection: take the lightest combination of full docs/chunks within budget."""
    budget = int(selection_payload.get("target_budget_tokens") or selection_payload.get("budget_tokens") or 300000)
    cases = selection_payload.get("cases", [])
    if not isinstance(cases, list):
        cases = []
    selected_full_docs: List[str] = []
    selected_chunks: Dict[str, List[int]] = {}
    used_tokens = 0

    for entry in cases:
        fs = entry.get("file_stem")
        if not fs:
            continue
        full_opts = entry.get("full_doc_options") or []
        full_tokens = min((opt.get("estimated_tokens", 0) for opt in full_opts), default=0)
        chunk_opts = entry.get("chunk_options") or []
        chunk_opts_sorted = sorted(chunk_opts, key=lambda opt: opt.get("estimated_tokens", 0))

        if full_tokens and used_tokens + full_tokens <= budget:
            selected_full_docs.append(fs)
            used_tokens += full_tokens
            continue

        chunk_indices: List[int] = []
        for opt in chunk_opts_sorted:
            ci = opt.get("chunk_index")
            tokens = opt.get("estimated_tokens", 0)
            if ci is None or tokens <= 0:
                continue
            if used_tokens + tokens > budget:
                break
            chunk_indices.append(int(ci))
            used_tokens += int(tokens)
        if chunk_indices:
            selected_chunks[fs] = chunk_indices

    return {
        "selected_full_docs": selected_full_docs,
        "selected_chunks": [
            {"file_stem": fs, "chunk_indices": indices}
            for fs, indices in selected_chunks.items()
        ],
        "notes": "heuristic fallback",
        "estimated_tokens": used_tokens,
    }


def assemble_selected_context(
    selection: Dict[str, Any],
    text_store: Dict[str, Dict[Any, str]],
    budget_tokens: int,
) -> Tuple[str, List[str], List[str]]:
    """Assemble final context from chosen full docs and chunks, respecting the budget."""
    debug_lines: List[str] = []
    full_docs = text_store.get("full_docs", {})
    chunks = text_store.get("chunks", {})

    total_tokens = 0
    parts: List[str] = []
    case_order: List[str] = []

    def _append_text(label: str, text: str) -> None:
        parts.append(label)
        parts.append(text)
        parts.append("\n\n---\n\n")

    for fs in selection.get("selected_full_docs", []) or []:
        text_blob = full_docs.get(fs)
        if not text_blob:
            continue
        token_est = max(1, len(text_blob) // 4)
        if total_tokens + token_est > budget_tokens:
            allowed_chars = max(0, (budget_tokens - total_tokens) * 4)
            if allowed_chars <= 0:
                debug_lines.append(f"[DRAFT][assemble] budget reached before full doc {fs}")
                break
            text_blob = text_blob[:allowed_chars]
            token_est = max(1, len(text_blob) // 4)
            debug_lines.append(f"[DRAFT][assemble] truncated {fs} to fit budget")
        _append_text(f"[Full case: {fs}.txt]", text_blob)
        case_order.append(fs)
        total_tokens += token_est

    for chunk in selection.get("selected_chunks", []) or []:
        fs = chunk.get("file_stem")
        indices = chunk.get("chunk_indices", [])
        if not fs or not indices:
            continue
        for ci in indices:
            text_blob = chunks.get((fs, int(ci)), "")
            if not text_blob:
                continue
            token_est = max(1, len(text_blob) // 4)
            if total_tokens + token_est > budget_tokens:
                allowed_chars = max(0, (budget_tokens - total_tokens) * 4)
                if allowed_chars <= 0:
                    debug_lines.append(f"[DRAFT][assemble] budget reached before chunk {fs}:{ci}")
                    break
                text_blob = text_blob[:allowed_chars]
                token_est = max(1, len(text_blob) // 4)
                debug_lines.append(f"[DRAFT][assemble] truncated chunk {fs}:{ci}")
            _append_text(f"[Case chunk: {fs}.txt | chunk {ci}]", text_blob)
            case_order.append(fs)
            total_tokens += token_est
        else:
            continue
        break

    merged_context = "".join(parts)
    merged_context = merged_context.rstrip("\n- ")
    case_order = list(dict.fromkeys(case_order))
    debug_lines.append(f"[DRAFT][assemble] final context tokens~{total_tokens}")
    return merged_context, case_order, debug_lines



def write_draft(
    profile: PetitionProfile,
    user_context: str | None,
    merged_context: str,
    case_stems: List[str],
) -> str:
    """
    Generate a structured fallback draft aligned with the required headings.
    """
    parties = profile.parties or {}
    caption = []
    if parties.get("petitioner"):
        caption.append(parties.get("petitioner"))
    if parties.get("respondent"):
        caption.append("vs " + parties.get("respondent"))
    caption_line = " ".join(part for part in caption if part).strip() or "<Add Case Caption>"

    timeline = profile.notable_facts[:6] if profile.notable_facts else ["[PLACEHOLDER: key event]"]
    issues = profile.key_issues[:3] if profile.key_issues else ["[PLACEHOLDER: Issue]"]
    citations = ", ".join(f"{fs}.txt" for fs in dict.fromkeys(case_stems)) if case_stems else "[PLACEHOLDER: authorities]"

    lines: List[str] = []
    lines.append("### 1. Title / Caption")
    lines.append("Written Submissions on behalf of the Respondent (Fallback)")
    lines.append(caption_line + "\n")

    lines.append("### 2. List of Dates (compact)")
    for item in timeline:
        lines.append(f"- {item}")

    lines.append("\n### 3. Questions for Consideration")
    for idx, issue in enumerate(issues, start=1):
        lines.append(f"Q{idx}. {issue}")

    lines.append("\n### 4. Submissions")
    lines.append("(A) Issue 1 – [PLACEHOLDER]")
    lines.append("1. Rule: cite statute/case [PLACEHOLDER]")
    lines.append("2. Application: apply to facts [PLACEHOLDER]")
    lines.append("3. Conclusion: [PLACEHOLDER]")
    lines.append("(B) Issue 2 – [PLACEHOLDER]")
    lines.append("1. Rule: [PLACEHOLDER]")
    lines.append("2. Application: [PLACEHOLDER]")
    lines.append("3. Conclusion: [PLACEHOLDER]")
    lines.append("(C) Maintainability")
    lines.append("1. Alternate remedy / laches / suppression [PLACEHOLDER]")

    lines.append("\n### 5. Prayer")
    lines.append("- [PLACEHOLDER: relief sought]")

    lines.append("\n### 6. List of Authorities")
    lines.append(f"- {citations}")

    lines.append("\n### 7. Verification & Signature")
    lines.append("Verification: [PLACEHOLDER]")
    lines.append("Counsel: [PLACEHOLDER]")

    lines.append("\n### 8. Annexure References (if any)")
    lines.append("- [PLACEHOLDER]")

    return "\n".join(lines).strip()


def write_draft_llm(
    profile: PetitionProfile,
    user_context: Optional[str],
    merged_context: str,
    case_stems: List[str],
    research_digest: str,
    petition_text: str,
    max_completion_tokens: Optional[int] = None,
) -> Tuple[str, List[str]]:
    """
    LLM-backed draft writer using OpenAI (gpt-5-nano). Falls back to template
    if no API key set. Uses only provided context for authorities.
    """

    load_dotenv()
    api_key = os.getenv("OPENAI_KEY")
    debug_lines: List[str] = []
    if not api_key:
        debug_lines.append("[DRAFT] OPENAI_KEY missing; using template draft")
        return write_draft(profile, user_context, merged_context, case_stems), debug_lines
    client = OpenAI(api_key=api_key)

    model_name = "gpt-5-nano-2025-08-07"
    if max_completion_tokens is None:
        max_completion_tokens = drafting_completion_tokens(model_name)
    context_budget_tokens = drafting_context_budget_tokens(model_name)
    char_limit = context_budget_tokens * 4
    ctxt = (merged_context or "").strip()
    if len(ctxt) > char_limit:
        ctxt = ctxt[:char_limit]
        debug_lines.append(f"[DRAFT] trimmed context to {char_limit} chars for writer")
    debug_lines.append(f"[DRAFT] writer context tokens~{len(ctxt)//4} (limit={context_budget_tokens})")

    sys = """You are a senior advocate drafting FINAL WRITTEN SUBMISSIONS for the Respondent before an Indian court. Produce a filing-ready brief that is persuasive, precise, and registry-compliant.\n\nMANDATORY RULES\n- Do NOT invent facts. Use only the Petition Profile, Client Context, Petition Excerpt, and Research Extracts provided. If a fact is missing, write [PLACEHOLDER: …].\n- Choose ONE procedural posture and be consistent: Writ under Article 32 OR SLP under Article 136. Never mix both.\n- Never confuse party roles. The Respondent is the State/agency/defendant as per “opponent_type”.\n- Prefer binding Supreme Court authorities (SCC / SCC OnLine / INSC). Use High Court only if indispensable and label it clearly.\n- Always include at least: (a) 2 binding SC precedents, (b) 1 persuasive precedent, and (c) the key statutory subsection(s) quoted verbatim where central.\n- Human-readable citations FIRST, retrieval IDs in brackets SECOND, e.g. State of Punjab v. Hari Kesh, 2025 INSC 50 [ref: 355.json].\n- Keep it assertive (“It is respectfully submitted…”), not academic. No long block quotes; paraphrase with pinpoint references.\n- Number paragraphs. Target 900–1400 words unless directed otherwise. Remove repetition.\n\nHEADINGS (use exactly these)\n### 1. Title / Caption\n### 2. List of Dates (compact)\n### 3. Questions for Consideration\n### 4. Submissions\n### 5. Prayer\n### 6. List of Authorities\n### 7. Verification & Signature\n### 8. Annexure References (if any)\n\nSTYLE & QUALITY CHECKS\n- Be concrete: refer to sanction file, note-sheet, FIR No. with [PLACEHOLDER] if unknown.\n- Use decisive phrases: “no failure of justice”, “presumption of regularity”, “prima facie”.\n- If an extract is irrelevant, ignore it. Do not hedge or speculate. Eliminate redundancy."""

    profile_lines = [
        f"document_type: {profile.document_type}",
        f"key_issues: {', '.join(profile.key_issues) if profile.key_issues else ''}",
        f"statutes: {', '.join(profile.statutes) if profile.statutes else ''}",
        f"requested_relief: {profile.requested_relief}",
        f"opponent_type: {profile.opponent_type}",
    ]
    optional_lines = []
    if user_context:
        optional_lines.append(f"user_context: {user_context.strip()}")
    if profile.domain_tags:
        optional_lines.append(f"domain_tags: {', '.join(profile.domain_tags)}")
    prof_block = "\n".join(profile_lines + optional_lines)

    petition_excerpt = (petition_text or "").strip()
    if len(petition_excerpt) > 4000:
        petition_excerpt = petition_excerpt[:4000]

    prompt = (
        f"PETITION PROFILE:\n{prof_block}\n\n"
        f"CLIENT CONTEXT:\n{(user_context or '').strip() or 'None provided.'}\n\n"
        "PETITION EXCERPT (use for factual narration; do not invent beyond this):\n"
        f"{petition_excerpt or 'Not supplied.'}\n\n"
        "CASE DIGEST (summaries of retrieved authorities – use the principles as needed):\n"
        f"{research_digest or 'None available.'}\n\n"
        "ADDITIONAL RESEARCH EXTRACTS (use only if they add value; never quote at length):\n"
        f"{ctxt}\n\n"
        "REQUIRED POSTURE CHOICE: Choose exactly one of ['Art. 32 Writ','Art. 136 SLP'] based on the PETITION EXCERPT and any prior High Court posture. If an HC order is under challenge, use Art. 136 SLP.\n"
        "REQUIRED AUTHORITY MIX: Include ≥2 binding Supreme Court precedents, ≥1 persuasive authority, and quote the key statutory subsection(s) once.\n"
        "CITATION FORMAT: Case Name, YEAR reporter page/ID [ref: file_stem.txt].\n\n"
        "QUALITY GATES (must pass before finalising):\n"
        "[ ] No party-role confusion.\n"
        "[ ] Exactly one procedural posture (Art. 32 OR Art. 136).\n"
        "[ ] ≥2 Supreme Court binding authorities + ≥1 persuasive + statute subsection quoted.\n"
        "[ ] Paragraphs numbered; 900–1400 words; no findings/judgment tone.\n"
        "[ ] Sections 5–7 present (Prayer, List of Authorities, Verification & Signature). Annexure References if applicable.\n\n"
        "Draft the Respondent's Written Submissions now following the mandated headings."
    )

    out = ""
    preview = ""
    try:
        resp = client.responses.create(
            model="gpt-5-nano-2025-08-07",
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": sys}]},
                {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
            ],
            max_output_tokens=max_completion_tokens,
        )
        out = _extract_resp(resp)
        preview = getattr(resp, "output_text", "") or out
        debug_lines.append(f"[DRAFT] Responses API output chars={len(preview)}")
    except Exception as exc:
        debug_lines.append(f"[DRAFT] Responses API error: {exc}")

    if not out:
        try:
            cc = client.chat.completions.create(
                model="gpt-5-nano-2025-08-07",
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=max_completion_tokens,
            )
            out = (cc.choices[0].message.content or "")
            debug_lines.append(f"[DRAFT] Chat Completions output chars={len(out)}")
        except Exception as exc:
            debug_lines.append(f"[DRAFT] Chat Completions error: {exc}")

    if not out and research_digest:
        debug_lines.append("[DRAFT] retrying writer with digest-only prompt")
        digest_prompt = (
            f"PETITION PROFILE:\n{prof_block}\n\n"
            f"CLIENT CONTEXT:\n{(user_context or '').strip() or 'None provided.'}\n\n"
            f"PETITION EXCERPT:\n{petition_excerpt or 'Not supplied.'}\n\n"
            f"CASE DIGEST:\n{research_digest or 'None available.'}\n\n"
            "Using the mandated headings, prepare the respondent's draft. Focus on maintainability and merits; cite cases succinctly."
        )
        try:
            resp2 = client.responses.create(
                model="gpt-5-nano-2025-08-07",
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": sys}]},
                    {"role": "user", "content": [{"type": "input_text", "text": digest_prompt}]},
                ],
                max_output_tokens=max_completion_tokens,
            )
            out = _extract_resp(resp2)
            digest_preview = getattr(resp2, "output_text", "") or out
            debug_lines.append(f"[DRAFT] digest retry output chars={len(digest_preview)}")
        except Exception as exc:
            debug_lines.append(f"[DRAFT] digest retry error: {exc}")

    # Ensure citations mention retrieved stems (if not already)
    if out and "### 8. Citations / References" in out and case_stems:
        if not any(fs in out for fs in case_stems):
            src_line = ", ".join(f"{fs}.txt" for fs in dict.fromkeys(case_stems))
            out = out.rstrip() + f"\n- {src_line}\n"
            debug_lines.append("[DRAFT] appended citations line from case stems")
    content = out.strip()
    if not content:
        debug_lines.append("[DRAFT] empty draft from LLM; using fallback template")
        return write_draft(profile, user_context, ctxt, case_stems), debug_lines
    debug_lines.append(f"[DRAFT] draft length={len(content)} chars")
    return content, debug_lines
