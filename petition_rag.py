"""
Petition-specific RAG helpers: filtration and drafting assembler.
Keeps petition prompts isolated from chat to avoid prompt pollution.
"""

import os
import json
from typing import List, Optional, Dict, Any, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from pydantic import BaseModel

from rag import FiltrationPlan, ChunkRef, ReasonFullDoc, ReasonChunk
from time_utils import now_ist_stamp


def _filtration_preview(stage: str, detail: str) -> None:
    """Emit a diagnostic line for petition filtration auth/state."""
    try:
        clean = " ".join(str(detail).split())
    except Exception:
        clean = str(detail)
    try:
        print(f"{now_ist_stamp()} [DRAFT-FILTRATION][{(stage or 'stage').upper()}] {clean}", flush=True)
    except Exception:
        pass


def _build_chunk_previews(docs: List[Document], max_chars: int = 800) -> List[dict]:
    previews = []
    for d in docs:
        m = d.metadata or {}
        previews.append({
            "file_stem": m.get("file_stem"),
            "chunk_index": m.get("chunk_index", -1),
            "court_name": m.get("court_name"),
            "case_number": m.get("case_number"),
            "date_of_judgment": m.get("date_of_judgment"),
            "legal_provisions_cited": m.get("legal_provisions_cited", []),
            "preview": (d.page_content or "").strip().replace("\n", " ")[:max_chars],
        })
    return previews


def petition_filtration_retriever(
    petition_text: str,
    fighting_points: List[str],
    user_q: str,
    docs: List[Document],
    mode: str = "chunk",
    case_metadatas: Optional[List[dict]] = None,
    desired_min_full_docs: int = 3,
    desired_max_chunks_per_case: int = 2,
    query_context: Optional[dict] = None,
    llm=None,
    model_name: Optional[str] = None,
    provider_name: Optional[str] = None,
) -> FiltrationPlan:
    """
    Petition-aware filtration: select cases/chunks most useful to rebut the petition.
    Returns a FiltrationPlan using a dedicated system prompt.
    """
    if not docs:
        return FiltrationPlan(selected_full_docs=[], selected_chunks=[], drop_chunks=[], context_budget_tokens=6000)

    if llm is None:
        _filtration_preview("llm_missing", "No LLM provided; falling back to heuristic")

    previews = _build_chunk_previews(docs) if mode == "chunk" else []
    from models import get_prompt
    system_msg = get_prompt("petition_filtration")
    _filtration_preview("llm_info", f"model={model_name} provider={provider_name}")

    effective_qc = dict(query_context or {})
    effective_qc.setdefault("retrieval_k", len(docs))

    user_payload = {
        "petition": (petition_text or "")[:8000],
        "fighting_points": fighting_points or [],
        "question": user_q,
        "mode": mode,
        "chunks": previews,
        "cases": (case_metadatas or []),
        "query_context": effective_qc,
        "preferences": {
            "desired_min_full_docs": max(1, int(desired_min_full_docs or 1)),
            "desired_max_chunks_per_case": max(1, int(desired_max_chunks_per_case or 1)),
        },
        "schema": {
            "selected_full_docs": ["<file_stem>"],
            "selected_chunks": [{"file_stem": "<file_stem>", "chunk_index": 0}],
            "drop_chunks": [{"file_stem": "<file_stem>", "chunk_index": 0}],
            "reasoning_full_docs": [{"file_stem": "<file_stem>", "reason": "why this case is needed"}],
            "reasoning_chunks": [{"file_stem": "<file_stem>", "chunk_index": 0, "reason": "why this chunk"}],
            "overall_reasoning": "optional global rationale",
            "notes": "optional short note",
            "request_more_cases": False,
            "expansion_reason": "optional short reason",
        },
    }

    try:
        prompt = (
            f"{system_msg}\n\nReturn ONLY JSON per the schema. Do not wrap in code fences.\nPAYLOAD:\n"
            f"{json.dumps(user_payload)}"
        )
        resp = llm.invoke(prompt)
        content = getattr(resp, "content", None) or str(resp)
        if isinstance(content, list):
            content = " ".join(str(x) for x in content)
        parsed = None
        try:
            parsed = json.loads(content)
        except Exception:
            import re as _re
            m = _re.search(r"\{.*\}", content, flags=_re.DOTALL)
            if m:
                parsed = json.loads(m.group(0))
        if parsed:
            plan = FiltrationPlan(**parsed)
            if not plan.reasoning_full_docs and plan.selected_full_docs:
                plan.reasoning_full_docs = [ReasonFullDoc(file_stem=fs, reason="supports rebuttal points") for fs in plan.selected_full_docs]
            if not plan.reasoning_chunks and plan.selected_chunks:
                plan.reasoning_chunks = [ReasonChunk(file_stem=c.file_stem, chunk_index=c.chunk_index, reason="addresses fighting_points") for c in plan.selected_chunks]
            if not plan.overall_reasoning:
                plan.overall_reasoning = "Selected items directly support rebuttal issues raised by the petition."
            _filtration_preview(
                "success",
                f"full={len(plan.selected_full_docs)} chunk={len(plan.selected_chunks)} reasoning={'yes' if plan.overall_reasoning else 'no'}",
            )
            return plan
    except Exception as exc:
        _filtration_preview("llm_error", str(exc))
        # Simple heuristic fallback
    from collections import Counter
    stems = [d.metadata.get("file_stem") for d in docs if d.metadata.get("file_stem")]
    sel_full: List[str] = []
    sel_chunks: List[ChunkRef] = []
    if stems:
        dominant, _ = Counter(stems).most_common(1)[0]
        sel_full = [dominant]
    for d in docs[:2]:
        fs = d.metadata.get("file_stem")
        ci = d.metadata.get("chunk_index", 0)
        if fs is not None:
            sel_chunks.append(ChunkRef(file_stem=fs, chunk_index=ci))
    return FiltrationPlan(
        selected_full_docs=sel_full,
        selected_chunks=sel_chunks,
        drop_chunks=[],
        context_budget_tokens=0,
        request_more_cases=False,
    )


def assemble_draft_reply(
    petition_text: str,
    fighting_points: List[str],
    context: str,
    llm,
) -> str:
    """
    Produce a structured rebuttal draft using ONLY the provided context.
    """
    load_dotenv()

    system_msg = (
        "You are a senior advocate drafting FINAL WRITTEN SUBMISSIONS for the Respondent before an Indian court. "
        "Produce a filing-ready brief that is persuasive, precise, and registry-compliant.\n\n"
        "MANDATORY RULES\n"
        "- Do NOT invent facts. Use only the Petition text, Fighting Points, and Research Context provided. If a fact is missing, write [PLACEHOLDER: …].\n"
        "- Choose ONE procedural posture and be consistent: Art. 32 Writ OR Art. 136 SLP. Never mix both.\n"
        "- Never confuse party roles; maintain Respondent posture.\n"
        "- Prefer binding Supreme Court authorities; use High Courts only if indispensable and label them clearly.\n"
        "- Include at minimum: (a) two binding Supreme Court precedents, (b) one additional persuasive authority, and (c) quote key statutory subsection(s) once where central (e.g., Section 19(3) PC Act).\n"
        "- Human-readable citations FIRST, retrieval anchors SECOND in brackets, e.g., Case Name, 2025 INSC 50 [ref: 355.txt].\n"
        "- Assertive drafting (“It is respectfully submitted that …”); avoid long block quotes; paraphrase with pinpoint references.\n"
        "- Number paragraphs; eliminate redundancy.\n\n"
        # "HEADINGS (use exactly these)\n"
        # "### 1. Title / Caption\n"
        # "### 2. List of Dates (compact)\n"
        # "### 3. Questions for Consideration\n"
        # "### 4. Submissions\n"
        # "### 5. Prayer\n"
        # "### 6. List of Authorities\n"
        # "### 7. Verification & Signature\n"
        # "### 8. Annexure References (if any)\n\n"
        "STYLE & QUALITY CHECKS\n"
        "- Be concrete: refer to the sanction file, note-sheet, FIR No., Crl.M.P. No., etc., using [PLACEHOLDER] if unknown.\n"
        "- Use decisive phrases when warranted (e.g., ‘no failure of justice’, ‘presumption of regularity’, ‘prima facie’).\n"
        "- If an extract is irrelevant, ignore it. Do not speculate.\n"
    )

    prompt = {
        "petition": petition_text or "",
        "fighting_points": fighting_points or [],
        "context": context or "",
        "instructions": "Draft the full Respondent’s Written Submissions now, following the mandated headings and rules."
    }

    out = llm.invoke([
        ("system", system_msg),
        ("user", json.dumps(prompt)),
    ])
    text = getattr(out, "content", None) or ""
    return text.strip()

