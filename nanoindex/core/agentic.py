"""Agentic retrieval: multi-round document exploration with LLM-driven navigation.

The agent iteratively:
  1. Examines the document tree outline (as JSON)
  2. Thinks about which sections are relevant, then selects them
  3. Reviews their content (text + optional page images)
  4. Requests more sections OR signals it has enough to answer

Retrieval and answer generation are separated: the agent loop only
gathers context, then a clean answer-generation step produces the final
response.  Typically completes in 2-3 retrieval rounds.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from nanoindex.config import NanoIndexConfig
from nanoindex.core.llm import LLMClient
from nanoindex.models import (
    Answer, BoundingBox, Citation, DocumentTree,
    PageDimensions, RetrievedNode, TreeNode,
)
from nanoindex.utils.tree_ops import (
    collect_text, find_node, find_siblings, iter_nodes, tree_to_json_outline,
)

logger = logging.getLogger(__name__)

_MAX_ROUNDS = 5
_MAX_PAGES_PER_ROUND = 20
_MAX_TOTAL_PAGES = 40
_SMALL_TREE_THRESHOLD = 25  # Skip tree nav for documents with fewer nodes

_REFUSAL_PATTERNS = re.compile(
    r"unable to identify|cannot determine|cannot answer|"
    r"does not contain|not contain a|no relevant|"
    r"information is not available|could not find|"
    r"not included in the provided|not present in",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Prompts — all user-role only, no system prompt (following PageIndex pattern)
# ---------------------------------------------------------------------------

_DECOMPOSE_FINANCIAL = """\
You are a financial analyst. Given a question about a financial document, \
identify exactly what data points are needed to answer it.

Question: {query}

Reply in JSON:
{{
    "data_points": ["<specific metric/number needed>", ...],
    "statements_needed": ["<financial statement or section type>", ...]
}}

Examples of data_points: "net income FY2022", "total assets end of FY2022", \
"operating cash flow FY2021", "revenue FY2017", "capital expenditures FY2019".
Examples of statements_needed: "income statement", "balance sheet", \
"cash flow statement", "segment data", "MD&A", "notes to financial statements".

Be exhaustive — if a ratio requires two inputs, list BOTH. \
If multi-year data is needed, list EACH year separately.
Directly return the JSON. Do not output anything else."""

_DECOMPOSE_GENERAL = """\
You are a document analysis expert. Given a question about a document, \
identify exactly what information is needed to answer it.

Question: {query}

Reply in JSON:
{{
    "data_points": ["<specific fact or data point needed>", ...],
    "statements_needed": ["<document section or type of content>", ...]
}}

Be exhaustive — if the answer requires combining multiple pieces of \
information, list ALL of them.
Directly return the JSON. Do not output anything else."""

_ROUND1_FINANCIAL = """\
You are given a question and a tree structure of a document.
Each node contains a node_id, title, and a corresponding summary.
Your task is to find all nodes that are likely to contain the data needed to \
answer the question.

{decomposition}

IMPORTANT — tips for financial documents (10-K, 10-Q, 8-K, earnings):
- Income Statement / Revenue / Operating Income → look in Consolidated \
Statements of Income or Operations
- Balance Sheet / Assets / Liabilities / Working Capital → look in \
Consolidated Balance Sheets
- Cash Flow / CapEx / FCF → look in Consolidated Statements of Cash Flows
- Restructuring / Acquisitions / Segment detail → look in Notes to \
Financial Statements (often deep sub-nodes)
- Ratios often require data from MULTIPLE statements — select all of them
- Short documents (8-K, earnings) may have flat structures; select broadly

Question: {query}

Document tree structure:
{outline}

Please reply in the following JSON format:
{{
    "thinking": "<Your reasoning about which nodes contain the answer>",
    "node_list": ["node_id_1", "node_id_2", ...]
}}
Directly return the final JSON structure. Do not output anything else."""

_ROUND1_GENERAL = """\
You are given a question and a tree structure of a document.
Each node contains a node_id, title, and a corresponding summary.
Your task is to find all nodes that are likely to contain the information \
needed to answer the question.

{decomposition}

Tips:
- Select broadly — it's better to include extra sections than miss relevant ones.
- Look at both top-level sections and their children for specific details.
- If the question requires combining information from multiple parts, select all.

Question: {query}

Document tree structure:
{outline}

Please reply in the following JSON format:
{{
    "thinking": "<Your reasoning about which nodes contain the answer>",
    "node_list": ["node_id_1", "node_id_2", ...]
}}
Directly return the final JSON structure. Do not output anything else."""

_REVIEW = """\
Here is the content from the sections you requested:

{content}

Question: {query}

Review the information carefully. Do you have ALL the data needed to \
answer the question precisely?

{remaining}

If you need data from additional sections, reply:
{{
    "thinking": "<Why you need more data>",
    "action": "select_more",
    "node_list": ["node_id_1", ...]
}}

If you have everything needed, reply:
{{
    "thinking": "<Confirm you have sufficient data>",
    "action": "done"
}}
Directly return the final JSON structure. Do not output anything else."""

_SUFFICIENCY = """\
Before finalizing, cross-check the retrieved content against these \
data requirements:

{checklist}

Question: {query}

Sections retrieved so far:
{retrieved_titles}

Is ANY required data point still missing from the retrieved sections? \
If yes, identify which sections from the unread list might contain it.

{remaining}

Reply in JSON:
{{
    "thinking": "<Check each data point against what you have>",
    "action": "select_more",
    "node_list": ["node_id_1", ...]
}}

If everything is covered, reply:
{{
    "thinking": "<Confirm all data points are present>",
    "action": "done"
}}
Directly return the JSON. Do not output anything else."""

_ANSWER_GENERAL = """\
Answer the question based on the context below. \
Show your reasoning step by step when the question involves \
numbers, comparisons, or multi-part analysis.

TEMPORAL CONTEXT: {temporal_context}

Question: {query}

Context:
{context}

{kb_reference}

RULES:
1. Answer ONLY from information in the context. Do not use outside knowledge.
2. If specific data is available in the context, provide a precise answer \
with exact values. Do not say "cannot be determined" if the data is there.
3. For comparative questions, list all candidates with their values before \
identifying the answer.
4. Quote or reference specific sections when possible.

Provide a clear, specific answer. Never refuse to answer if the data \
is available in the context."""

_ANSWER = """\
Answer the question based on the context below. \
Show your calculations and reasoning step by step when the question \
involves numbers, ratios, or comparisons.

TEMPORAL CONTEXT: {temporal_context}

Question: {query}

Context:
{context}

{kb_reference}

CRITICAL RULES:
1. If a FINANCIAL REFERENCE block is provided above, you MUST use \
the EXACT formula and conventions specified there. Do not invent \
alternative formulas. When the reference says "PREFERRED", use that \
formula unless the question explicitly requests an alternative.
2. If the question asks for a ratio, margin, or derived metric and the \
component data is in the context, you MUST compute the answer. \
Do NOT say "cannot be calculated" if inputs exist.
3. If a calculation requires data from multiple statements (e.g., \
income statement + balance sheet), use values from the SAME fiscal year.
4. Answer from the perspective of the TEMPORAL CONTEXT date above. \
Events described as planned or expected in the filing should be treated \
as future events, even if you know they have since occurred. Do NOT \
use knowledge of events after the document date.
5. If the question asks what is "directly outlined" in a financial \
statement, also check the Notes to Financial Statements for line items \
that are disclosed there rather than on the face of the statement.
6. FISCAL YEAR NAMING: Companies may have non-calendar fiscal years. \
When a filing says "FY2023" or "Fiscal 2023", use the company's own \
definition. If the filing is dated early 2023 and covers a fiscal year \
ending in January 2023, that IS the company's "FY2023" (or "Fiscal 2022" \
depending on their convention). Always match the company's own labels \
for fiscal years — do NOT override with calendar year assumptions.
7. For COMPARATIVE questions ("best performing", "highest", "largest", \
"most improved", etc.), you MUST list ALL candidates with their values, \
then identify the winner. CRITICAL DISTINCTION: \
- "performed the best" / "best performance" = highest GROWTH RATE (%). \
- "largest" / "highest revenue" / "biggest" = highest ABSOLUTE value. \
When the question says "performed the best (by top line)", "best" \
modifies performance which means GROWTH, not absolute size. You MUST \
compute year-over-year percentage change for each candidate and pick \
the one with the highest growth rate. Do NOT pick the largest absolute \
revenue category — that answers "which is largest" not "which performed best".
8. When computing ratios (ROA, ROE, asset turnover, etc.), use \
END-OF-PERIOD values from the CURRENT year's balance sheet ONLY. \
Do NOT compute averages of beginning and ending balances unless the \
question EXPLICITLY says "average". If the question references \
"the statement of financial position", that means ONE balance sheet \
date — use that single period's values.
9. For companies with NON-CONTROLLING INTERESTS: ALWAYS use \
"Net income attributable to [Company name]" or "Net income attributable \
to common shareholders" — NEVER use total "Net income" that includes \
non-controlling interests. This applies to ALL ratios: ROA, ROE, EPS, \
net margin, etc.
10. For ACCELERATION / DECELERATION questions: ANY difference matters, \
even 0.1 percentage points. If growth goes from 3.6% to 3.5%, that IS \
deceleration. Give a DEFINITIVE answer ("No, it will decelerate" or \
"Yes, it will accelerate"). Do NOT hedge with "it depends on the metric" \
or present multiple interpretations — pick the most standard metric \
(headline adjusted EPS) and give a clear yes/no.
11. When COUNTING stores, locations, branches, or facilities: include \
ALL formats and segments (domestic + international, all store brands/formats) \
unless the question explicitly specifies one format. For example, if a \
retailer has "Best Buy stores", "Outlet stores", and "Pacific Sales stores", \
the total store count = sum of ALL formats. Always report the TOTAL count \
across all formats, not just the primary brand.
12. For YES/NO and THRESHOLD questions (e.g., "is the quick ratio \
healthy?", "is this capital-intensive?", "did it accelerate?"): give \
ONE clear definitive answer. If a ratio is below a threshold (e.g., \
quick ratio < 1.0), state it clearly as NOT healthy — do not then \
argue it's actually fine because of other factors. The question asks \
about the specific metric, not a holistic assessment. Do NOT say \
"it depends" or "viewed in isolation" for threshold-based questions.
13. For WORKING CAPITAL of fintech/payment companies (PayPal, Square, \
Stripe, etc.): customer funds held and corresponding customer payables \
are PASS-THROUGH items that inflate both sides of the balance sheet. \
When computing working capital, EXCLUDE customer funds/payables from \
both current assets and current liabilities to get the meaningful \
operating working capital figure.

Provide a clear, specific answer. Never refuse to answer if the data \
is available in the context."""

_VERIFY = """\
You are a calculation auditor. Review this answer for numerical accuracy.

Question: {query}
Answer: {answer}

Source data (for reference):
{context}

Instructions:
1. Identify every numerical claim or calculation in the answer.
2. For each, re-derive the result step by step using the source data.
3. If ALL calculations are correct, return the original answer unchanged.
4. If ANY calculation is wrong, return a CORRECTED answer with the right numbers.

Return ONLY the final answer (corrected if needed, or original if correct). \
Do not include meta-commentary about the verification process."""

_HAS_NUMBERS = re.compile(r"\d+[\d,]*\.?\d*\s*[%$BMKbmk]|\$\s*\d|ratio|margin|ROA|ROE|EPS", re.IGNORECASE)

# Financial document indicators
_FINANCIAL_KEYWORDS = {
    "10-k", "10-q", "8-k", "10k", "10q", "8k", "sec filing",
    "income statement", "balance sheet", "cash flow", "revenue",
    "earnings", "fiscal", "eps", "ebitda", "net income", "operating income",
    "gross margin", "roe", "roa", "working capital", "dividend",
    "shareholders", "consolidated statements",
}


def _is_financial_doc(doc_name: str, query: str = "") -> bool:
    """Detect if the document/query is financial in nature."""
    combined = (doc_name + " " + query).lower()
    matches = sum(1 for kw in _FINANCIAL_KEYWORDS if kw in combined)
    return matches >= 2 or any(
        tag in doc_name.upper() for tag in ("10K", "10Q", "8K", "10-K", "10-Q", "EARNINGS")
    )


# ------------------------------------------------------------------
# Query decomposition
# ------------------------------------------------------------------

async def _decompose_query(query: str, llm: LLMClient, *, financial: bool = True) -> dict:
    """Break the query into required data points and statements."""
    template = _DECOMPOSE_FINANCIAL if financial else _DECOMPOSE_GENERAL
    messages = [{"role": "user", "content": template.format(query=query)}]
    try:
        resp = await llm.chat(messages, temperature=0.0, max_tokens=512)
        data = _parse_agent_response(resp)
        if "data_points" in data:
            return data
    except Exception:
        logger.warning("Query decomposition failed, proceeding without it", exc_info=True)
    return {}


def _format_decomposition(decomp: dict) -> str:
    """Format decomposition result for injection into the Round 1 prompt."""
    if not decomp:
        return ""
    parts = ["DATA REQUIREMENTS for this question:"]
    dps = decomp.get("data_points", [])
    if dps:
        parts.append("  Data points needed: " + "; ".join(dps))
    stmts = decomp.get("statements_needed", [])
    if stmts:
        parts.append("  Look in: " + ", ".join(stmts))
    parts.append("Make sure to select nodes covering ALL of the above.")
    return "\n".join(parts)


# ------------------------------------------------------------------
# JSON parsing
# ------------------------------------------------------------------

def _parse_agent_response(text: str) -> dict:
    """Extract a JSON object from the LLM response, robustly."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Brace-matching for the outermost JSON object
    start = text.find("{")
    if start != -1:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"' and not escape:
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break

    # Regex fallback — extract key fields
    action_m = re.search(r'"action"\s*:\s*"([^"]+)"', text)
    thinking_m = re.search(r'"thinking"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    ids_m = re.search(r'"node_list"\s*:\s*\[([^\]]*)\]', text)
    if not ids_m:
        ids_m = re.search(r'"node_ids"\s*:\s*\[([^\]]*)\]', text)

    result: dict = {}
    if action_m:
        result["action"] = action_m.group(1)
    if thinking_m:
        result["thinking"] = thinking_m.group(1)
    if ids_m:
        result["node_list"] = re.findall(r'"([^"]+)"', ids_m.group(1))
    if result:
        return result

    return {"action": "error", "raw": text[:500]}


def _parse_node_ids(data: dict) -> list[str]:
    ids = data.get("node_list") or data.get("node_ids") or []
    return [str(x) for x in ids]


# ------------------------------------------------------------------
# Content building helpers
# ------------------------------------------------------------------

def _build_section_text(nodes: list[RetrievedNode]) -> str:
    parts: list[str] = []
    for rn in nodes:
        header = f"--- Section: {rn.node.title} [{rn.node.node_id}]"
        if rn.node.start_index:
            header += f" (pp. {rn.node.start_index}-{rn.node.end_index})"
        header += " ---"
        text = rn.text or "(no text available)"
        parts.append(f"{header}\n{text}")
    return "\n\n".join(parts)


def _collect_page_numbers(nodes: list[RetrievedNode], limit: int = _MAX_PAGES_PER_ROUND) -> list[int]:
    pages: list[int] = []
    seen: set[int] = set()
    for rn in nodes:
        if rn.node.start_index:
            for p in range(rn.node.start_index, rn.node.end_index + 1):
                if p not in seen:
                    pages.append(p)
                    seen.add(p)
    return sorted(pages)[:limit]


def _remaining_outline_json(tree: DocumentTree, read_ids: set[str]) -> str:
    """Render unread nodes as a flat JSON list for the review prompt."""
    items: list[dict] = []
    for node in iter_nodes(tree.structure):
        if node.node_id not in read_ids:
            d: dict = {"node_id": node.node_id, "title": node.title}
            if node.summary:
                d["summary"] = node.summary
            if node.start_index:
                d["start_page"] = node.start_index
                d["end_page"] = node.end_index
            items.append(d)
    if not items:
        return ""
    return f"Sections you haven't read yet:\n{json.dumps(items, indent=2)}"


# ------------------------------------------------------------------
# Node resolution
# ------------------------------------------------------------------

def _resolve_node(
    structure: list[TreeNode],
    nid: str,
    title_hint: str | None = None,
) -> TreeNode | None:
    """Resolve a node ID, walking up the dot-separated hierarchy.

    If the ID is completely unresolvable and a *title_hint* is given,
    falls back to fuzzy title matching across all nodes.
    """
    candidate = nid
    while candidate:
        node = find_node(structure, candidate)
        if node is not None:
            return node
        if "." not in candidate:
            break
        candidate = candidate.rsplit(".", 1)[0]

    if title_hint:
        return _fuzzy_find_by_title(structure, title_hint)
    return None


def _fuzzy_find_by_title(
    structure: list[TreeNode],
    title: str,
    threshold: float = 0.6,
) -> TreeNode | None:
    """Find the node whose title best matches *title* (above *threshold*)."""
    from difflib import SequenceMatcher

    best_node: TreeNode | None = None
    best_score = threshold
    title_lower = title.lower().strip()

    for node in iter_nodes(structure):
        score = SequenceMatcher(None, title_lower, node.title.lower().strip()).ratio()
        if score > best_score:
            best_score = score
            best_node = node

    if best_node:
        logger.info(
            "Fuzzy title match: '%s' → '%s' [%s] (score=%.2f)",
            title, best_node.title, best_node.node_id, best_score,
        )
    return best_node


_MAX_SIBLINGS = 2


def _resolve_nodes(
    structure: list[TreeNode],
    node_ids: list[str],
    seen: set[str],
    thinking: str = "",
) -> list[RetrievedNode]:
    results: list[RetrievedNode] = []
    for nid in node_ids:
        node = _resolve_node(structure, nid, title_hint=None)
        if node is None and thinking:
            node = _resolve_node(structure, nid, title_hint=thinking)
        if node is None:
            logger.warning("Agent selected non-existent node '%s'", nid)
            continue
        if node.node_id in seen:
            continue
        seen.add(node.node_id)
        text = node.text or collect_text(node)
        results.append(RetrievedNode(node=node, text=text))

        if not node.nodes:
            siblings = find_siblings(structure, node.node_id, max_each_side=_MAX_SIBLINGS)
            for sib in siblings:
                if sib.node_id in seen:
                    continue
                seen.add(sib.node_id)
                sib_text = sib.text or collect_text(sib)
                results.append(RetrievedNode(node=sib, text=sib_text))
            if siblings:
                logger.info(
                    "Auto-expanded node '%s' with %d siblings: %s",
                    nid, len(siblings), [s.node_id for s in siblings],
                )

    return results


# ------------------------------------------------------------------
# Agent retrieval loop (retrieval only — no answering)
# ------------------------------------------------------------------

async def _run_retrieval(
    query: str,
    tree: DocumentTree,
    llm: LLMClient,
    *,
    pdf_path: str | Path | None = None,
    use_vision: bool = False,
    max_rounds: int = _MAX_ROUNDS,
    decomposition: dict | None = None,
    financial: bool = True,
) -> tuple[list[RetrievedNode], list[int]]:
    """Multi-round retrieval.  Returns (retrieved_nodes, page_numbers)."""

    outline = tree_to_json_outline(tree.structure)
    all_retrieved: list[RetrievedNode] = []
    seen_ids: set[str] = set()
    all_page_numbers: list[int] = []
    conversation: list[dict[str, Any]] = []

    decomp_text = _format_decomposition(decomposition or {})

    # ---- Round 1: show outline, ask for selection ----
    round1_template = _ROUND1_FINANCIAL if financial else _ROUND1_GENERAL
    round1_msg = round1_template.format(query=query, outline=outline, decomposition=decomp_text)
    conversation.append({"role": "user", "content": round1_msg})

    resp_text = await llm.chat(conversation, temperature=0.0, max_tokens=1024)
    conversation.append({"role": "assistant", "content": resp_text})
    logger.info("Agent round 1 response: %s", resp_text[:200])

    data = _parse_agent_response(resp_text)
    node_ids = _parse_node_ids(data)

    if not node_ids:
        thinking = data.get("thinking", "")
        fallback_ids = re.findall(r"\b(\d{4}(?:\.\d{4})*)\b", thinking)
        if fallback_ids:
            node_ids = fallback_ids
            logger.info("Extracted %d node IDs from thinking text", len(node_ids))

    if not node_ids:
        logger.warning("Agent returned no node IDs in round 1")
        return [], []

    agent_thinking = data.get("thinking", "")
    new_nodes = _resolve_nodes(tree.structure, node_ids, seen_ids, thinking=agent_thinking)
    all_retrieved.extend(new_nodes)
    logger.info(
        "Agent round 1 selected %d nodes: %s",
        len(new_nodes), [n.node.node_id for n in new_nodes],
    )

    if not new_nodes:
        return [], []

    # ---- Rounds 2+: show content, ask for more or done ----
    sufficiency_checked = False

    for round_num in range(2, max_rounds + 1):
        content_parts: list[dict[str, Any]] = []

        if use_vision and pdf_path:
            new_pages = _collect_page_numbers(
                new_nodes,
                limit=_MAX_TOTAL_PAGES - len(all_page_numbers),
            )
            deduped = [p for p in new_pages if p not in set(all_page_numbers)]
            all_page_numbers.extend(deduped)

            if deduped:
                from nanoindex.utils.pdf import render_pages
                image_uris = render_pages(pdf_path, deduped)
                for uri in image_uris:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": uri},
                    })

        section_text = _build_section_text(new_nodes)
        remaining = _remaining_outline_json(tree, seen_ids)

        review_msg = _REVIEW.format(
            content=section_text[:120000],
            query=query,
            remaining=remaining[:20000] if remaining else "All sections have been read.",
        )
        content_parts.append({"type": "text", "text": review_msg})

        if len(content_parts) == 1 and content_parts[0]["type"] == "text":
            conversation.append({"role": "user", "content": content_parts[0]["text"]})
        else:
            conversation.append({"role": "user", "content": content_parts})

        resp_text = await llm.chat(conversation, temperature=0.0, max_tokens=1024)
        conversation.append({"role": "assistant", "content": resp_text})
        logger.info("Agent round %d response: %s", round_num, resp_text[:200])

        data = _parse_agent_response(resp_text)
        action = data.get("action", "")

        if action == "done":
            # Run sufficiency check once before truly finishing
            if not sufficiency_checked and decomposition and decomposition.get("data_points"):
                sufficiency_checked = True
                remaining_outline = _remaining_outline_json(tree, seen_ids)
                if remaining_outline:
                    checklist = "\n".join(
                        f"  - {dp}" for dp in decomposition["data_points"]
                    )
                    retrieved_titles = "\n".join(
                        f"  - [{rn.node.node_id}] {rn.node.title}"
                        for rn in all_retrieved
                    )
                    suf_msg = _SUFFICIENCY.format(
                        checklist=checklist,
                        query=query,
                        retrieved_titles=retrieved_titles,
                        remaining=remaining_outline[:20000],
                    )
                    conversation.append({"role": "user", "content": suf_msg})
                    suf_resp = await llm.chat(conversation, temperature=0.0, max_tokens=1024)
                    conversation.append({"role": "assistant", "content": suf_resp})
                    logger.info("Sufficiency check response: %s", suf_resp[:200])

                    suf_data = _parse_agent_response(suf_resp)
                    suf_action = suf_data.get("action", "")
                    if suf_action in ("select_more", "select"):
                        suf_ids = _parse_node_ids(suf_data)
                        if suf_ids:
                            suf_thinking = suf_data.get("thinking", "")
                            new_nodes = _resolve_nodes(tree.structure, suf_ids, seen_ids, thinking=suf_thinking)
                            all_retrieved.extend(new_nodes)
                            logger.info(
                                "Sufficiency check retrieved %d more nodes: %s",
                                len(new_nodes), [n.node.node_id for n in new_nodes],
                            )
                            if new_nodes:
                                continue
            logger.info("Agent signalled done after round %d", round_num)
            break

        round_thinking = data.get("thinking", "")

        # Any variation of "select more"
        if action in ("select_more", "select", "request_more_sections"):
            more_ids = _parse_node_ids(data)
            if not more_ids:
                fallback_ids = re.findall(r"\b(\d{4}(?:\.\d{4})*)\b", round_thinking)
                if fallback_ids:
                    more_ids = fallback_ids
                else:
                    logger.info("Agent wants more but gave no IDs, treating as done")
                    break
            new_nodes = _resolve_nodes(tree.structure, more_ids, seen_ids, thinking=round_thinking)
            all_retrieved.extend(new_nodes)
            logger.info(
                "Agent round %d requested %d more nodes: %s",
                round_num, len(new_nodes), [n.node.node_id for n in new_nodes],
            )
            if not new_nodes:
                logger.info("No new resolvable nodes, treating as done")
                break
            continue

        # If no action recognised but there's a node_list, treat as select
        extra_ids = _parse_node_ids(data)
        if extra_ids:
            new_nodes = _resolve_nodes(tree.structure, extra_ids, seen_ids, thinking=round_thinking)
            all_retrieved.extend(new_nodes)
            if new_nodes:
                logger.info(
                    "Agent round %d (implicit select) %d nodes: %s",
                    round_num, len(new_nodes), [n.node.node_id for n in new_nodes],
                )
                continue

        logger.info("Unrecognised response in round %d, treating as done", round_num)
        break

    # Collect all page numbers from retrieved nodes
    if not all_page_numbers:
        all_page_numbers = _collect_page_numbers(all_retrieved, limit=_MAX_TOTAL_PAGES)

    return all_retrieved, all_page_numbers


# ------------------------------------------------------------------
# Answer generation (separate from retrieval)
# ------------------------------------------------------------------

async def _generate_answer(
    query: str,
    nodes: list[RetrievedNode],
    llm: LLMClient,
    *,
    pdf_path: str | Path | None = None,
    use_vision: bool = False,
    page_numbers: list[int] | None = None,
    kb_reference: str = "",
    temporal_context: str = "Answer based only on information in the document.",
    financial: bool = True,
) -> str:
    """Generate a final answer from gathered context.

    If vision mode fails (e.g. payload too large), automatically falls back
    to text-only mode.
    """
    section_text = _build_section_text(nodes)
    answer_template = _ANSWER if financial else _ANSWER_GENERAL
    prompt = answer_template.format(
        query=query,
        context=section_text[:200000],
        kb_reference=kb_reference,
        temporal_context=temporal_context,
    )

    # Try vision mode first, fall back to text if it fails
    if use_vision and pdf_path and page_numbers:
        try:
            from nanoindex.utils.pdf import render_pages
            pages = sorted(page_numbers)[:_MAX_TOTAL_PAGES]
            image_uris = render_pages(pdf_path, pages)
            content_parts: list[dict[str, Any]] = []
            for uri in image_uris:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": uri},
                })
            content_parts.append({"type": "text", "text": prompt})
            messages: list[dict[str, Any]] = [
                {"role": "user", "content": content_parts}
            ]
            return await llm.chat(messages, temperature=0.0, max_tokens=2048)
        except Exception as exc:
            logger.warning(
                "Vision generation failed (%s), falling back to text-only mode",
                exc,
            )

    # Text-only mode (or vision fallback)
    messages = [{"role": "user", "content": prompt}]
    return await llm.chat(messages, temperature=0.0, max_tokens=2048)


async def _verify_calculations(
    query: str,
    answer: str,
    nodes: list[RetrievedNode],
    llm: LLMClient,
) -> str:
    """Re-check numerical calculations in the answer. Returns corrected text."""
    if not _HAS_NUMBERS.search(answer):
        return answer

    context = _build_section_text(nodes)
    prompt = _VERIFY.format(
        query=query,
        answer=answer,
        context=context[:120000],
    )
    try:
        verified = await llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2048,
        )
        if verified and len(verified.strip()) > 10:
            logger.info("Calculation verification complete")
            return verified
    except Exception:
        logger.warning("Calculation verification failed, using original answer", exc_info=True)
    return answer


# ------------------------------------------------------------------
# Fix 1: Small document full-content mode
# ------------------------------------------------------------------

_SMALL_TREE_MAX_TEXT = 200_000  # Only dump full content if it fits in context

def _is_small_tree(tree: DocumentTree) -> bool:
    """Return True when the tree is small enough to skip navigation.

    Considers both node count and total text size — a tree with few nodes
    but massive text (e.g. exhibits) should use agentic retrieval so the
    agent can navigate to the relevant parts instead of truncating.
    """
    count = sum(1 for _ in iter_nodes(tree.structure))
    if count >= _SMALL_TREE_THRESHOLD:
        return False
    total_text = sum(len(n.text or "") for n in iter_nodes(tree.structure))
    return total_text < _SMALL_TREE_MAX_TEXT


def _dump_all_content(tree: DocumentTree) -> list[RetrievedNode]:
    """Return ALL leaf text from the tree, for small documents."""
    results: list[RetrievedNode] = []
    for node in iter_nodes(tree.structure):
        text = node.text or collect_text(node)
        if text and len(text.strip()) > 20:
            results.append(RetrievedNode(node=node, text=text))
    return results


# ------------------------------------------------------------------
# Fix 4: Keyword fallback — scan all leaf nodes by keyword
# ------------------------------------------------------------------

def _keyword_search_nodes(
    tree: DocumentTree,
    query: str,
    decomposition: dict | None = None,
    already_seen: set[str] | None = None,
    max_results: int = 10,
) -> list[RetrievedNode]:
    """Brute-force keyword search across all leaf nodes."""
    seen = already_seen or set()
    keywords: list[str] = []

    for word in query.lower().split():
        cleaned = word.strip("?.,!\"'()[]")
        if len(cleaned) > 3 and cleaned not in {"what", "which", "does", "have", "that", "this", "from", "with", "based", "about", "answer", "following", "question"}:
            keywords.append(cleaned)

    if decomposition:
        for dp in decomposition.get("data_points", []):
            for w in dp.lower().split():
                cleaned = w.strip("?.,!\"'()")
                if len(cleaned) > 3:
                    keywords.append(cleaned)

    if not keywords:
        return []

    scored: list[tuple[int, TreeNode, str]] = []
    for node in iter_nodes(tree.structure):
        if node.node_id in seen:
            continue
        text = node.text or collect_text(node)
        if not text or len(text.strip()) < 50:
            continue
        text_lower = text.lower()
        title_lower = (node.title or "").lower()
        summary_lower = (node.summary or "").lower()
        score = sum(
            1 for kw in keywords
            if kw in text_lower or kw in title_lower or kw in summary_lower
        )
        if score > 0:
            scored.append((score, node, text))

    scored.sort(key=lambda x: x[0], reverse=True)
    results: list[RetrievedNode] = []
    for _, node, text in scored[:max_results]:
        if node.node_id not in seen:
            seen.add(node.node_id)
            results.append(RetrievedNode(node=node, text=text))
    return results


# ------------------------------------------------------------------
# Fix 5: Financial statement completeness guard
# ------------------------------------------------------------------

_STATEMENT_KEYWORDS: dict[str, list[str]] = {
    "income statement": ["income", "operations", "earnings", "revenue", "profit", "loss"],
    "balance sheet": ["balance sheet", "financial position", "assets", "liabilities"],
    "cash flow statement": ["cash flow", "cash flows"],
    "segment data": ["segment", "business segment", "operating segment"],
    "notes to financial statements": ["note ", "notes to"],
}


def _check_statement_coverage(
    decomposition: dict | None,
    retrieved: list[RetrievedNode],
    tree: DocumentTree,
    seen_ids: set[str],
) -> list[RetrievedNode]:
    """If decomposition requested specific statements, verify they're covered."""
    if not decomposition:
        return []

    needed = decomposition.get("statements_needed", [])
    if not needed:
        return []

    retrieved_text = " ".join(
        (rn.node.title or "") + " " + (rn.node.summary or "")
        for rn in retrieved
    ).lower()

    missing_keywords: list[str] = []
    for stmt in needed:
        stmt_lower = stmt.lower()
        kw_list = _STATEMENT_KEYWORDS.get(stmt_lower, [stmt_lower])
        if not any(kw in retrieved_text for kw in kw_list):
            missing_keywords.extend(kw_list)

    if not missing_keywords:
        return []

    logger.info("Statement guard: missing coverage for keywords %s", missing_keywords)

    extra: list[RetrievedNode] = []
    for node in iter_nodes(tree.structure):
        if node.node_id in seen_ids:
            continue
        title_lower = (node.title or "").lower()
        summary_lower = (node.summary or "").lower()
        combined = title_lower + " " + summary_lower
        if any(kw in combined for kw in missing_keywords):
            seen_ids.add(node.node_id)
            text = node.text or collect_text(node)
            if text and len(text.strip()) > 50:
                extra.append(RetrievedNode(node=node, text=text))

    if extra:
        logger.info(
            "Statement guard added %d nodes: %s",
            len(extra), [n.node.node_id for n in extra],
        )
    return extra


# ------------------------------------------------------------------
# Fix 6: Targeted Notes-to-FS retrieval
# ------------------------------------------------------------------

_NOTES_TRIGGER_KEYWORDS: list[str] = [
    "derivative", "notional", "hedging", "fair value",
    "geographic", "region", "country", "international",
    "segment", "business unit",
    "restructuring", "impairment", "goodwill",
    "lease", "right-of-use",
    "stock-based compensation", "RSU", "restricted stock",
    "acquisition", "merger", "divestiture",
    "pension", "retirement", "benefit plan",
    "debt", "borrowing", "credit facility",
    "contingent", "litigation", "legal proceeding",
]


def _targeted_notes_retrieval(
    query: str,
    decomposition: dict | None,
    retrieved: list[RetrievedNode],
    tree: DocumentTree,
    seen_ids: set[str],
) -> list[RetrievedNode]:
    """Scan Notes-to-FS sections when the query involves topics typically in Notes."""
    query_lower = query.lower()
    decomp_text = " ".join(
        decomposition.get("data_points", []) + decomposition.get("statements_needed", [])
    ).lower() if decomposition else ""
    combined_query = query_lower + " " + decomp_text

    triggered = any(kw in combined_query for kw in _NOTES_TRIGGER_KEYWORDS)
    if not triggered:
        return []

    retrieved_titles = " ".join(
        (rn.node.title or "").lower() for rn in retrieved
    )
    if "note" in retrieved_titles:
        return []

    extra: list[RetrievedNode] = []
    matching_keywords = [kw for kw in _NOTES_TRIGGER_KEYWORDS if kw in combined_query]
    logger.info("Notes retrieval triggered for keywords: %s", matching_keywords)

    for node in iter_nodes(tree.structure):
        if node.node_id in seen_ids:
            continue
        title_lower = (node.title or "").lower()
        summary_lower = (node.summary or "").lower()
        node_combined = title_lower + " " + summary_lower
        is_note = "note" in title_lower
        has_topic = any(kw in node_combined for kw in matching_keywords)
        if is_note and has_topic:
            seen_ids.add(node.node_id)
            text = node.text or collect_text(node)
            if text and len(text.strip()) > 50:
                extra.append(RetrievedNode(node=node, text=text))

    if extra:
        logger.info(
            "Notes retrieval added %d nodes: %s",
            len(extra), [n.node.node_id for n in extra],
        )
    return extra


# ------------------------------------------------------------------
# Fix 7: Multi-year data completeness check
# ------------------------------------------------------------------

_YEAR_PATTERN = re.compile(r"(?:FY|fy|fiscal\s*(?:year)?\s*)?20[12]\d")


def _check_multi_year_coverage(
    query: str,
    decomposition: dict | None,
    retrieved: list[RetrievedNode],
    tree: DocumentTree,
    seen_ids: set[str],
) -> list[RetrievedNode]:
    """If the query/decomposition references multiple years, ensure all are covered."""
    combined = query
    if decomposition:
        combined += " " + " ".join(decomposition.get("data_points", []))

    years_needed = set(
        int(m[-4:]) for m in _YEAR_PATTERN.findall(combined)
    )
    if len(years_needed) < 2:
        return []

    retrieved_text = " ".join(rn.text or "" for rn in retrieved)
    years_found = set(
        int(m[-4:]) for m in _YEAR_PATTERN.findall(retrieved_text)
    )
    missing_years = years_needed - years_found
    if not missing_years:
        return []

    logger.info("Multi-year guard: missing years %s in retrieved content", missing_years)
    year_strs = [str(y) for y in missing_years]

    extra: list[RetrievedNode] = []
    for node in iter_nodes(tree.structure):
        if node.node_id in seen_ids:
            continue
        text = node.text or collect_text(node)
        if not text or len(text.strip()) < 50:
            continue
        if any(ys in text for ys in year_strs):
            seen_ids.add(node.node_id)
            extra.append(RetrievedNode(node=node, text=text))

    if extra:
        logger.info(
            "Multi-year guard added %d nodes for missing years %s: %s",
            len(extra), missing_years, [n.node.node_id for n in extra],
        )
    return extra


# ------------------------------------------------------------------
# Fix 8: Temporal context from document name
# ------------------------------------------------------------------

_DOC_NAME_PATTERN = re.compile(
    r"^(?P<company>.+?)_"
    r"(?P<year>20[12]\d)"
    r"(?:Q(?P<quarter>[1-4]))?"
    r"_(?P<filing_type>.+?)(?:_dated-(?P<date>.+))?$",
    re.IGNORECASE,
)

_FILING_DATE_OFFSETS: dict[str, str] = {
    "10K": "filed approximately 60 days after the fiscal year ended",
    "10Q": "filed approximately 45 days after the quarter ended",
    "8K": "filed within a few days of the event it describes",
    "EARNINGS": "released shortly after the quarter ended",
}


def _infer_temporal_context(doc_name: str) -> str:
    """Build a temporal context string from the document name.

    Returns a sentence like: "This is a 10-K filing for FY2018 by 3M,
    filed approximately February 2019.  Answer as if today is shortly
    after the filing date."
    """
    m = _DOC_NAME_PATTERN.match(doc_name)
    if not m:
        return "Answer based only on information in the document."

    company = m.group("company").replace("_", " ")
    year = int(m.group("year"))
    quarter = m.group("quarter")
    filing_type = m.group("filing_type").upper()
    explicit_date = m.group("date")

    if explicit_date:
        return (
            f"This is a {filing_type} filing by {company}, dated {explicit_date}. "
            f"Answer as if today's date is shortly after {explicit_date}. "
            f"Do NOT use knowledge of events occurring after this date."
        )

    timing = _FILING_DATE_OFFSETS.get(filing_type, "filed around that period")

    if filing_type == "10K":
        approx_date = f"early {year + 1}"
        period_desc = f"FY{year}"
    elif filing_type == "10Q" and quarter:
        q = int(quarter)
        quarter_end_months = {1: "March", 2: "June", 3: "September", 4: "December"}
        end_month = quarter_end_months.get(q, "")
        approx_date = f"shortly after {end_month} {year}"
        period_desc = f"Q{q} {year}"
    elif filing_type == "EARNINGS" and quarter:
        q = int(quarter)
        quarter_end_months = {1: "March", 2: "June", 3: "September", 4: "December"}
        end_month = quarter_end_months.get(q, "")
        approx_date = f"shortly after {end_month} {year}"
        period_desc = f"Q{q} {year}"
    else:
        approx_date = f"around {year}"
        period_desc = str(year)

    fy_note = (
        "IMPORTANT — Fiscal year naming: Some companies have non-calendar fiscal years "
        "(e.g., fiscal year ending in January or June). When answering questions about "
        f"'FY{year}', use the company's OWN fiscal year labels as stated in the filing. "
        "For Q4 earnings releases, the covered fiscal year may have ALREADY ended — "
        "the data IS available in the filing even though the filing date is after year-end. "
        "Do NOT refuse to answer by claiming data is for a 'future' period if the filing "
        "clearly contains full-year results."
    )

    return (
        f"This is a {filing_type} filing for {period_desc} by {company}, "
        f"{timing}. Approximate filing date: {approx_date}. "
        f"Answer as if today's date is {approx_date}. "
        f"Do NOT use knowledge of events occurring after the filing date. "
        f"{fy_note}"
    )


# ------------------------------------------------------------------
# Graph-seeded agentic retrieval (skip Round 1, start from graph nodes)
# ------------------------------------------------------------------

async def _run_graph_seeded_retrieval(
    query: str,
    tree: DocumentTree,
    llm: LLMClient,
    *,
    seed_nodes: list[RetrievedNode],
    seen_ids: set[str],
    pdf_path: str | Path | None = None,
    use_vision: bool = False,
    max_rounds: int = 4,
    decomposition: dict | None = None,
) -> tuple[list[RetrievedNode], list[int]]:
    """Agentic retrieval starting from graph-seeded nodes instead of full tree.

    Skips the expensive Round 1 (full tree outline → LLM selection) since the
    graph already identified relevant nodes. Goes straight to review rounds
    where the agent can request additional sections.
    """
    all_retrieved = list(seed_nodes)
    all_page_numbers: list[int] = _collect_page_numbers(seed_nodes, limit=_MAX_TOTAL_PAGES)
    conversation: list[dict] = []
    new_nodes = seed_nodes

    logger.info(
        "Graph-seeded retrieval: starting with %d seed nodes",
        len(seed_nodes),
    )

    decomp_text = _format_decomposition(decomposition or {})
    sufficiency_checked = False

    for round_num in range(1, max_rounds + 1):
        content_parts: list[dict] = []

        if use_vision and pdf_path:
            new_pages = _collect_page_numbers(
                new_nodes,
                limit=_MAX_TOTAL_PAGES - len(all_page_numbers),
            )
            deduped = [p for p in new_pages if p not in set(all_page_numbers)]
            all_page_numbers.extend(deduped)

            if deduped:
                from nanoindex.utils.pdf import render_pages
                image_uris = render_pages(pdf_path, deduped)
                for uri in image_uris:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": uri},
                    })

        section_text = _build_section_text(new_nodes)
        remaining = _remaining_outline_json(tree, seen_ids)

        if round_num == 1:
            # First review: show seed content + context about how we got here
            review_msg = f"""These sections were identified via entity graph matching for your question.

{decomp_text}

{_REVIEW.format(
    content=section_text[:120000],
    query=query,
    remaining=remaining[:20000] if remaining else "All sections have been read.",
)}"""
        else:
            review_msg = _REVIEW.format(
                content=section_text[:120000],
                query=query,
                remaining=remaining[:20000] if remaining else "All sections have been read.",
            )

        content_parts.append({"type": "text", "text": review_msg})

        if len(content_parts) == 1 and content_parts[0]["type"] == "text":
            conversation.append({"role": "user", "content": content_parts[0]["text"]})
        else:
            conversation.append({"role": "user", "content": content_parts})

        resp_text = await llm.chat(conversation, temperature=0.0, max_tokens=1024)
        conversation.append({"role": "assistant", "content": resp_text})
        logger.info("Graph-seeded round %d response: %s", round_num, resp_text[:200])

        data = _parse_agent_response(resp_text)
        action = data.get("action", "")

        if action == "done":
            if not sufficiency_checked and decomposition and decomposition.get("data_points"):
                sufficiency_checked = True
                remaining_outline = _remaining_outline_json(tree, seen_ids)
                if remaining_outline:
                    checklist = "\n".join(
                        f"  - {dp}" for dp in decomposition["data_points"]
                    )
                    retrieved_titles = "\n".join(
                        f"  - [{rn.node.node_id}] {rn.node.title}"
                        for rn in all_retrieved
                    )
                    suf_msg = _SUFFICIENCY.format(
                        checklist=checklist,
                        query=query,
                        retrieved_titles=retrieved_titles,
                        remaining=remaining_outline[:20000],
                    )
                    conversation.append({"role": "user", "content": suf_msg})
                    suf_resp = await llm.chat(conversation, temperature=0.0, max_tokens=1024)
                    conversation.append({"role": "assistant", "content": suf_resp})

                    suf_data = _parse_agent_response(suf_resp)
                    if suf_data.get("action") in ("select_more", "select"):
                        suf_ids = _parse_node_ids(suf_data)
                        if suf_ids:
                            new_nodes = _resolve_nodes(
                                tree.structure, suf_ids, seen_ids,
                                thinking=suf_data.get("thinking", ""),
                            )
                            all_retrieved.extend(new_nodes)
                            if new_nodes:
                                continue
            logger.info("Graph-seeded agent done after round %d", round_num)
            break

        if action in ("select_more", "select", "request_more_sections"):
            more_ids = _parse_node_ids(data)
            thinking = data.get("thinking", "")
            if not more_ids:
                fallback_ids = re.findall(r"\b(\d{4}(?:\.\d{4})*)\b", thinking)
                if fallback_ids:
                    more_ids = fallback_ids
                else:
                    break
            new_nodes = _resolve_nodes(tree.structure, more_ids, seen_ids, thinking=thinking)
            all_retrieved.extend(new_nodes)
            logger.info(
                "Graph-seeded round %d: +%d nodes",
                round_num, len(new_nodes),
            )
            if not new_nodes:
                break
            continue

        extra_ids = _parse_node_ids(data)
        if extra_ids:
            new_nodes = _resolve_nodes(tree.structure, extra_ids, seen_ids)
            all_retrieved.extend(new_nodes)
            if new_nodes:
                continue
        break

    if not all_page_numbers:
        all_page_numbers = _collect_page_numbers(all_retrieved, limit=_MAX_TOTAL_PAGES)

    return all_retrieved, all_page_numbers


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------

async def agentic_ask(
    query: str,
    tree: DocumentTree,
    llm: LLMClient,
    config: NanoIndexConfig,
    *,
    pdf_path: str | Path | None = None,
    use_vision: bool = False,
    max_rounds: int = _MAX_ROUNDS,
    include_metadata: bool = False,
    graph: "DocumentGraph | None" = None,
) -> Answer:
    """Multi-round agentic retrieval then answer generation.

    Phase 0: Query decomposition + temporal context inference.
    Phase 1a: Small-doc shortcut — dump all content for tiny documents.
    Phase 1b: Multi-round tree navigation retrieval for larger docs.
    Phase 1c: Statement completeness guard — fill gaps in required statements.
    Phase 1d: Targeted Notes-to-FS retrieval for specialized topics.
    Phase 1e: Multi-year data completeness check.
    Phase 2: Answer generation with temporal context + KB + must-compute prompt.
    Phase 3: Calculation verification.
    Phase 4: Self-evaluation — if the answer looks like a refusal, retry with
              keyword fallback retrieval and/or full-content dump.
    """
    # Detect if this is a financial document (affects prompts)
    financial = _is_financial_doc(tree.doc_name, query)
    logger.info("Domain: %s", "financial" if financial else "general")

    # Phase 0: Query decomposition
    decomposition = await _decompose_query(query, llm, financial=financial)
    if decomposition:
        logger.info(
            "Query decomposition: %d data points, %d statements",
            len(decomposition.get("data_points", [])),
            len(decomposition.get("statements_needed", [])),
        )

    # Temporal context from document name
    temporal_ctx = _infer_temporal_context(tree.doc_name)
    logger.info("Temporal context: %s", temporal_ctx)

    seen_ids: set[str] = set()

    # Phase 1a: Small document full-content mode (Fix 1)
    if _is_small_tree(tree):
        logger.info("Small document (%d nodes) — using full-content mode", sum(1 for _ in iter_nodes(tree.structure)))
        nodes = _dump_all_content(tree)
        for rn in nodes:
            seen_ids.add(rn.node.node_id)
        page_numbers = _collect_page_numbers(nodes, limit=_MAX_TOTAL_PAGES)
    elif graph and graph.entities:
        # Phase 1b-graph: Graph-seeded agentic retrieval
        # Use graph to find seed nodes (skips expensive Round 1 full-tree LLM call),
        # then let the agentic review loop reason and expand from there.
        from nanoindex.core.graph_builder import (
            build_entity_to_nodes, build_nx_graph,
            entity_keyword_match, graph_expand,
        )
        entity_to_nodes = build_entity_to_nodes(graph)
        seed_ids = entity_keyword_match(query, entity_to_nodes)

        if decomposition:
            for dp in decomposition.get("data_points", []):
                seed_ids |= entity_keyword_match(dp, entity_to_nodes)

        if seed_ids:
            nx_graph = build_nx_graph(graph)
            seed_ids |= graph_expand(nx_graph, seed_ids, entity_to_nodes, hops=2)

            # Cap seed nodes — if graph matching returns too many (>50% of doc),
            # it's not being selective enough. Fall back to standard agentic.
            all_node_ids = {n.node_id for n in iter_nodes(tree.structure)}
            seed_ids &= all_node_ids
            max_seeds = max(30, len(all_node_ids) // 3)  # cap at 1/3 of doc

            if len(seed_ids) > max_seeds:
                logger.info(
                    "Graph-seeded agentic: %d seeds too broad (max %d), falling back to standard agentic",
                    len(seed_ids), max_seeds,
                )
                seed_ids = set()  # trigger fallback below
            else:
                logger.info("Graph-seeded agentic: %d seed nodes from entity graph", len(seed_ids))

            seed_node_ids = sorted(seed_ids)
            seed_nodes = _resolve_nodes(tree.structure, seed_node_ids, seen_ids) if seed_ids else []

            if seed_nodes:
                # Run agentic review rounds starting from graph-seeded content
                nodes, page_numbers = await _run_graph_seeded_retrieval(
                    query, tree, llm,
                    seed_nodes=seed_nodes,
                    seen_ids=seen_ids,
                    pdf_path=pdf_path,
                    use_vision=use_vision,
                    max_rounds=max_rounds - 1,  # save a round since we skip Round 1
                    decomposition=decomposition,
                )
            else:
                # Graph seeds didn't resolve, fall back to standard agentic
                nodes, page_numbers = await _run_retrieval(
                    query, tree, llm,
                    pdf_path=pdf_path,
                    use_vision=use_vision,
                    max_rounds=max_rounds,
                    decomposition=decomposition,
                    financial=financial,
                )
                for rn in nodes:
                    seen_ids.add(rn.node.node_id)
        else:
            # No graph matches, fall back to standard agentic
            logger.info("Graph-seeded agentic: no entity matches, falling back to standard")
            nodes, page_numbers = await _run_retrieval(
                query, tree, llm,
                pdf_path=pdf_path,
                use_vision=use_vision,
                max_rounds=max_rounds,
                decomposition=decomposition,
                financial=financial,
            )
            for rn in nodes:
                seen_ids.add(rn.node.node_id)
    else:
        # Phase 1b: Standard agentic retrieval (no graph available)
        nodes, page_numbers = await _run_retrieval(
            query, tree, llm,
            pdf_path=pdf_path,
            use_vision=use_vision,
            max_rounds=max_rounds,
            decomposition=decomposition,
            financial=financial,
        )
        for rn in nodes:
            seen_ids.add(rn.node.node_id)

    if not nodes:
        # Keyword fallback before giving up entirely (Fix 4)
        nodes = _keyword_search_nodes(tree, query, decomposition, seen_ids)
        if not nodes:
            return Answer(
                content="Unable to identify relevant sections in the document.",
                mode="agentic",
            )
        logger.info("Keyword fallback rescued retrieval with %d nodes", len(nodes))
        page_numbers = _collect_page_numbers(nodes, limit=_MAX_TOTAL_PAGES)

    # Phase 1c-1e: Financial-specific completeness guards (only for financial docs)
    if financial:
        # Phase 1c: Statement completeness guard (Fix 5)
        guard_nodes = _check_statement_coverage(decomposition, nodes, tree, seen_ids)
        if guard_nodes:
            nodes.extend(guard_nodes)
            page_numbers = _collect_page_numbers(nodes, limit=_MAX_TOTAL_PAGES)

        # Phase 1d: Targeted Notes-to-FS retrieval (Fix 6)
        notes_nodes = _targeted_notes_retrieval(query, decomposition, nodes, tree, seen_ids)
        if notes_nodes:
            nodes.extend(notes_nodes)
            page_numbers = _collect_page_numbers(nodes, limit=_MAX_TOTAL_PAGES)

        # Phase 1e: Multi-year data completeness check (Fix 7)
        year_nodes = _check_multi_year_coverage(query, decomposition, nodes, tree, seen_ids)
        if year_nodes:
            nodes.extend(year_nodes)
            page_numbers = _collect_page_numbers(nodes, limit=_MAX_TOTAL_PAGES)

    logger.info(
        "Agentic retrieval complete: %d nodes, %d pages",
        len(nodes), len(page_numbers),
    )

    # Phase 1f: Knowledge base lookup — inject canonical definitions (financial only)
    kb_ref = ""
    if financial:
        from nanoindex.knowledge import lookup_relevant_terms
        kb_ref = lookup_relevant_terms(query, decomposition, max_results=4)
        if kb_ref:
            logger.info("KB injected %d chars of financial reference", len(kb_ref))

    # Phase 2: Answer generation (with temporal context + KB reference)
    answer_text = await _generate_answer(
        query, nodes, llm,
        pdf_path=pdf_path,
        use_vision=use_vision,
        page_numbers=page_numbers,
        kb_reference=kb_ref,
        temporal_context=temporal_ctx,
        financial=financial,
    )

    # Phase 3: Calculation verification
    answer_text = await _verify_calculations(query, answer_text, nodes, llm)

    # Phase 4: Self-evaluation + re-retrieval (Fix 2)
    _SELF_EVAL_MAX_RETRIES = 2
    for retry_num in range(_SELF_EVAL_MAX_RETRIES):
        if not _REFUSAL_PATTERNS.search(answer_text):
            break

        logger.info(
            "Self-eval retry %d: answer looks like a refusal, attempting recovery",
            retry_num + 1,
        )

        if retry_num == 0:
            extra_nodes = _keyword_search_nodes(
                tree, query, decomposition, seen_ids, max_results=15,
            )
        else:
            extra_nodes = _dump_all_content(tree)
            for rn in extra_nodes:
                seen_ids.add(rn.node.node_id)

        if extra_nodes:
            combined = nodes + [rn for rn in extra_nodes if rn.node.node_id not in {n.node.node_id for n in nodes}]
            page_numbers = _collect_page_numbers(combined, limit=_MAX_TOTAL_PAGES)
            logger.info(
                "Self-eval retry %d: re-answering with %d nodes (was %d)",
                retry_num + 1, len(combined), len(nodes),
            )
            nodes = combined
            answer_text = await _generate_answer(
                query, nodes, llm,
                pdf_path=pdf_path,
                use_vision=use_vision,
                page_numbers=page_numbers,
                kb_reference=kb_ref,
                temporal_context=temporal_ctx,
                financial=financial,
            )
            answer_text = await _verify_calculations(query, answer_text, nodes, llm)
        else:
            break

    return Answer(
        content=answer_text,
        citations=_build_citations(nodes, tree, include_metadata),
        mode="agentic",
    )


def _build_citations(
    nodes: list[RetrievedNode],
    tree: DocumentTree | None = None,
    include_metadata: bool = False,
) -> list[Citation]:
    citations: list[Citation] = []
    for rn in nodes:
        pages = (
            list(range(rn.node.start_index, rn.node.end_index + 1))
            if rn.node.start_index else []
        )
        # Always propagate bounding boxes from the node itself
        bboxes: list[BoundingBox] = list(rn.node.bounding_boxes)
        dims: list[PageDimensions] = []
        if include_metadata and pages and tree:
            page_set = set(pages)
            # Enrich with all bboxes for cited pages from the tree
            tree_bboxes = [bb for bb in tree.all_bounding_boxes if bb.page in page_set]
            if tree_bboxes:
                bboxes = tree_bboxes
            dims = [pd for pd in tree.page_dimensions if pd.page in page_set]
        citations.append(Citation(
            node_id=rn.node.node_id,
            title=rn.node.title,
            doc_name=rn.doc_name,
            pages=pages,
            bounding_boxes=bboxes,
            page_dimensions=dims,
        ))
    return citations
