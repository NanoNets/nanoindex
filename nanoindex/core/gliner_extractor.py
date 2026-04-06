"""Entity extraction using GLiNER zero-shot NER.

Better than spaCy: domain-adaptive entity types, no retraining needed.
Model: ~400MB download on first use.

pip install gliner
"""

from __future__ import annotations

import logging

from nanoindex.models import DocumentGraph, DocumentTree, Entity, Relationship
from nanoindex.utils.tree_ops import iter_nodes

logger = logging.getLogger(__name__)

# Domain label presets — tuned for real document types
DOMAIN_LABELS = {
    "financial": [
        "Company", "Person", "Revenue", "NetIncome", "OperatingIncome",
        "GrossProfit", "EPS", "Segment", "FiscalPeriod", "FinancialMetric",
        "CashFlow", "Debt", "Dividend", "Acquisition", "Location",
    ],
    "sec_10k": [
        "Company", "ExecutiveName", "Revenue", "NetIncome", "OperatingIncome",
        "GrossMargin", "EPS", "BusinessSegment", "FiscalYear", "FinancialMetric",
        "CapitalExpenditure", "WorkingCapital", "TotalAssets", "TotalDebt",
        "SharesOutstanding", "Dividend", "Acquisition", "Restructuring",
    ],
    "sec_10q": [
        "Company", "Revenue", "NetIncome", "EPS", "BusinessSegment",
        "FiscalQuarter", "FinancialMetric", "CashFlow", "WorkingCapital",
        "QuickRatio", "InventoryTurnover", "DebtSecurity", "Location",
    ],
    "earnings": [
        "Company", "ExecutiveName", "Revenue", "EPS", "Guidance",
        "BusinessSegment", "FiscalQuarter", "GrowthRate", "FinancialMetric",
        "AdjustedEPS", "FreeCashFlow", "ARR", "NetRevenueRetention",
    ],
    "legal": [
        "Party", "Court", "CaseNumber", "Statute", "Jurisdiction",
        "Damages", "LegalTerm", "Judge", "FilingDate", "Attorney",
    ],
    "medical": [
        "Patient", "Diagnosis", "Drug", "Procedure", "Dosage",
        "Symptom", "LabTest", "Physician", "Hospital", "InsuranceCode",
    ],
    "insurance": [
        "Insurer", "PolicyNumber", "ClaimNumber", "CoverageType",
        "Premium", "Deductible", "LossAmount", "ReserveAmount", "ClaimDate",
    ],
    "generic": [
        "Organization", "Person", "Location", "Date", "Product",
        "Event", "Money", "Percentage", "Document",
    ],
}


def _detect_domain(text: str, doc_name: str = "") -> str:
    """Detect domain from document name and content."""
    name_lower = doc_name.lower()

    # SEC filing type detection from doc name
    if "_10K" in doc_name or "_10k" in doc_name or "10-K" in doc_name:
        return "sec_10k"
    if "_10Q" in doc_name or "_10q" in doc_name or "10-Q" in doc_name:
        return "sec_10q"
    if "EARNINGS" in doc_name or "earnings" in name_lower:
        return "earnings"

    # Content-based detection
    text_lower = text[:5000].lower()
    scores = {
        "financial": sum(1 for w in ["revenue", "earnings", "fiscal", "eps", "ebitda", "sec", "10-k", "margin", "operating income", "net income"] if w in text_lower),
        "legal": sum(1 for w in ["court", "plaintiff", "defendant", "statute", "jurisdiction", "filing", "verdict", "case no"] if w in text_lower),
        "medical": sum(1 for w in ["patient", "diagnosis", "treatment", "clinical", "dosage", "symptom", "hospital"] if w in text_lower),
        "insurance": sum(1 for w in ["policy", "claim", "premium", "coverage", "deductible", "insured", "loss run"] if w in text_lower),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] >= 3 else "generic"


def _load_gliner():
    """Load GLiNER model, installing if needed."""
    try:
        from gliner import GLiNER
    except ImportError:
        raise ImportError("pip install gliner — required for GLiNER entity extraction")

    logger.info("Loading GLiNER model (first time downloads ~400MB)...")
    return GLiNER.from_pretrained("urchade/gliner_medium-v2.1")


def extract_entities_gliner(tree: DocumentTree) -> DocumentGraph:
    """Extract entities using GLiNER zero-shot NER + spaCy dependency parsing for relationships."""
    all_nodes = list(iter_nodes(tree.structure))
    full_text = " ".join(n.text or "" for n in all_nodes[:5])

    # Detect domain from doc name + content, get labels
    domain = _detect_domain(full_text, doc_name=tree.doc_name)
    labels = DOMAIN_LABELS.get(domain, DOMAIN_LABELS["generic"])

    model = _load_gliner()

    # Extract entities from each node
    entity_mentions: dict[str, dict] = {}  # normalized_name -> {name, type, descriptions, node_ids}

    for node in all_nodes:
        text = node.text or ""
        if len(text) < 20:
            continue

        # GLiNER extraction (process in chunks of 500 chars for accuracy)
        for chunk_start in range(0, min(len(text), 10000), 500):
            chunk = text[chunk_start:chunk_start + 512]
            try:
                entities = model.predict_entities(chunk, labels, threshold=0.4)
            except Exception:
                continue

            for ent in entities:
                name = ent["text"].strip()
                if len(name) < 2 or len(name) > 80:
                    continue

                key = name.lower()
                if key not in entity_mentions:
                    entity_mentions[key] = {
                        "name": name,
                        "type": ent["label"],
                        "descriptions": set(),
                        "node_ids": set(),
                    }
                entity_mentions[key]["node_ids"].add(node.node_id)

                # Get sentence context as description
                idx = text.find(name)
                if idx >= 0:
                    start = max(0, text.rfind(".", 0, idx) + 1)
                    end = text.find(".", idx + len(name))
                    if end < 0:
                        end = min(len(text), idx + 150)
                    sent = text[start:end].strip()
                    if len(sent) < 200:
                        entity_mentions[key]["descriptions"].add(sent)

    # Use spaCy for relationship extraction (SVO triples)
    relationships = _extract_relationships_spacy(all_nodes, entity_mentions)

    # Build entities list
    entities = []
    for key, info in entity_mentions.items():
        desc = ""
        if info["descriptions"]:
            descs = sorted(info["descriptions"], key=len)
            desc = descs[0]
        entities.append(Entity(
            name=info["name"],
            entity_type=info["type"],
            description=desc[:200],
            source_node_ids=sorted(info["node_ids"]),
        ))

    logger.info(
        "GLiNER extraction (%s domain): %d entities, %d relationships from %d nodes",
        domain, len(entities), len(relationships), len(all_nodes),
    )

    return DocumentGraph(doc_name=tree.doc_name, entities=entities, relationships=relationships)


def _extract_relationships_spacy(nodes, entity_mentions):
    """Use spaCy dependency parsing for SVO triples (not co-occurrence)."""
    import spacy

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        import sys

        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm", "--quiet"])
        nlp = spacy.load("en_core_web_sm")

    relationships = []
    seen: set[tuple[str, str, str]] = set()

    for node in nodes:
        text = node.text or ""
        if len(text) < 20:
            continue

        doc = nlp(text[:10000])

        # SVO triples
        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass") and token.head.pos_ == "VERB":
                subj = token.text.lower()
                verb = token.head
                for child in verb.children:
                    if child.dep_ in ("dobj", "attr", "pobj", "oprd"):
                        obj = child.text.lower()
                        if subj in entity_mentions and obj in entity_mentions:
                            key = (entity_mentions[subj]["name"], entity_mentions[obj]["name"], verb.lemma_)
                            if key not in seen:
                                seen.add(key)
                                relationships.append(Relationship(
                                    source=entity_mentions[subj]["name"],
                                    target=entity_mentions[obj]["name"],
                                    keywords=verb.lemma_,
                                    source_node_ids=[node.node_id],
                                ))

        # Prepositional relationships: "CEO of Apple", "headquartered in Cupertino"
        for token in doc:
            if token.dep_ == "prep" and token.head.pos_ in ("NOUN", "PROPN", "VERB"):
                head_text = token.head.text.lower()
                for pobj in token.children:
                    if pobj.dep_ == "pobj":
                        pobj_text = pobj.text.lower()
                        if head_text in entity_mentions and pobj_text in entity_mentions:
                            rel_text = f"{token.head.text} {token.text}"
                            key = (entity_mentions[head_text]["name"], entity_mentions[pobj_text]["name"], rel_text)
                            if key not in seen:
                                seen.add(key)
                                relationships.append(Relationship(
                                    source=entity_mentions[head_text]["name"],
                                    target=entity_mentions[pobj_text]["name"],
                                    keywords=rel_text,
                                    source_node_ids=[node.node_id],
                                ))

    return relationships
