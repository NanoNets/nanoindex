"""Entity extraction using GLiNER2 zero-shot NER.

Better than spaCy: domain-adaptive entity types, no retraining needed.
Model: ~1GB download on first use.

pip install gliner2
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


_MODEL_CACHE: dict[str, object] = {}


def _load_gliner2():
    """Load GLiNER2 large model (cached after first load). Auto-detects GPU."""
    if "gliner2" in _MODEL_CACHE:
        return _MODEL_CACHE["gliner2"]
    try:
        from gliner2 import GLiNER2
    except ImportError:
        raise ImportError("pip install gliner2 — required for GLiNER2 entity extraction")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading GLiNER2 large model on %s...", device)
    model = GLiNER2.from_pretrained("fastino/gliner2-large-v1")
    if device == "cuda":
        model = model.to(device)
    _MODEL_CACHE["gliner2"] = model
    return model


def _load_gliner_v1():
    """Load GLiNER v1 medium model (faster, cached after first load)."""
    if "gliner_v1" in _MODEL_CACHE:
        return _MODEL_CACHE["gliner_v1"]
    try:
        from gliner import GLiNER
    except ImportError:
        raise ImportError("pip install gliner — required for GLiNER entity extraction")
    logger.info("Loading GLiNER v1 medium model...")
    model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
    _MODEL_CACHE["gliner_v1"] = model
    return model


def extract_entities_gliner(tree: DocumentTree) -> DocumentGraph:
    """Extract entities using GLiNER zero-shot NER + spaCy dependency parsing for relationships.

    Auto-selects the best model for the hardware:
    - GPU available → GLiNER2 large (deberta-v3-large, best quality, batch_size=16)
    - CPU only → GLiNER v1 medium (deberta-v3-base, 4x faster, batch_size=4)

    Both use domain-adaptive labels, one chunk per node, and batch inference.
    """
    all_nodes = list(iter_nodes(tree.structure))
    full_text = " ".join(n.text or "" for n in all_nodes[:5])

    # Detect domain from doc name + content, get labels
    domain = tree.domain or _detect_domain(full_text, doc_name=tree.doc_name)
    if not tree.domain:
        tree.domain = domain  # Tag the tree for downstream use
    labels = DOMAIN_LABELS.get(domain, DOMAIN_LABELS["generic"])

    # Auto-select model: GLiNER2 large on GPU, GLiNER v1 medium on CPU
    import torch
    use_v2 = torch.cuda.is_available()

    # Try GLiNER2 first if available (even on CPU, user may prefer quality)
    if use_v2:
        try:
            model = _load_gliner2()
            model_version = "v2"
        except ImportError:
            model = _load_gliner_v1()
            model_version = "v1"
    else:
        # CPU: prefer v1 for speed, fall back to v2 if v1 not installed
        try:
            model = _load_gliner_v1()
            model_version = "v1"
        except ImportError:
            model = _load_gliner2()
            model_version = "v2"

    on_gpu = use_v2 and next(model.parameters()).is_cuda

    # One chunk per node — keeps it simple and fast
    _MAX_TEXT = 4096 if model_version == "v2" else 512

    chunks: list[str] = []
    chunk_meta: list[tuple] = []    # (node_index, node_id, chunk_start)

    for ni_idx, node in enumerate(all_nodes):
        text = node.text or ""
        if len(text) < 20:
            continue
        # For v1 with smaller context, use multiple chunks per node
        if model_version == "v1":
            for cs in range(0, min(len(text), 10000), 500):
                chunk = text[cs:cs + 512]
                if len(chunk) >= 20:
                    chunks.append(chunk)
                    chunk_meta.append((ni_idx, node.node_id, cs))
        else:
            chunks.append(text[:_MAX_TEXT])
            chunk_meta.append((ni_idx, node.node_id, 0))

    batch_size = 16 if on_gpu else 4
    threshold = 0.3 if model_version == "v2" else 0.4

    logger.info(
        "GLiNER %s: %d chunks from %d nodes (batch_size=%d, device=%s)",
        model_version, len(chunks), len(all_nodes), batch_size,
        "cuda" if on_gpu else "cpu",
    )

    entity_mentions: dict[str, dict] = {}

    if model_version == "v2":
        # GLiNER2 batch API
        try:
            batch_results = model.batch_extract_entities(
                chunks, labels, batch_size=batch_size, threshold=threshold,
            )
        except Exception:
            logger.warning("Batch extraction failed, falling back to sequential", exc_info=True)
            batch_results = []
            for chunk in chunks:
                try:
                    batch_results.append(model.extract_entities(chunk, labels, threshold=threshold))
                except Exception:
                    batch_results.append({})
    else:
        # GLiNER v1 sequential API (already fast at ~56ms/call)
        batch_results = []
        for chunk in chunks:
            try:
                ents = model.predict_entities(chunk, labels, threshold=threshold)
                # Convert v1 list format to v2 dict format for uniform processing
                d: dict[str, list[str]] = {}
                for e in ents:
                    d.setdefault(e["label"], []).append(e["text"])
                batch_results.append({"entities": d})
            except Exception:
                batch_results.append({})

    # Process results
    for idx, result in enumerate(batch_results):
        _, node_id, chunk_start = chunk_meta[idx]
        node = all_nodes[chunk_meta[idx][0]]
        text = node.text or ""

        entities_dict = result if isinstance(result, dict) else {}
        if "entities" in entities_dict:
            entities_dict = entities_dict["entities"]

        for label, names in entities_dict.items():
            if not isinstance(names, list):
                continue
            for name in names:
                if not isinstance(name, str):
                    continue
                name = name.strip()
                if len(name) < 2 or len(name) > 80:
                    continue

                key = name.lower()
                if key not in entity_mentions:
                    entity_mentions[key] = {
                        "name": name,
                        "type": label,
                        "descriptions": set(),
                        "node_ids": set(),
                    }
                entity_mentions[key]["node_ids"].add(node_id)

                # Get sentence context as description
                search_start = max(0, chunk_start - 50)
                idx_in_text = text.find(name, search_start)
                if idx_in_text >= 0:
                    start = max(0, text.rfind(".", 0, idx_in_text) + 1)
                    end = text.find(".", idx_in_text + len(name))
                    if end < 0:
                        end = min(len(text), idx_in_text + 150)
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
        "GLiNER2 extraction (%s domain): %d entities, %d relationships from %d nodes",
        domain, len(entities), len(relationships), len(all_nodes),
    )

    return DocumentGraph(doc_name=tree.doc_name, entities=entities, relationships=relationships)


def extract_entities_gliner_v1(tree: DocumentTree, *, skip_relationships: bool = False) -> DocumentGraph:
    """Fast extraction using GLiNER v1 medium — ~56ms/call, ~20s per 150-node doc.

    Use this for bulk graph building. Optionally skip spaCy relationships for speed.
    """
    all_nodes = list(iter_nodes(tree.structure))
    full_text = " ".join(n.text or "" for n in all_nodes[:5])

    domain = tree.domain or _detect_domain(full_text, doc_name=tree.doc_name)
    if not tree.domain:
        tree.domain = domain
    labels = DOMAIN_LABELS.get(domain, DOMAIN_LABELS["generic"])

    model = _load_gliner_v1()

    _CHUNK_SIZE = 512
    _CHUNK_STEP = 500
    _MAX_TEXT = 10000

    entity_mentions: dict[str, dict] = {}

    for node in all_nodes:
        text = node.text or ""
        if len(text) < 20:
            continue

        for chunk_start in range(0, min(len(text), _MAX_TEXT), _CHUNK_STEP):
            chunk = text[chunk_start:chunk_start + _CHUNK_SIZE]
            if len(chunk) < 20:
                continue

            try:
                ents = model.predict_entities(chunk, labels, threshold=0.4)
            except Exception:
                continue

            for ent in ents:
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

                idx = text.find(name, max(0, chunk_start - 50))
                if idx >= 0:
                    start = max(0, text.rfind(".", 0, idx) + 1)
                    end = text.find(".", idx + len(name))
                    if end < 0:
                        end = min(len(text), idx + 150)
                    sent = text[start:end].strip()
                    if len(sent) < 200:
                        entity_mentions[key]["descriptions"].add(sent)

    # Relationships (optional — skip for bulk builds)
    if skip_relationships:
        relationships = []
    else:
        relationships = _extract_relationships_spacy(all_nodes, entity_mentions)

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
        "GLiNER v1 extraction (%s domain): %d entities, %d relationships from %d nodes",
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
