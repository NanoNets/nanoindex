"""Detect contradictions across documents in a Knowledge Base."""

from nanoindex.models import DocumentGraph


def find_contradictions(graphs: dict[str, DocumentGraph]) -> list[dict]:
    """Find entities with conflicting descriptions across documents."""
    # Collect all descriptions per entity across all docs
    entity_descriptions: dict[str, list[tuple[str, str]]] = {}

    for doc_name, graph in graphs.items():
        for entity in graph.entities:
            key = entity.name.lower()
            if key not in entity_descriptions:
                entity_descriptions[key] = []
            if entity.description:
                entity_descriptions[key].append((doc_name, entity.description))

    contradictions = []
    for entity_key, descriptions in entity_descriptions.items():
        if len(descriptions) < 2:
            continue

        # Check for numeric contradictions
        import re
        numbers_per_doc: dict[str, list[str]] = {}
        for doc, desc in descriptions:
            nums = re.findall(r'\$?[\d,]+\.?\d*%?', desc)
            if nums:
                numbers_per_doc[doc] = nums

        if len(numbers_per_doc) >= 2:
            # Check if different docs report different numbers
            num_sets = [set(nums) for nums in numbers_per_doc.values()]
            # If any two docs have non-identical number sets, flag as discrepancy
            has_discrepancy = any(
                num_sets[i] != num_sets[j]
                for i in range(len(num_sets))
                for j in range(i + 1, len(num_sets))
            )
            if has_discrepancy:
                contradictions.append({
                    "entity": entity_key,
                    "type": "numeric_discrepancy",
                    "descriptions": descriptions,
                    "numbers": numbers_per_doc,
                })

    return contradictions
