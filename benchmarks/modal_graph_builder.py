"""Build GLiNER2 entity graphs on Modal GPU — parallel across 96 docs.

Usage:
    modal run benchmarks/modal_graph_builder.py
    modal run benchmarks/modal_graph_builder.py --doc-names "3M_2018_10K,AMAZON_2019_10K"
"""

import modal
import json
import time
from pathlib import Path

app = modal.App("nanoindex-graph-builder")

gliner_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "pydantic>=2.0",
        "networkx>=3.0",
        "httpx>=0.27.0",
        "tiktoken>=0.7.0",
        "python-dotenv>=1.0",
        "pyyaml>=6.0",
        "openai>=1.50.0",
        "anthropic>=0.30.0",
        "gliner2>=0.1.0",
        "spacy>=3.7.0",
        "torch>=2.0",
    )
    .run_commands("python -m spacy download en_core_web_sm")
    .run_commands(
        "python -c \"from gliner2 import GLiNER2; GLiNER2.from_pretrained('fastino/gliner2-large-v1')\""
    )
    .add_local_python_source("nanoindex")
)


@app.function(
    image=gliner_image,
    gpu="T4",
    timeout=1800,
)
def build_graph(doc_name: str, tree_json: str) -> dict:
    """Build GLiNER2 graph for one doc on GPU. Tree JSON passed directly — no volume needed."""
    import logging
    logging.basicConfig(level=logging.INFO)

    from nanoindex.models import DocumentTree
    from nanoindex.core.gliner_extractor import extract_entities_gliner
    from nanoindex.core.entity_resolver import resolve_entities
    from nanoindex.core.tree_validator import validate_tree
    from nanoindex.utils.tree_ops import iter_nodes

    tree = DocumentTree(**json.loads(tree_json))
    node_count = len(list(iter_nodes(tree.structure)))
    validation = validate_tree(tree)

    t0 = time.time()
    graph = extract_entities_gliner(tree)
    gliner_time = time.time() - t0

    before = len(graph.entities)
    graph = resolve_entities(graph)
    total_time = time.time() - t0

    return {
        "doc_name": doc_name,
        "graph_json": graph.model_dump(),
        "stats": {
            "nodes": node_count,
            "entities_raw": before,
            "entities_resolved": len(graph.entities),
            "relationships": len(graph.relationships),
            "gliner_time_s": round(gliner_time, 1),
            "total_time_s": round(total_time, 1),
            "validation_passed": validation.passed,
            "page_coverage": validation.stats.get("page_coverage", 0),
        },
    }


@app.local_entrypoint()
def main(doc_names: str = ""):
    tree_dir = Path("benchmarks/cache_v3")
    graph_dir = Path("benchmarks/graphs_v4")
    graph_dir.mkdir(parents=True, exist_ok=True)

    if doc_names:
        names = [n.strip() for n in doc_names.split(",")]
    else:
        names = sorted(f.stem for f in tree_dir.glob("*.json"))

    existing = {f.stem for f in graph_dir.glob("*.json") if f.stem != "_build_stats"}
    to_build = [n for n in names if n not in existing]

    print(f"Total: {len(names)}, Cached: {len(existing)}, To build: {len(to_build)}")
    if not to_build:
        print("All graphs already built!")
        return

    # Read all trees into memory — pass directly to workers (no volume needed)
    inputs = []
    for name in to_build:
        tree_json = (tree_dir / f"{name}.json").read_text()
        inputs.append((name, tree_json))

    print(f"Building {len(to_build)} graphs on Modal T4 GPUs...")
    t0 = time.time()
    all_stats = []

    for result in build_graph.starmap(inputs):
        doc_name = result["doc_name"]
        stats = result["stats"]
        all_stats.append(stats)

        with open(graph_dir / f"{doc_name}.json", "w") as f:
            json.dump(result["graph_json"], f)

        print(
            f"  [{len(all_stats)}/{len(to_build)}] {doc_name}: "
            f"{stats['entities_resolved']}e/{stats['relationships']}r "
            f"in {stats['total_time_s']}s ({stats['nodes']}n)"
        )

    total_wall = time.time() - t0
    total_compute = sum(s["total_time_s"] for s in all_stats)

    print(f"\n{'='*60}")
    print(f"  Graph Building Complete")
    print(f"{'='*60}")
    print(f"  Documents:           {len(all_stats)}")
    print(f"  Wall time:           {total_wall:.0f}s ({total_wall/60:.1f}min)")
    print(f"  Total GPU compute:   {total_compute:.0f}s ({total_compute/60:.1f}min)")
    print(f"  Avg per doc:         {total_compute/len(all_stats):.1f}s")
    print(f"  Parallelism:         {total_compute/max(total_wall,1):.1f}x")
    print(f"  Total entities:      {sum(s['entities_resolved'] for s in all_stats)}")
    print(f"  Total relationships: {sum(s['relationships'] for s in all_stats)}")
    print(f"{'='*60}")

    with open(graph_dir / "_build_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)
