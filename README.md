# NanoIndex

**Turn any PDF into a searchable tree. Ask questions, get answers with page citations.**

NanoIndex reads your document, understands its structure (headings, sections, tables, figures), and builds a tree you can search with plain English. No vector databases. No chunk tuning. No embeddings.

Built on [Nanonets OCR-3](https://nanonets.com) for extraction. Fully open source.

---

## Get Started in 3 Steps

### 1. Install

```bash
pip install nanoindex
```

### 2. Get your free API key

Go to [docstrange.nanonets.com/app](https://docstrange.nanonets.com/app) and create an API key.

Extract first 10,000 pages for free. After that, $0.01 per page.

```bash
export NANONETS_API_KEY=your_key_here
```

### 3. Ask a question

```python
import nanoindex

tree = nanoindex.index("report.pdf")
answer = nanoindex.ask("What was the revenue?", tree)

print(answer.content)
print(answer.citations)  # page numbers + section references
```

That's it. Three lines to go from PDF to cited answer.

---

## What Happens Under the Hood

```
Your PDF
   |
   v
Nanonets OCR-3 reads the document (1 API call)
   |  Extracts: text, headings, tables, page layouts, bounding boxes
   v
Tree builder creates a navigable index (zero LLM calls)
   |  Sections become tree nodes with titles, summaries, page ranges
   v
You ask a question
   |
   v
LLM searches the tree, finds the right sections, gives you an answer
   |  with page numbers and citations
   v
Done.
```

---

## What Can You Do With It

### Ask questions about documents

```python
answer = nanoindex.ask("What was Q3 gross margin?", tree)
print(answer.content)    # "Gross margin was 42.3% in Q3..."
print(answer.citations)  # [Citation(title="Income Statement", pages=[45, 46])]
```

### Use the command line

```bash
nanoindex ask report.pdf "What was the revenue?"
nanoindex index report.pdf -o tree.json
nanoindex search tree.json "capital expenditure"
```

### Save and reuse trees

```python
from nanoindex.utils.tree_ops import save_tree, load_tree

save_tree(tree, "my_tree.json")
tree = load_tree("my_tree.json")  # instant, no re-extraction
```

### Visualize your tree

```python
nanoindex.visualize(tree)  # opens interactive dashboard in browser
```

Or from the command line:

```bash
nanoindex viz tree.json
```

### Use any LLM for answering

NanoIndex uses Nanonets OCR-3 for extraction but you can use any LLM for the reasoning step:

```python
from nanoindex import NanoIndex

ni = NanoIndex(
    nanonets_api_key="your_key",
    reasoning_llm_model="gpt-4o",
    reasoning_llm_api_key="sk-...",
    reasoning_llm_base_url="https://api.openai.com/v1",
)

tree = ni.index("report.pdf")
answer = ni.ask("What was the revenue?", tree, mode="agentic_vision")
```

Works with OpenAI, Anthropic, Ollama, or any OpenAI-compatible endpoint.

### Search across multiple documents

```python
from nanoindex import NanoIndex, DocumentStore

ni = NanoIndex(nanonets_api_key="...")
store = DocumentStore()

for pdf in ["q1.pdf", "q2.pdf", "q3.pdf"]:
    tree = ni.index(pdf)
    store.add(tree)

answer = ni.multi_ask("Compare revenue across quarters", store)
```

---

## Retrieval Modes

| Mode | What it does | Best for |
|---|---|---|
| `text` | LLM searches the tree, answers from text | Simple questions |
| `vision` | Same, but also sees page images | Charts, figures, layouts |
| `agentic_vision` | Multi-round search with page images | Complex questions, highest accuracy |
| `fast` | Embedding pre-filter + entity graph, then LLM | High volume, 3x cheaper |

```python
# Simple and fast
answer = ni.ask("What was revenue?", tree, mode="text")

# Best accuracy
answer = ni.ask("What was revenue?", tree, mode="agentic_vision", pdf_path="report.pdf")

# Cheapest at scale
answer = ni.ask("What was revenue?", tree, mode="fast")
```

---

## Fast Mode: Entity Graph

For high-volume use, NanoIndex can build an entity graph alongside the tree. This pre-filters candidates before the LLM sees anything, cutting costs by 3x.

```python
# Build graph + embeddings (one-time, at index time)
graph = await ni.async_build_graph(tree)
embeddings = await ni.async_build_embeddings(tree)

# Queries are now 3x cheaper
answer = ni.ask("What was revenue?", tree, mode="fast")
```

The graph extracts entities (companies, metrics, dates, people) and their relationships from every section. At query time: embedding search finds candidate nodes, the graph expands to related sections, and the LLM only reads 20 nodes instead of 300.

---

## Open-Source Mode (No API Key for Parsing)

NanoIndex can also work without Nanonets OCR-3, using PyMuPDF for extraction:

```python
ni = NanoIndex(parser="pymupdf")
tree = ni.index("report.pdf")  # no API key needed for parsing
```

PyMuPDF gives you basic text and table extraction. The tree will be simpler (no heading detection, no hierarchy), but it works for quick experiments. For production quality, use Nanonets OCR-3.

---

## Benchmarks

Tested on real-world documents with LLM-as-judge evaluation:

| Benchmark | Documents | Avg Pages | Accuracy |
|---|---|---|---|
| FinanceBench (SEC 10-K filings) | 84 | 143 | **94.5%** |
| DocBench Legal (court filings, legislation) | 51 | 54 | **96.0%** |

Evidence page retrieval accuracy: **93.3%**

---

## How It Compares

| | Traditional RAG | NanoIndex |
|---|---|---|
| Indexing | Chunk + embed + vector DB | Extract + build tree |
| Retrieval | Similarity search | LLM reasons over structure |
| Tables | Poorly handled | Natively extracted |
| Figures | Not supported | Vision mode |
| Scanned docs | Needs separate OCR | Built-in |
| Structure-aware | No | Yes |
| Citations | Approximate | Exact page + bounding box |

---

## Project Structure

```
nanoindex/
  __init__.py         # Public API: index(), search(), ask(), visualize()
  cli.py              # Command line interface
  models.py           # Data models (TreeNode, DocumentTree, Entity, etc.)
  config.py           # Configuration

  core/
    extractor.py      # Nanonets OCR-3 extraction
    tree_builder.py   # Deterministic tree construction
    enricher.py       # LLM summary generation
    retriever.py      # Tree search
    generator.py      # Answer generation (text + vision)
    agentic.py        # Multi-round retrieval
    fast_retriever.py # Embedding + graph retrieval
    entity_extractor.py   # Entity/relationship extraction
    graph_builder.py      # NetworkX graph operations
    embedder.py           # Node embedding + cosine search
    parsers/              # Pluggable parsers (nanonets, pymupdf)
    modal_processors/     # Image + table processing for graph

  utils/
    tree_ops.py       # Save, load, traverse trees
    tokens.py         # Token counting
    pdf.py            # PDF page rendering

viz/                  # Interactive visualization dashboard (Next.js)
examples/             # Usage examples
benchmarks/           # Evaluation scripts
```

---

## Development

```bash
git clone https://github.com/nanonets/nanoindex.git
cd nanoindex
pip install -e ".[dev]"
pytest
```

---

## License

MIT
