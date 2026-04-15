<div align="center">

<img src="assets/Nanoindex.png" alt="NanoIndex" width="200"/>

# NanoIndex

**Open-source agentic harness for long documents.**
**Self-validating trees. Entity graphs. Karpathy-inspired LLM wikis. Cited answers down to the pixel.**

<p>
  <a href="https://pypi.org/project/nanoindex/"><img src="https://img.shields.io/pypi/v/nanoindex?style=for-the-badge&logo=pypi&logoColor=white&label=PyPI" /></a>
  <a href="https://github.com/NanoNets/nanoindex/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/NanoNets/nanoindex/ci.yml?style=for-the-badge&logo=githubactions&logoColor=white&label=CI" /></a>
  <a href="https://pypi.org/project/nanoindex/"><img src="https://img.shields.io/pypi/pyversions/nanoindex?style=for-the-badge&logo=python&logoColor=white" /></a>
  <a href="https://pepy.tech/project/nanoindex"><img src="https://img.shields.io/pepy/dt/nanoindex?style=for-the-badge&logo=pypi&logoColor=white&label=Downloads" /></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-20C997?style=for-the-badge" /></a>
</p>
<p>
  <a href="https://nanonets.com/research/nanonets-ocr-3"><img src="https://img.shields.io/badge/Built%20on-Nanonets%20OCR--3-546FFF?style=for-the-badge" /></a>
</p>

<p>
  <a href="https://docstrange.nanonets.com/app"><img src="https://img.shields.io/badge/Get%20API%20Key-Free%2010K%20Pages-FCC419?style=for-the-badge" /></a>
  <a href="https://colab.research.google.com/github/NanoNets/nanoindex/blob/main/examples/nanoindex_quickstart.ipynb"><img src="https://img.shields.io/badge/Try%20in-Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" /></a>
</p>

| Benchmark | Accuracy |
|---|---|
| **FinanceBench** (84 SEC filings, avg 143 pages) | **95%** |
| **DocBench Legal** (51 court filings, avg 54 pages) | **96%** |

If NanoIndex is useful, a ⭐ helps others find it.

<p align="center">
  <img src="assets/hero.gif" alt="NanoIndex" width="900"/>
</p>

</div>

---

## The problem

Most RAG systems chop documents into chunks and turn them into embeddings. Two things break.

**Structure is lost.** A 200-page filing has a table of contents, numbered sections, tables with rows and columns. Chunking throws all of that away. Section 3.2 is no longer inside Section 3. A balance sheet table gets split across two chunks. The hierarchy the author wrote is gone.

**Multi-hop questions fail.** Many real questions need data from multiple sections. Computing a ratio requires the income statement and the balance sheet. Checking a legal clause means reading the clause, its definitions, and its exceptions. A chunk retriever finds one section, not the three you need, because the question doesn't match all of them equally in embedding space.

The result: wrong answers with citations that say "chunk_47" instead of a page and location an auditor can verify.

---

## Who is this for?

- Developers building RAG over long, structured documents (10-Ks, contracts, medical records)
- Teams where citation accuracy is a compliance or audit requirement
- Anyone hitting the limits of chunk-and-embed on multi-section documents

**Not the right fit if:** you're querying short documents (<10 pages) or need sub-second latency.

---

## Part 1: Querying within a single long document

NanoIndex preserves document structure instead of destroying it. [Nanonets OCR-3](https://nanonets.com/research/nanonets-ocr-3) extracts the table of contents, section hierarchy, and heading structure. NanoIndex builds a tree from these.

<p align="center">
  <img src="assets/light-loop.gif" alt="NanoIndex Pipeline" width="900"/>
</p>

| Document type | Examples | How NanoIndex navigates |
|---|---|---|
| **Structured** | 10-K filings, contracts, research papers | Uses the table of contents. Agent reads the outline, goes straight to the right section. |
| **Semi-structured** | Earnings releases, quarterly reports | Disambiguates repetitive headings ("Reconciliation" x8 becomes "Reconciliation: Q2 2023 Segment Data"). |
| **Unstructured** | Transcripts, scans, flat reports | Splits by page, extracts entities (people, companies, dates, amounts). The entity graph becomes the map. |

When you ask a question, an LLM agent navigates this tree across multiple rounds. It reads page images directly. It verifies its calculations. It cites every answer with the exact page and pixel coordinates.

### Quick start

```bash
pip install nanoindex
```

```bash
export NANONETS_API_KEY=your_key    # free at docstrange.nanonets.com (10K pages)
export ANTHROPIC_API_KEY=your_key   # or OPENAI_API_KEY, GOOGLE_API_KEY
```

```python
from nanoindex import NanoIndex

# Pick your LLM
ni = NanoIndex(llm="anthropic:claude-sonnet-4-6")
# ni = NanoIndex(llm="openai:gpt-5.4")
# ni = NanoIndex(llm="gemini:gemini-2.5-flash")
# ni = NanoIndex(llm="ollama:llama3")  # fully local

# Index a document
tree = ni.index("10k_filing.pdf")
answer = ni.ask("What was the free cash flow?", tree)

print(answer.content)                     # computed answer with reasoning
print(answer.citations[0].pages)          # [52]
print(answer.citations[0].bounding_boxes) # exact coordinates on the page
```

### Build entity graph (optional)

By default, `index()` builds only the tree. To also extract entities and relationships:

```python
ni = NanoIndex(llm="anthropic:claude-sonnet-4-6", build_graph=True)
tree = ni.index("10k_filing.pdf")  # tree + entity graph
graph = ni.get_graph(tree)         # 921 entities, 103 relationships
```

The entity graph enables `fast_vision` and `agentic_graph_vision` modes. Without it, `agentic_vision` (the default) works fine using tree navigation alone.

### Save and reload trees

Index once, query many times. Trees and graphs are JSON files you can save and load:

```python
from nanoindex.utils.tree_ops import save_tree, load_tree, load_graph

# Save after indexing
save_tree(tree, "3M_2018_10K.json")

# Load later - no re-indexing needed
tree = load_tree("3M_2018_10K.json")
graph = load_graph("3M_2018_10K_graph.json")
answer = ni.ask("What was the operating margin?", tree)
```

### Query modes

| Mode | LLM calls | Best for |
|---|---|---|
| `agentic_vision` (default) | 5-8 | Highest accuracy. Agent navigates tree, reads page images. |
| `agentic_graph_vision` | 4-6 | Entity graph seeds the search, agent reasons from there. |
| `fast_vision` | 2 | Simple fact lookups. Cheapest. |

---

## Part 2: Querying across multiple documents (Karpathy-inspired wiki)

The harder problem is synthesis across documents: "How has 3M's revenue changed over 5 years?" or "Which company in my portfolio has the highest ROA?"

Inspired by [Karpathy's LLM wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f), NanoIndex compiles documents into a persistent, interlinked wiki that gets richer with every source you add and every question you ask.

```python
from nanoindex.kb import KnowledgeBase

kb = KnowledgeBase("./sec-filings")
kb.add("3M_2018_10K.pdf")     # extracts entities, builds concept pages
kb.add("3M_2019_10K.pdf")     # updates existing concepts, flags changes
kb.add("3M_2020_10K.pdf")     # cross-references across all three years

answer = kb.ask("How has 3M's revenue changed from 2018 to 2020?")
kb.lint()  # find contradictions, stale claims, orphan pages
```

Add pre-built trees and graphs directly:

```python
from nanoindex.utils.tree_ops import load_tree, load_graph

tree = load_tree("3M_2018_10K.json")
graph = load_graph("3M_2018_10K_graph.json")
kb.add_tree(tree, graph)
```

The wiki is a directory of markdown files. Open it in Obsidian and browse concept pages with `[[backlinks]]`, entity graphs, and an activity log.

Three layers:
- **Raw sources** - your PDFs, immutable, never modified
- **The wiki** - markdown pages with cross-references. The LLM writes and maintains all of it.
- **The schema** - how the wiki is structured, what entity types to track, domain conventions

---

## How it compares

| | Chunk + Embed | Microsoft GraphRAG | PageIndex | **NanoIndex** |
|---|---|---|---|---|
| **Indexing** | Chunk text, embed | LLM per chunk | LLM per page | 1 OCR API call |
| **Structure** | Lost | Lost | Tree | Tree + entity graph |
| **Navigation** | Similarity search | Map-reduce | LLM tree walk | Multi-round agent |
| **Multi-document** | Vector DB | No | No | Wiki with [[backlinks]] |
| **Citations** | Chunk ID | None | Page number | Pixel coordinates |
| **Vision** | No | No | No | Page images to LLM |
| **Cost per doc** | Low | High | High | Low |

---

## Roadmap

- [ ] **Agentic extraction** self-correcting structured extraction for tables and forms (invoice line items, insurance loss runs, bank statement reconciliation)
- [ ] **Real-world long document benchmarks** bank statement reconciliation, insurance loss run extraction, multi-document contract analysis
- [ ] **Streaming tree building** real-time tree construction as pages are parsed
- [ ] **Multi-agent wiki** multiple agents maintaining different sections of the wiki concurrently

---

## CLI

```bash
nanoindex index report.pdf -o tree.json
nanoindex ask report.pdf "What was the revenue?"
nanoindex viz tree.json
```

## Development

```bash
git clone https://github.com/nanonets/nanoindex.git && cd nanoindex
uv sync --extra dev && uv run pytest    # or: pip install -e ".[dev]" && pytest
```

Entity extraction: `pip install nanoindex[gliner]` (CPU) or `pip install nanoindex[gliner-gpu]` (GPU).

---

<div align="center">
  <img src="assets/social-card.gif" alt="NanoIndex" width="800"/>
</div>

Apache 2.0. Built on [Nanonets OCR-3](https://nanonets.com/research/nanonets-ocr-3).
