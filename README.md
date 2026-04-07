<div align="center">

<img src="assets/Nanoindex.png" alt="NanoIndex" width="200"/>

# NanoIndex

**Turn any PDF into a tree you can reason over.**

<p>
  <a href="https://pypi.org/project/nanoindex/"><img src="https://img.shields.io/pypi/v/nanoindex?style=for-the-badge&logo=pypi&logoColor=white&label=PyPI" /></a>
  <a href="https://github.com/nanonets/nanoindex"><img src="https://img.shields.io/badge/GitHub-NanoIndex-181717?style=for-the-badge&logo=github&logoColor=white" /></a>
  <a href="https://nanonets.com/research/nanonets-ocr-3"><img src="https://img.shields.io/badge/Built%20on-Nanonets%20OCR--3-546FFF?style=for-the-badge" /></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-20C997?style=for-the-badge" /></a>
</p>

<p>
  <a href="https://docstrange.nanonets.com/app"><img src="https://img.shields.io/badge/Get%20API%20Key-Free%2010K%20Pages-FCC419?style=for-the-badge" /></a>
  <a href="https://colab.research.google.com/github/NanoNets/nanoindex/blob/main/examples/nanoindex_quickstart.ipynb"><img src="https://img.shields.io/badge/Try%20in-Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" /></a>
</p>

| Benchmark | Docs | Avg Pages | Accuracy |
|---|---|---|---|
| **FinanceBench** (SEC 10-K, 10-Q, 8-K) | 84 | 143 | **94.5%** |
| **DocBench Legal** (court filings, legislation) | 51 | 54 | **96.0%** |

</div>

---

Here's the problem with RAG: it doesn't read documents. It chops them into pieces, turns those pieces into vectors, and prays the right piece is close enough to your question in embedding space.

That works for "what color is the logo?" It fails for "compute the free cash flow conversion ratio from the income statement and cash flow statement on pages 47 and 52."

NanoIndex takes a different approach. It reads the document the way you would.

## What it does

**1. Builds a tree.** NanoIndex parses the PDF and preserves its structure. Section 3.2 is inside Section 3. The income statement is a child of the financial statements section. Tables keep their rows and columns. The tree validates itself: if pages are missing or sections are empty, you know before you ask a single question.

**2. Builds an entity graph.** Using GLiNER (a zero-shot NER model that runs locally), it extracts every entity in the document: companies, people, dollar amounts, dates, legal clauses, medical terms. Then it links them. "Tim Cook" is the CEO of "Apple." "Revenue $394B" belongs to "FY2023." No LLM calls. No training data. Just a local model that knows what matters in your document type.

**3. Connects them.** Every entity maps to the tree nodes where it appears. Every node lists the entities it contains. The tree gives you structure. The graph gives you semantics. Together, they form a single navigable index that an LLM agent can explore.

**4. Lets agents reason over it.** When you ask a question, an LLM agent navigates this index in multiple rounds. It decomposes the question. It picks sections from the tree. It reads them (from rendered page images, not OCR text, so tables look exactly as printed). If it needs more data, it follows entity relationships to jump to related sections. It verifies its own calculations. Then it cites the exact page and pixel coordinates where each answer lives.

## Quick Start

```bash
pip install nanoindex
```

You need two API keys:

```bash
export NANONETS_API_KEY=...    # document parsing — free 10K pages at docstrange.nanonets.com
export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY, GOOGLE_API_KEY
```

Then:

```python
from nanoindex import NanoIndex

ni = NanoIndex()
tree = ni.index("annual_report.pdf")
answer = ni.ask("What was the free cash flow conversion ratio?", tree)

print(answer.content)
# "FCF conversion = 154.7%. Operating cash flow was $6,439M,
#  CapEx was $1,577M, giving FCF of $4,862M. Net income was
#  $3,143M. FCF/Net Income = 154.7%."

print(answer.citations[0].pages)          # [52]
print(answer.citations[0].bounding_boxes) # [BoundingBox(page=52, x=234, y=567, ...)]
```

That's it. Tree + graph + agent + citations. Five lines.

## How it thinks

Give NanoIndex a 200-page 10-K filing and ask: "Is free cash flow conversion improving?"

Here's what happens:

```
Agent: I need FCF and net income for multiple years.
       FCF = Operating Cash Flow - CapEx.
       Let me find the cash flow statement and income statement.

Tree:  Node 0024.0020 "Consolidated Statement of Income" (pp. 47-48)
       Node 0024.0022 "Consolidated Statement of Cash Flows" (pp. 51-52)

Agent: [reads page images of pp. 47-48, 51-52]
       FY2022: OCF $6,439M - CapEx $1,577M = FCF $4,862M
       Net Income $3,143M → FCF Conversion = 154.7%

       I need prior year too. Let me check the same statements.
       FY2021: OCF $5,984M - CapEx $1,373M = FCF $4,611M
       Net Income $3,193M → FCF Conversion = 144.4%

Agent: 154.7% vs 144.4% — yes, improving by ~10 points.

Citations: page 47 (income statement), page 52 (cash flow statement)
           with bounding boxes for each number used.
```

The agent reads page images directly, not extracted text. No OCR errors. No table parsing failures. It sees the document exactly as you would.

## The entity graph

Every document gets an entity graph automatically. No expensive LLM calls — just a local NER model that knows what to look for.

```python
graph = ni.get_graph(tree)
# 921 entities, 103 relationships

# What's in this document?
for e in graph.entities[:5]:
    print(f"  [{e.entity_type}] {e.name}")
    # [Company] 3M
    # [Revenue] $32,765 million
    # [FiscalYear] 2018
    # [ExecutiveName] Michael Roman
    # [BusinessSegment] Safety and Industrial
```

The system auto-detects the document type and loads the right entity labels:

| Domain | Entity types |
|---|---|
| **SEC filings** | Company, Revenue, NetIncome, EPS, BusinessSegment, FiscalYear, Acquisition, Restructuring... |
| **Legal** | Party, Court, CaseNumber, Statute, Jurisdiction, Damages, Judge, Attorney... |
| **Medical** | Patient, Diagnosis, Drug, Procedure, Dosage, Symptom, LabTest... |
| **Insurance** | Insurer, PolicyNumber, ClaimNumber, CoverageType, Premium, LossAmount... |

Or pass your own labels. GLiNER extracts whatever entity types you define, zero-shot, no training.

## Query modes

| Mode | What it does | When to use it |
|---|---|---|
| `agentic_vision` | Agent navigates tree, reads page images | Best accuracy. Default. |
| `agentic_graph_vision` | Graph seeds the agent, then reasons + expands | Faster. Almost as accurate. |
| `fast_vision` | Graph lookup, no agent loop | Cheapest. Simple lookups. |
| `global` | Map-reduce over entity communities | "What are the main themes?" |

## What it knows about finance

NanoIndex ships with a knowledge base of 31 financial formulas (ROA, ROE, FCF, EBITDA, working capital, effective tax rate...) with strict conventions:

- "FCF = Operating Cash Flow minus CapEx. Not just OCF."
- "Tax rates can be negative. Preserve the sign."
- "Use D&A from the cash flow statement for EBITDA."

When you ask a financial question, the relevant formulas are injected into the agent's prompt with enforcement language. The agent doesn't invent formulas. It uses the canonical ones.

## Self-validating trees

Before you ask a single question, NanoIndex validates the tree:

- Does the tree cover every page in the document?
- Are there empty nodes? Duplicate IDs?
- Is the hierarchy deep enough, or is it suspiciously flat?
- Are bounding boxes present for citations?

If the tree is broken, you know immediately. Not after you've built a pipeline on top of it.

## Pixel-level citations

Every answer comes with bounding boxes:

```python
for c in answer.citations:
    for bb in c.bounding_boxes:
        print(f"Page {bb.page}: ({bb.x}, {bb.y}) — '{bb.text}'")
```

This isn't "see page 47." It's "this specific number is at coordinates (234, 567) on page 47, 89 pixels wide." You can draw a box around the exact cell in the exact table.

## Pick your LLM

```python
ni = NanoIndex(llm="anthropic:claude-sonnet-4-6")
ni = NanoIndex(llm="openai:gpt-5.4")
ni = NanoIndex(llm="gemini:gemini-2.5-flash")
ni = NanoIndex(llm="groq:llama-3.3-70b-versatile")
```

## Fully local mode

No API keys at all:

```python
ni = NanoIndex(parser="pymupdf", llm="ollama:llama3")
tree = ni.index("report.pdf")  # everything runs locally
```

## CLI

```bash
nanoindex index report.pdf -o tree.json
nanoindex ask report.pdf "What was the revenue?"
nanoindex viz tree.json                           # opens D3 graph viewer
```

## Benchmarks

**FinanceBench** (150 questions, 84 SEC filings, avg 143 pages):

| Mode | Accuracy | Evidence Page Hit | Avg Time |
|---|---|---|---|
| `agentic_vision` | **94.5%** | 93.3% | ~45s |
| `agentic_graph_vision` | 85.3% | 90.0% | ~67s |
| `fast_vision` | 77.3% | 82.7% | ~17s |

**DocBench Legal** (51 documents, avg 54 pages): **96.0%**

96 entity graphs across the benchmark: 68,045 entities, 12,672 relationships.

## How it compares

| | Vector RAG | Microsoft GraphRAG | PageIndex | **NanoIndex** |
|---|---|---|---|---|
| **Reads structure** | No (chunks) | No (chunks) | Yes (tree) | Yes (tree + graph) |
| **Indexing cost** | Low | High (LLM/chunk) | High (LLM/page) | Low (1 API call) |
| **Entity graph** | None | LLM-only | None | Local NER (free) |
| **Agent retrieval** | No | No | No | Multi-round |
| **Vision** | No | No | No | Page images |
| **Citations** | Chunk-level | None | Page-level | Pixel-level |

## Development

```bash
git clone https://github.com/nanonets/nanoindex.git
cd nanoindex

# With uv
uv sync --extra dev
uv run pytest

# With pip
pip install -e ".[dev]"
pytest
```

For GLiNER entity extraction:

```bash
pip install nanoindex[gliner]       # GLiNER v1 (CPU, fast)
pip install nanoindex[gliner-gpu]   # GLiNER2 large (GPU, best quality)
```

Auto-selects the right model for your hardware. GPU gets GLiNER2 large. CPU gets GLiNER v1 medium.

---

Built on [Nanonets OCR-3](https://nanonets.com/research/nanonets-ocr-3). Apache 2.0.
