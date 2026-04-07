# Introducing NanoIndex: the agent harness that teaches AI to actually read documents

**Most RAG systems search documents. NanoIndex reads them.**

Today we're open-sourcing NanoIndex — a new approach to document understanding that gives AI agents the ability to structurally navigate, reason over, and cite from documents of any length. Instead of chopping PDFs into chunks and hoping the right piece floats to the top, NanoIndex builds a hierarchical tree of the document's structure, extracts a knowledge graph of entities and relationships, then lets agents explore the document in multiple rounds — the same way a human analyst would.

On FinanceBench (150 questions across 84 SEC filings, avg 143 pages each), NanoIndex achieves **85% accuracy with 90% evidence page hit rate** — citing the exact page and pixel coordinates where each answer lives.

For the documents that run your business — financial filings, legal contracts, medical records, insurance claims — NanoIndex doesn't just find text. It understands structure.

---

## The problem with every RAG system today

Over the past two years, we kept seeing the same failure pattern. Teams would build a RAG pipeline — chunk their PDFs, embed the chunks, wire up a vector database — and it would work great on demos. Simple questions, short documents, clean formatting.

Then they'd try it on a real 200-page 10-K filing. Or a 50-page legal contract. Or a medical record with dense tables across 30 pages.

The accuracy would collapse.

Not because the LLM was bad. Because the retrieval was fundamentally broken.

**The issue isn't that models can't read documents. It's that chunking destroys the thing that makes documents useful: their structure.**

A revenue figure in a 10-K isn't just a number floating in space. It lives in the Consolidated Statement of Income, under a specific fiscal year column, in a section that references Notes to Financial Statements three pages later. Chunk that document into 512-token pieces and all of that context evaporates.

We talked to teams processing financial documents, legal filings, insurance claims, and medical records. The pattern was always the same:

- **Chunking split tables mid-row.** A balance sheet becomes meaningless fragments.
- **Embeddings couldn't distinguish context.** "Revenue of $32B" in the MD&A discussion vs. the actual income statement — same embedding, completely different authority.
- **Single-pass retrieval missed multi-part answers.** Computing a ratio requires data from the income statement AND the balance sheet. The retriever found one, not both.
- **Citations were fake.** The system said "page 47" but couldn't point to the exact cell in the exact table. Compliance teams couldn't verify anything.

The workaround was always the same: hire people to check the output. Human-in-the-loop on every answer. At that point, what's the AI even doing?

---

## NanoIndex's approach: give agents a structural map, not a bag of chunks

The rise of agentic AI pointed to a better way. If agents could navigate codebases, browse the web, and execute multi-step tasks — why were we still treating document understanding as a one-shot similarity search?

NanoIndex takes a fundamentally different approach. Instead of chunking a document, it builds two things:

**1. A self-validating document tree.**
The document's own hierarchy — sections, subsections, tables, exhibits — preserved exactly as the author structured them. The tree validates itself: checks page coverage, node depth, empty sections, duplicate IDs. If a 160-page filing has gaps, you know before you query.

**2. A domain-adaptive knowledge graph.**
Entities (companies, people, dollar amounts, dates, business segments) extracted using zero-shot NER — no training, no expensive LLM calls. The system auto-detects the document type (SEC 10-K? Legal contract? Medical record?) and loads domain-specific entity labels. Relationships are extracted via dependency parsing. Communities are detected via Louvain algorithm.

When you ask a question, an LLM agent navigates this structure in multiple rounds:

- **Round 1:** Decompose the question. "I need net income and total assets to compute ROA."
- **Round 2:** Navigate the tree. "Net income is in the income statement — node 24.20. Total assets are in the balance sheet — node 24.22."
- **Round 3:** Read the content. Review the actual text and page images. Request more sections if needed.
- **Round 4:** Verify. Cross-check calculations. Run sufficiency checks against its own plan.

The agent doesn't search. It reasons.

---

## What this looks like in practice

```python
from nanoindex import NanoIndex

ni = NanoIndex(llm="anthropic:claude-sonnet-4-6")

# Index a 200-page SEC filing -> structured tree + knowledge graph
tree = ni.index("3M_2018_10K.pdf")
# tree.domain = "sec_10k"
# 153 nodes, 921 entities, 103 relationships
# Self-validation: 100% page coverage, depth 4

# Ask a question -- agent reasons in multiple rounds
answer = ni.ask("What was 3M's free cash flow conversion ratio in FY2018?", tree)
# Agent: decomposes -> finds operating cash flow in cash flow statement
#         -> finds CapEx -> finds net income -> computes FCF -> computes ratio
# Answer: "FCF conversion = 154.7% ($4,862M FCF / $3,143M adjusted net income)"
# Citations: bounding boxes at (x=234, y=567, w=89, h=14) on page 52
```

Five lines of code. No chunking strategy to tune. No embedding model to pick. No vector database to manage.

---

## The modes: from fast lookup to deep reasoning

Not every question needs an agent loop. NanoIndex offers a spectrum:

| Mode | How it works | Cost | Best for |
|---|---|---|---|
| **fast_vision** | Entity graph lookup, LLM picks from candidates | 2 LLM calls, ~17s | Simple lookups: "What was revenue?" |
| **agentic_graph_vision** | Graph seeds initial nodes, agent reasons + expands | 4-6 LLM calls, ~67s | Complex reasoning: "Is FCF conversion improving?" |
| **global** | Map-reduce across community summaries | N+1 calls | Broad questions: "What are the key risk factors?" |

The graph-seeded agentic mode is the key innovation. The knowledge graph narrows the search space (from 153 nodes to ~20 relevant ones), then the agent reasons from there. Best of both worlds: graph precision + agentic depth.

---

## Built for the documents that matter most

NanoIndex auto-detects document domains and loads specialized knowledge:

**Financial (SEC 10-K, 10-Q, 8-K, earnings):** 31 canonical formulas — ROA, ROE, FCF conversion, EBITDA, working capital, effective tax rate. Each with conventions like "tax rates can be negative — preserve the sign" and "FCF = OCF minus CapEx, not OCF alone." The system doesn't just know formulas. It enforces them.

**Legal:** Party, court, case number, statute, jurisdiction, damages. Entity labels tuned for contracts and litigation.

**Medical:** Patient, diagnosis, drug, procedure, dosage. Structured for clinical documents.

**Insurance:** Policy, claim, coverage type, premium, loss amount. Built for loss runs and claims processing.

Or bring your own domain labels. GLiNER's zero-shot NER means no training data required — just a list of entity types.

---

## The numbers

| Benchmark | Documents | Avg Pages | Accuracy | Evidence Page Hit |
|---|---|---|---|---|
| FinanceBench (SEC filings) | 84 | 143 | **85.3%** | 90.0% |
| DocBench Legal | 51 | 54 | **96.0%** | -- |

On FinanceBench, broken down by filing type:

| Filing Type | Accuracy | Count |
|---|---|---|
| 10-K (annual reports) | **87%** | 112 |
| 8-K (current reports) | **100%** | 9 |
| Earnings releases | **93%** | 14 |

96 entity graphs built across the benchmark — 68,045 entities and 12,672 relationships — using GLiNER2 on GPU in 15 minutes.

---

## Open source. pip install. Five lines of code.

```bash
pip install nanoindex
```

```bash
export NANONETS_API_KEY=your_key    # Document parsing (free 10K pages)
export ANTHROPIC_API_KEY=your_key   # Or OPENAI_API_KEY, GOOGLE_API_KEY
```

```python
from nanoindex import NanoIndex
ni = NanoIndex()
tree = ni.index("document.pdf")
answer = ni.ask("Your question", tree)
print(answer.content, answer.citations)
```

NanoIndex is not a framework. It's a library. No dependency trees, no abstractions, no lock-in. Built on [Nanonets OCR-3](https://nanonets.com/research/nanonets-ocr-3). Apache 2.0.

GitHub: [github.com/nanonets/nanoindex](https://github.com/nanonets/nanoindex)
PyPI: [pypi.org/project/nanoindex](https://pypi.org/project/nanoindex)

---

## What's next

NanoIndex is the first step toward a world where AI agents can actually read documents — not search them, not summarize them, but structurally understand them. We're working on:

- **Multi-document reasoning** — agents that cross-reference across a portfolio of filings
- **Streaming extraction** — real-time tree building as pages are parsed
- **Custom knowledge bases** — Obsidian-compatible wikis that grow smarter with every query

The documents that run your business deserve better than ctrl+F with extra steps.
