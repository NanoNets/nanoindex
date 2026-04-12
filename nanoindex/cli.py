"""Click-based CLI for NanoIndex.

Registered as the ``nanoindex`` entry point in ``pyproject.toml``.
Usage:
    nanoindex index report.pdf -o tree.json
    nanoindex search tree.json "What was the revenue?"
    nanoindex ask report.pdf "What was the revenue?"
"""

from __future__ import annotations


import click
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree as RichTree

console = Console()


def _lazy_nanoindex(**kwargs):
    from nanoindex import NanoIndex

    return NanoIndex(**kwargs)


def _common_llm_options(fn):
    """Shared CLI options for LLM configuration."""
    fn = click.option("--llm-base-url", default=None, help="LLM API base URL")(fn)
    fn = click.option("--llm-api-key", default=None, help="LLM API key")(fn)
    fn = click.option("--llm-model", default=None, help="LLM model name")(fn)
    fn = click.option("--nanonets-api-key", default=None, help="Nanonets API key")(fn)
    return fn


def _build_kwargs(
    nanonets_api_key: str | None,
    llm_base_url: str | None,
    llm_api_key: str | None,
    llm_model: str | None,
) -> dict:
    kw = {}
    if nanonets_api_key:
        kw["nanonets_api_key"] = nanonets_api_key
    if llm_base_url:
        kw["llm_base_url"] = llm_base_url
    if llm_api_key:
        kw["llm_api_key"] = llm_api_key
    if llm_model:
        kw["llm_model"] = llm_model
    return kw


# ------------------------------------------------------------------
# CLI group
# ------------------------------------------------------------------


@click.group()
@click.version_option(package_name="nanoindex")
def main():
    """NanoIndex — Nanonets-powered document intelligence."""


# ------------------------------------------------------------------
# index
# ------------------------------------------------------------------


@main.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), default=None, help="Output JSON path")
@click.option("--add-summaries/--no-summaries", default=True, help="Generate LLM summaries")
@click.option("--add-description/--no-description", default=False, help="Generate doc description")
@_common_llm_options
def index(
    file_path: str,
    output: str | None,
    add_summaries: bool,
    add_description: bool,
    nanonets_api_key: str | None,
    llm_base_url: str | None,
    llm_api_key: str | None,
    llm_model: str | None,
):
    """Index a document and build a tree structure."""
    kwargs = _build_kwargs(nanonets_api_key, llm_base_url, llm_api_key, llm_model)
    ni = _lazy_nanoindex(**kwargs)

    with console.status("[bold green]Indexing document…"):
        tree = ni.index(
            file_path,
            add_summaries=add_summaries,
            add_doc_description=add_description,
        )

    if output:
        from nanoindex.utils.tree_ops import save_tree

        save_tree(tree, output)
        console.print(f"[green]✓[/] Tree saved to {output}")
    else:
        _print_tree(tree)

    console.print(
        f"\n[dim]Nodes: {_count_nodes(tree.structure)} | "
        f"Pages: {tree.extraction_metadata.get('pages_processed', '?')}[/]"
    )


# ------------------------------------------------------------------
# search
# ------------------------------------------------------------------


@main.command()
@click.argument("tree_path", type=click.Path(exists=True))
@click.argument("query")
@_common_llm_options
def search(
    tree_path: str,
    query: str,
    nanonets_api_key: str | None,
    llm_base_url: str | None,
    llm_api_key: str | None,
    llm_model: str | None,
):
    """Search an indexed tree for relevant sections."""
    from nanoindex.utils.tree_ops import load_tree

    kwargs = _build_kwargs(nanonets_api_key, llm_base_url, llm_api_key, llm_model)
    ni = _lazy_nanoindex(**kwargs)
    tree = load_tree(tree_path)

    with console.status("[bold green]Searching…"):
        results = ni.search(query, tree)

    if not results:
        console.print("[yellow]No relevant sections found.[/]")
        return

    for rn in results:
        console.print(
            Panel(
                f"[bold]{rn.node.title}[/] [{rn.node.node_id}]\n"
                f"Pages {rn.node.start_index}-{rn.node.end_index}\n\n"
                f"{rn.text[:500]}{'…' if len(rn.text) > 500 else ''}",
                border_style="cyan",
            )
        )


# ------------------------------------------------------------------
# ask
# ------------------------------------------------------------------


@main.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.argument("query")
@click.option("--mode", type=click.Choice(["text", "vision"]), default="text")
@click.option("--tree-path", type=click.Path(), default=None, help="Pre-built tree JSON")
@click.option(
    "--metadata", is_flag=True, default=False, help="Include bounding box metadata in citations"
)
@_common_llm_options
def ask(
    file_path: str,
    query: str,
    mode: str,
    tree_path: str | None,
    metadata: bool,
    nanonets_api_key: str | None,
    llm_base_url: str | None,
    llm_api_key: str | None,
    llm_model: str | None,
):
    """Index (if needed), search, and answer a question."""
    kwargs = _build_kwargs(nanonets_api_key, llm_base_url, llm_api_key, llm_model)
    ni = _lazy_nanoindex(**kwargs)

    if tree_path:
        from nanoindex.utils.tree_ops import load_tree

        tree = load_tree(tree_path)
    else:
        with console.status("[bold green]Indexing document…"):
            tree = ni.index(file_path)

    with console.status("[bold green]Generating answer…"):
        answer = ni.ask(query, tree, mode=mode, pdf_path=file_path, include_metadata=metadata)

    console.print(Panel(answer.content, title="Answer", border_style="green"))

    if answer.citations:
        console.print("\n[bold]Citations:[/]")
        for c in answer.citations:
            pages = f"pp. {', '.join(str(p) for p in c.pages)}" if c.pages else ""
            console.print(f"  • {c.title} [{c.node_id}] {pages}")
            if c.bounding_boxes:
                console.print(f"    [dim]{len(c.bounding_boxes)} bounding boxes[/]")
                for bb in c.bounding_boxes[:5]:
                    console.print(
                        f"    [dim]  page {bb.page}: "
                        f"({bb.x:.3f}, {bb.y:.3f}) {bb.width:.3f}×{bb.height:.3f} "
                        f"[{bb.region_type}] conf={bb.confidence:.2f}[/]"
                    )
                if len(c.bounding_boxes) > 5:
                    console.print(f"    [dim]  … and {len(c.bounding_boxes) - 5} more[/]")
            if c.page_dimensions:
                for pd in c.page_dimensions:
                    console.print(f"    [dim]  page {pd.page}: {pd.width}×{pd.height}px[/]")


# ------------------------------------------------------------------
# viz
# ------------------------------------------------------------------


@main.command()
@click.argument("tree_path", type=click.Path(exists=True), required=False, default=None)
@click.option("--port", type=int, default=3000, help="Port for the dashboard")
@click.option("--no-open", is_flag=True, default=False, help="Don't auto-open browser")
def viz(tree_path: str | None, port: int, no_open: bool):
    """Launch the interactive visualization dashboard.

    \b
    Usage:
        nanoindex viz                  # Start dashboard with all cached trees
        nanoindex viz tree.json        # Open a specific tree
        nanoindex viz --port 8080      # Use a custom port
    """
    import os
    import subprocess
    import shutil
    from pathlib import Path

    viz_dir = Path(__file__).resolve().parent.parent / "viz"

    if not viz_dir.exists():
        console.print("[red]Viz directory not found.[/] Make sure you installed from source.")
        raise SystemExit(1)

    # Check Node.js is available
    if not shutil.which("node"):
        console.print("[red]Node.js not found.[/] Install it from https://nodejs.org")
        raise SystemExit(1)

    # Install deps if needed
    node_modules = viz_dir / "node_modules"
    if not node_modules.exists():
        console.print("[dim]Installing frontend dependencies…[/]")
        subprocess.run(["npm", "install"], cwd=viz_dir, check=True, capture_output=True)

    # If a specific tree was given, copy it to the cache dir for visibility
    if tree_path:
        from nanoindex.utils.tree_ops import load_tree

        tree = load_tree(tree_path)
        doc_name = tree.doc_name or Path(tree_path).stem

        # Ensure the tree is in the cache dir the viz reads from
        cache_dir = viz_dir.parent / "benchmarks" / "cache_v3"
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / f"{doc_name}.json"
        if not target.exists():
            import shutil as _shutil

            _shutil.copy2(tree_path, target)
            console.print(f"[dim]Copied tree to {target}[/]")

    url = f"http://localhost:{port}"
    if tree_path:
        from nanoindex.utils.tree_ops import load_tree

        tree = load_tree(tree_path)
        doc_name = tree.doc_name or Path(tree_path).stem
        url += f"/tree?name={doc_name}"

    console.print(f"[bold green]Starting NanoIndex dashboard[/] at [underline]{url}[/]")

    # Open browser
    if not no_open:
        import webbrowser
        import threading

        threading.Timer(2.0, lambda: webbrowser.open(url)).start()

    # Start Next.js dev server
    env = {**os.environ, "PORT": str(port)}
    try:
        subprocess.run(
            ["npx", "next", "dev", "--port", str(port)],
            cwd=viz_dir,
            env=env,
        )
    except KeyboardInterrupt:
        console.print("\n[dim]Dashboard stopped.[/]")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _print_tree(doc_tree):
    """Render a DocumentTree as a rich Tree widget."""

    rich_tree = RichTree(f"[bold]{doc_tree.doc_name}[/]")

    def _add_nodes(parent, nodes):
        for node in nodes:
            label = f"[cyan]{node.node_id}[/] {node.title}"
            if node.start_index:
                label += f" [dim](pp. {node.start_index}-{node.end_index})[/]"
            branch = parent.add(label)
            _add_nodes(branch, node.nodes)

    _add_nodes(rich_tree, doc_tree.structure)
    console.print(rich_tree)


def _count_nodes(nodes) -> int:
    count = len(nodes)
    for n in nodes:
        count += _count_nodes(n.nodes)
    return count


# ------------------------------------------------------------------
# kb (Knowledge Base commands)
# ------------------------------------------------------------------


@main.group()
def kb():
    """Knowledge Base commands."""
    pass


@kb.command("create")
@click.argument("path", type=click.Path())
@_common_llm_options
def kb_create(path, nanonets_api_key, llm_base_url, llm_api_key, llm_model):
    """Create a new knowledge base."""
    from nanoindex.kb import KnowledgeBase

    kwargs = _build_kwargs(nanonets_api_key, llm_base_url, llm_api_key, llm_model)
    kb_inst = KnowledgeBase(path, **kwargs)
    console.print(f"[green]Created knowledge base at {path}[/green]")
    stats = kb_inst.status()
    console.print(f"  Documents: {stats['documents']}  Concepts: {stats['concepts']}")


@kb.command("add")
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("--wiki", "-w", type=click.Path(), default=".", help="Wiki directory")
@_common_llm_options
def kb_add(pdf_path, wiki, nanonets_api_key, llm_base_url, llm_api_key, llm_model):
    """Add a document to the knowledge base."""
    from nanoindex.kb import KnowledgeBase

    kwargs = _build_kwargs(nanonets_api_key, llm_base_url, llm_api_key, llm_model)
    kb_inst = KnowledgeBase(wiki, **kwargs)
    with console.status("[bold green]Indexing and compiling..."):
        doc = kb_inst.add(pdf_path)
    console.print(f"[green]Added:[/green] {doc.doc_name}")
    stats = kb_inst.status()
    console.print(f"  Documents: {stats['documents']}  Concepts: {stats['concepts']}")


@kb.command("ask")
@click.argument("question")
@click.option("--wiki", "-w", type=click.Path(), default=".", help="Wiki directory")
@click.option("--mode", default="fast", help="Query mode: fast or agentic")
@_common_llm_options
def kb_ask(question, wiki, mode, nanonets_api_key, llm_base_url, llm_api_key, llm_model):
    """Ask a question across the knowledge base."""
    from nanoindex.kb import KnowledgeBase

    kwargs = _build_kwargs(nanonets_api_key, llm_base_url, llm_api_key, llm_model)
    kb_inst = KnowledgeBase(wiki, **kwargs)
    with console.status("[bold green]Searching..."):
        answer = kb_inst.ask(question, mode=mode)
    console.print(Panel(answer.content, title="Answer", border_style="green"))
    if answer.citations:
        for c in answer.citations:
            console.print(f"  [dim]{c.title} (pp. {c.pages})[/dim]")


@kb.command("status")
@click.option("--wiki", "-w", type=click.Path(), default=".", help="Wiki directory")
def kb_status(wiki):
    """Show knowledge base statistics."""
    from nanoindex.kb import KnowledgeBase

    kb_inst = KnowledgeBase(wiki)
    stats = kb_inst.status()
    console.print(f"[bold]Knowledge Base:[/bold] {wiki}")
    console.print(f"  Documents:     {stats['documents']}")
    console.print(f"  Concepts:      {stats['concepts']}")
    console.print(f"  Queries:       {stats['queries']}")
    console.print(f"  Entities:      {stats['entities']}")
    console.print(f"  Relationships: {stats['relationships']}")


@kb.command("lint")
@click.option("--wiki", "-w", type=click.Path(), default=".", help="Wiki directory")
def kb_lint(wiki):
    """Find inconsistencies in the knowledge base."""
    from nanoindex.kb import KnowledgeBase

    kb_inst = KnowledgeBase(wiki)
    warnings = kb_inst.lint()
    if warnings:
        for w in warnings:
            console.print(f"  [yellow]⚠[/yellow] {w}")
        console.print(f"\n[yellow]{len(warnings)} warnings[/yellow]")
    else:
        console.print("[green]No issues found[/green]")


if __name__ == "__main__":
    main()
