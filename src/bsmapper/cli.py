"""Command line entry point. Each command is owned by the workstream noted below."""

import sys
from pathlib import Path

import typer

app = typer.Typer(help="Automatic Beat Saber map generator.", no_args_is_help=True)
data_app = typer.Typer(help="Dataset scraping and building (WS3).")
app.add_typer(data_app, name="data")


@app.command()
def generate(
    audio: Path,
    out: Path = Path("out"),
    difficulties: str = "Easy,Normal,Hard,Expert,ExpertPlus",
    modes: str = "Standard",
    lighting: str = "v3",
    zip_output: bool = typer.Option(True, "--zip/--no-zip"),
    style: float | None = typer.Option(None, min=0.0, max=1.0, help="0 = flowy, 1 = technical."),
    install: bool = typer.Option(False, help="Copy the result into Beat Saber's CustomLevels."),
    seed: int = 0,
) -> None:
    """Generate a full map folder from an audio file (WS7)."""
    raise NotImplementedError


@data_app.command("scrape")
def data_scrape(
    out: Path = Path("data/raw"),
    limit: int = typer.Option(1000, help="Total maps wanted in the manifest (resumes count)."),
    min_score: float = typer.Option(0.75, help="Minimum BeatSaver rating (0..1)."),
    min_votes: int = typer.Option(30, help="Minimum upvotes + downvotes."),
    since: str = typer.Option("2021-01-01", help="Only maps uploaded on or after this date."),
    max_gb: float = typer.Option(50.0, help="Stop once downloaded zips reach this many GB."),
    min_free_gb: float = typer.Option(20.0, help="Stop if the disk has less free space."),
    gold_list: str | None = typer.Option(
        None, help="Text file of curated BeatSaver keys (one per line) to tag as gold."
    ),
    rps: float = typer.Option(2.0, help="Maximum requests per second to BeatSaver."),
    max_pages: int | None = typer.Option(None, help="Stop after this many search pages."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Download filtered maps from BeatSaver into OUT, with a manifest. Safe to rerun (WS3)."""
    import logging
    from datetime import UTC, datetime

    from rich.logging import RichHandler

    from bsmapper.data import BeatSaverClient, FilterConfig, ScrapeConfig, load_curated_keys
    from bsmapper.data import scrape as run_scrape

    # Song titles are often non-Latin; don't let a cp1252 console or log file choke on them.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(show_path=False)],
    )
    logging.getLogger("httpx").setLevel(logging.INFO if verbose else logging.WARNING)
    since_dt = datetime.fromisoformat(since)
    if since_dt.tzinfo is None:
        since_dt = since_dt.replace(tzinfo=UTC)
    cfg = ScrapeConfig(
        out_dir=out,
        limit=limit,
        max_bytes=int(max_gb * 1024**3),
        min_free_bytes=int(min_free_gb * 1024**3),
        max_pages=max_pages,
        filters=FilterConfig(min_score=min_score, min_votes=min_votes, uploaded_after=since_dt),
        curated_keys=load_curated_keys(Path(gold_list)) if gold_list else frozenset(),
    )
    with BeatSaverClient(cache_dir=out / "cache", requests_per_second=rps) as client:
        result = run_scrape(client, cfg)
    typer.echo(
        f"Stopped ({result.stop_reason}): {result.downloaded} new, {result.total} total, "
        f"{result.gold} gold, {result.total_bytes / 1024**3:.2f} GB, "
        f"{len(result.failed)} failed this run."
    )
    if result.rejected:
        typer.echo("Rejected: " + ", ".join(f"{k}={v}" for k, v in result.rejected.most_common()))


@data_app.command("build")
def data_build(raw: Path = Path("data/raw"), out: Path = Path("data/processed")) -> None:
    """Parse, analyze and tokenize downloaded maps into training shards (WS3)."""
    raise NotImplementedError


@app.command()
def train(config: Path) -> None:
    """Train a model from a YAML config (WS6)."""
    raise NotImplementedError


@app.command()
def evaluate(checkpoint: Path, split: str = "test") -> None:
    """Score a checkpoint against held-out human maps (WS10)."""
    raise NotImplementedError


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Run the local web UI (WS9)."""
    raise NotImplementedError
