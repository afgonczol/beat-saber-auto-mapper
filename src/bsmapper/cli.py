"""Command line entry point. Each command is owned by the workstream noted below."""

import sys
from pathlib import Path
from typing import Annotated

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


def _setup_logging(verbose: bool) -> None:
    import logging

    from rich.logging import RichHandler

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


def _echo_ratings_coverage(out: Path) -> None:
    from bsmapper.data import ratings_coverage, read_manifest

    cov = ratings_coverage(read_manifest(out))
    typer.echo(
        f"BeatLeader ratings: {cov.maps_with_ratings}/{cov.maps} maps, "
        f"{cov.diffs_with_tech}/{cov.diffs} difficulties have a techRating"
        + (f", {cov.not_looked_up} maps not looked up yet." if cov.not_looked_up else ".")
    )


@data_app.command("scrape")
def data_scrape(
    out: Path = Path("data/raw"),
    limit: int = typer.Option(
        1000,
        help="Total maps wanted in the manifest (resumes count). With --characteristics, "
        "only maps having one of them count.",
    ),
    min_score: float = typer.Option(0.75, help="Minimum BeatSaver rating (0..1)."),
    min_votes: int = typer.Option(30, help="Minimum upvotes + downvotes."),
    since: str = typer.Option("2021-01-01", help="Only maps uploaded on or after this date."),
    characteristics: str | None = typer.Option(
        None,
        help="Only maps with any of these characteristics, comma-separated, e.g. "
        "OneSaber,360Degree,90Degree. Has its own resume cursor.",
    ),
    max_gb: float = typer.Option(50.0, help="Stop once downloaded zips reach this many GB."),
    min_free_gb: float = typer.Option(20.0, help="Stop if the disk has less free space."),
    gold_list: str | None = typer.Option(
        None, help="Text file of curated BeatSaver keys (one per line) to tag as gold."
    ),
    ratings: bool = typer.Option(True, help="Also fetch BeatLeader per-difficulty ratings."),
    rps: float = typer.Option(2.0, help="Maximum requests per second, per API."),
    max_pages: int | None = typer.Option(None, help="Stop after this many search pages."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Download filtered maps from BeatSaver into OUT, with a manifest. Safe to rerun (WS3)."""
    from contextlib import ExitStack
    from datetime import UTC, datetime

    from bsmapper.data import (
        BeatLeaderClient,
        BeatSaverClient,
        FilterConfig,
        ScrapeConfig,
        load_curated_keys,
    )
    from bsmapper.data import scrape as run_scrape

    _setup_logging(verbose)
    since_dt = datetime.fromisoformat(since)
    if since_dt.tzinfo is None:
        since_dt = since_dt.replace(tzinfo=UTC)
    chars = tuple(c.strip() for c in (characteristics or "").split(",") if c.strip())
    try:
        filters = FilterConfig(
            min_score=min_score,
            min_votes=min_votes,
            uploaded_after=since_dt,
            characteristics=chars,
        )
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="--characteristics") from e
    cfg = ScrapeConfig(
        out_dir=out,
        limit=limit,
        max_bytes=int(max_gb * 1024**3),
        min_free_bytes=int(min_free_gb * 1024**3),
        max_pages=max_pages,
        filters=filters,
        curated_keys=load_curated_keys(Path(gold_list)) if gold_list else frozenset(),
    )
    with ExitStack() as stack:
        client = stack.enter_context(
            BeatSaverClient(cache_dir=out / "cache", requests_per_second=rps)
        )
        bl = None
        if ratings:
            bl = stack.enter_context(
                BeatLeaderClient(cache_dir=out / "cache", requests_per_second=rps)
            )
        result = run_scrape(client, cfg, ratings=bl)
    if chars:
        typer.echo(f"{result.matching} maps in the manifest have {', '.join(chars)}.")
    typer.echo(
        f"Stopped ({result.stop_reason}): {result.downloaded} new, {result.total} total, "
        f"{result.gold} gold, {result.total_bytes / 1024**3:.2f} GB, "
        f"{len(result.failed)} failed this run."
    )
    if result.rejected:
        typer.echo("Rejected: " + ", ".join(f"{k}={v}" for k, v in result.rejected.most_common()))
    if ratings:
        _echo_ratings_coverage(out)


@data_app.command("ratings")
def data_ratings(
    out: Path = Path("data/raw"),
    refresh: bool = typer.Option(False, help="Re-fetch every map, ignoring the cache."),
    rps: float = typer.Option(2.0, help="Maximum requests per second to BeatLeader."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Add BeatLeader ratings to maps already in OUT's manifest (WS3)."""
    from bsmapper.data import BeatLeaderClient, backfill_ratings

    _setup_logging(verbose)
    ttl = 0.0 if refresh else None
    with BeatLeaderClient(cache_dir=out / "cache", cache_ttl=ttl, requests_per_second=rps) as bl:
        n = backfill_ratings(bl, out, refresh=refresh)
    typer.echo(f"Updated {n} maps.")
    _echo_ratings_coverage(out)


@data_app.command("build")
def data_build(raw: Path = Path("data/raw"), out: Path = Path("data/processed")) -> None:
    """Parse, analyze and tokenize downloaded maps into training shards (WS3)."""
    raise NotImplementedError


@app.command()
def train(
    config: Path,
    override: Annotated[
        list[str] | None,
        typer.Option(
            "--override",
            "-o",
            help="section.key=value, e.g. -o optim.max_steps=500 (repeatable).",
        ),
    ] = None,
    resume: bool = typer.Option(True, help="Continue from run.out_dir/last.pt if it exists."),
    max_steps: int | None = typer.Option(
        None, help="Stop at this step (default: optim.max_steps)."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Train the note/lighting model from a YAML config, e.g. configs/note_v0.yaml (WS6)."""
    try:
        from bsmapper.training import TrainConfig
        from bsmapper.training.trainer import Trainer
    except ImportError as e:
        raise typer.Exit(f"Training needs the ml extra: pip install -e .[ml] ({e})") from e

    _setup_logging(verbose)
    try:
        cfg = TrainConfig.load(config, override)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    trainer = Trainer(cfg, resume=resume)
    typer.echo(
        f"{trainer.model.num_parameters() / 1e6:.1f}M parameters on {trainer.device}, "
        f"{len(trainer.notes)} note windows"
        + (f", {len(trainer.lights)} light windows" if trainer.lights is not None else "")
        + f"; step {trainer.step} -> {max_steps or cfg.optim.max_steps}, out: {cfg.run.out_dir}"
    )
    result = trainer.train(max_steps)
    typer.echo(" ".join(f"{k}={v:.4g}" for k, v in result.items()))


@app.command()
def evaluate(checkpoint: Path, split: str = "test") -> None:
    """Score a checkpoint against held-out human maps (WS10)."""
    raise NotImplementedError


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    work_dir: Path = Path("out/server"),
    web_dist: str | None = typer.Option(None, help="Built web UI folder; default: web/dist."),
    mock_delay: float = typer.Option(0.4, help="Seconds per mock pipeline step."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the local web UI and jobs API (WS9). Uses the mock pipeline for now."""
    try:
        import uvicorn

        from bsmapper.server.app import create_app
        from bsmapper.server.mock import MockJobRunner
    except ImportError as e:
        raise typer.Exit(
            f"The server needs the server extra: pip install -e .[server] ({e})"
        ) from e

    _setup_logging(verbose)
    application = create_app(
        MockJobRunner(step_delay=mock_delay), work_dir, Path(web_dist) if web_dist else None
    )
    typer.echo(f"bsmapper UI on http://{host}:{port}/  (API docs at /docs)")
    uvicorn.run(application, host=host, port=port, log_level="debug" if verbose else "info")
