"""Command line entry point. Each command is owned by the workstream noted below."""

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
def data_scrape(out: Path = Path("data/raw"), limit: int = 1000) -> None:
    """Download curated maps from BeatSaver (WS3)."""
    raise NotImplementedError


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
