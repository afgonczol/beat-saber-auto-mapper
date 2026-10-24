"""Command line entry point. Each command is owned by the workstream noted below."""

import sys
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="Automatic Beat Saber map generator.", no_args_is_help=True)
data_app = typer.Typer(help="Dataset scraping and building (WS3).")
app.add_typer(data_app, name="data")
lights_app = typer.Typer(help="Lighting style labels and scene tables (WS8b).")
app.add_typer(lights_app, name="lights")


_MODE_ALIASES = {"90": "90Degree", "360": "360Degree", "onehand": "OneSaber"}
_DIFF_ALIASES = {"expert+": "ExpertPlus", "e+": "ExpertPlus", "ex+": "ExpertPlus"}


def _parse_jump_option(text: str | None, param: str, low: float, high: float):
    """``"0.8"`` -> (0.8, {}); ``"Expert=0.8,Expert+=0.7"`` -> (None, {Difficulty: 0.8, ...})."""
    from bsmapper.schema import Difficulty

    if not text:
        return None, {}

    def number(raw: str) -> float:
        try:
            value = float(raw)
        except ValueError:
            raise typer.BadParameter(f"{raw!r} is not a number", param_hint=param) from None
        if not low <= value <= high:
            raise typer.BadParameter(f"{value:g} is outside {low:g}..{high:g}", param_hint=param)
        return value

    if "=" not in text:
        return number(text.strip()), {}
    per = {}
    for part in (p.strip() for p in text.split(",") if p.strip()):
        name, _, value = part.partition("=")
        (diff,) = _parse_enum_list(name, Difficulty, _DIFF_ALIASES, param)
        per[diff] = number(value.strip())
    return None, per


def _parse_enum_list(text: str, enum, aliases: dict[str, str], param: str) -> list:
    by_name = {m.value.lower(): m for m in enum}
    out = []
    for raw in (t.strip() for t in text.split(",")):
        if not raw:
            continue
        key = aliases.get(raw.lower(), raw).lower()
        if key not in by_name:
            choices = ", ".join(m.value for m in enum)
            raise typer.BadParameter(
                f"unknown value {raw!r}; choose from {choices}", param_hint=param
            )
        if by_name[key] not in out:
            out.append(by_name[key])
    if not out:
        raise typer.BadParameter("give at least one value", param_hint=param)
    return out


@app.command()
def generate(
    audio: Path,
    out: Path = Path("out"),
    difficulties: str = typer.Option(
        "Easy,Normal,Hard,Expert,ExpertPlus", help="Comma-separated; Expert+ works too."
    ),
    modes: str = typer.Option(
        "Standard", help="Comma-separated: Standard, OneSaber, NoArrows, 90Degree, 360Degree."
    ),
    lighting: str = typer.Option("v3", help="Lightshow style: v3, vanilla or chroma."),
    zip_output: bool = typer.Option(True, "--zip/--no-zip", help="Also write a zip."),
    style: float | None = typer.Option(None, min=0.0, max=1.0, help="0 = flowy, 1 = technical."),
    intensity: float | None = typer.Option(
        None,
        min=0.0,
        max=1.0,
        help="Challenge within each difficulty: 0.5 = typical human map, 1 = about the hardest "
        "10%. Default: [generate] intensity in ~/.bsmapper/config.toml, else 0.5.",
    ),
    repetition: float | None = typer.Option(
        None,
        min=0.0,
        max=1.0,
        help="How much repeated music reuses earlier patterns (note_v1): 0 = never, 0.5 = "
        "the learned amount, 1 = twice as strongly. Default: [generate] repetition, else learned.",
    ),
    install: bool = typer.Option(False, help="Copy the result into Beat Saber's CustomLevels."),
    seed: int = 0,
    temperature: float = typer.Option(1.0, min=0.0, max=2.0, help="Sampling randomness."),
    njs: str | None = typer.Option(
        None, help="NJS: one value for every difficulty, or per difficulty (Expert=16,Expert+=17)."
    ),
    reaction_time: str | None = typer.Option(
        None,
        help="Seconds from spawn to strike: one value for Expert and Expert+, or per "
        "difficulty (Hard=0.9,Expert+=0.75). Defaults: 1.2/1.0/0.88/0.8/0.7.",
    ),
    environment: str | None = typer.Option(None, help="Info.dat environment, e.g. EDMEnvironment."),
    generator: str = typer.Option(
        "auto", help="Notes from: model, stub, or auto (the model when a checkpoint is found)."
    ),
    model: str | None = typer.Option(
        None,
        help="Model checkpoint; default [model] checkpoint in ~/.bsmapper/config.toml, "
        "then checkpoints/note_v0/best.pt.",
    ),
    lighting_source: str = typer.Option(
        "auto", help="Lights from: model, rules, or auto (the model when notes are)."
    ),
    light_style: str | None = typer.Option(
        None,
        help="Look of model lights (light_v2 checkpoints): auto (picked for the song), none, "
        "pulse, calm, scenes, flash or accent. Default: [generate] light_style, else auto.",
    ),
    device: str | None = typer.Option(None, help="cuda or cpu; default cuda when available."),
    bombs: bool | None = typer.Option(None, "--bombs/--no-bombs", help="Default: per difficulty."),
    walls: bool | None = typer.Option(None, "--walls/--no-walls", help="Default: on."),
    arcs: bool | None = typer.Option(
        None, "--arcs/--no-arcs", help="Default: [generate] arcs in the config, else Hard and up."
    ),
    chains: bool | None = typer.Option(
        None,
        "--chains/--no-chains",
        help="Default: [generate] chains in the config, else Expert and up.",
    ),
    tracker: str = typer.Option("auto", help="Beat tracker: auto, beat_this or librosa."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Generate a full map folder (and zip) from an audio file (WS7).

    Notes and lights come from the trained WS6 model when a checkpoint is found, else from
    the placeholder stub generator and the rule lightshow.
    """
    from bsmapper.generation import (
        GenerateOptions,
        JumpOverride,
        configured_jumps,
        install_map,
        load_config,
        make_pipeline,
    )
    from bsmapper.generation.density import configured_objects
    from bsmapper.generation.install import config_path
    from bsmapper.generation.intensity import (
        clamp_intensity,
        configured_intensity,
        configured_repetition,
    )
    from bsmapper.generation.lighting import LIGHT_STYLE_CHOICES, configured_light_style
    from bsmapper.lighting import STYLES
    from bsmapper.schema import Characteristic, Difficulty

    _setup_logging(verbose)
    if not audio.is_file():
        raise typer.BadParameter(f"{audio} is not a file", param_hint="AUDIO")
    if lighting not in STYLES:
        raise typer.BadParameter(f"choose from {', '.join(STYLES)}", param_hint="--lighting")
    for value, choices, hint in (
        (generator, ("auto", "model", "stub"), "--generator"),
        (lighting_source, ("auto", "model", "rules"), "--lighting-source"),
        (light_style or "auto", LIGHT_STYLE_CHOICES, "--light-style"),
    ):
        if value not in choices:
            raise typer.BadParameter(f"choose from {', '.join(choices)}", param_hint=hint)
    njs_all, njs_per = _parse_jump_option(njs, "--njs", 1.0, 40.0)
    rt_all, rt_per = _parse_jump_option(reaction_time, "--reaction-time", 0.3, 2.0)
    if rt_all is not None:
        rt_per = {Difficulty.EXPERT: rt_all, Difficulty.EXPERT_PLUS: rt_all} | rt_per
    jump = {d: JumpOverride(njs_per.get(d), rt_per.get(d)) for d in {*njs_per, *rt_per}}
    try:
        jump_defaults = configured_jumps()
        if intensity is None:
            intensity = configured_intensity()
        if repetition is None:
            repetition = configured_repetition()
        objects = configured_objects()
        if light_style is None:
            light_style = configured_light_style()
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    opts = GenerateOptions(
        modes=_parse_enum_list(modes, Characteristic, _MODE_ALIASES, "--modes"),
        difficulties=_parse_enum_list(difficulties, Difficulty, _DIFF_ALIASES, "--difficulties"),
        lighting=lighting,  # type: ignore[arg-type]
        style=style,
        intensity=intensity,
        repetition=repetition,
        seed=seed,
        temperature=temperature,
        njs=njs_all,
        jump=jump,
        jump_defaults=jump_defaults,
        environment=environment,
        lighting_source=lighting_source,
        light_style=light_style,
        bombs=objects.get("bombs") if bombs is None else bombs,
        walls=objects.get("walls") if walls is None else walls,
        arcs=objects.get("arcs") if arcs is None else arcs,
        chains=objects.get("chains") if chains is None else chains,
    )
    try:
        pipeline = make_pipeline(generator, lighting_source, model, device, tracker=tracker)
    except FileNotFoundError as e:
        raise typer.BadParameter(str(e), param_hint="--model") from e
    setup = pipeline.setup
    source = f"model {setup.checkpoint}" if setup.generator == "model" else "stub generator"
    typer.echo(f"Notes: {source}" + (f" ({setup.reason})" if setup.reason else ""))
    rep = "learned" if repetition is None else f"{repetition:.2f}"
    typer.echo(f"Intensity: {clamp_intensity(intensity):.2f}, repetition: {rep}")
    arcs_from, chains_from = "on from Hard", "on from Expert"
    onoff = {True: "on", False: "off"}
    typer.echo(
        f"Arcs: {onoff.get(opts.arcs, arcs_from)}, chains: {onoff.get(opts.chains, chains_from)}"
    )
    game = load_config()
    if lighting == "chroma" and not game.chroma_installed():
        typer.secho(
            f"Warning: Chroma is not installed ({game.chroma_dll} is missing). The map still "
            "loads, but its Chroma colours only show with the Chroma mod; use --lighting v3 "
            f"for unmodded Beat Saber. Set [game] beat_saber_dir in {config_path()} if the "
            "game is elsewhere.",
            fg=typer.colors.YELLOW,
            err=True,
        )

    def progress(stage: str, fraction: float, message: str) -> None:
        typer.echo(f"[{fraction:4.0%}] {stage:<8} {message}")

    result = pipeline.run(audio, out, opts, zip_output=zip_output, progress=progress)
    bm = result.beatmap
    typer.echo(
        f"\n{bm.song_name} - {bm.song_author or '?'}  ({result.song.bpm:g} BPM, "
        f"{bm.environment}, {lighting} lighting from {pipeline.last_lighting}"
        + (f" ({pipeline.last_light_style} style)" if pipeline.last_light_style else "")
        + f", {bm.mapper})"
    )
    for res in result.difficulties:
        d = res.dmap
        typer.echo(
            f"  {d.characteristic:<10} {d.difficulty:<10} {len(d.notes):5d} notes "
            f"{res.nps:5.2f} NPS  NJS {d.njs:g} offset {d.njs_offset:+g}  "
            f"errors/100 {res.per_100('error'):.2f} warnings/100 {res.per_100('warning'):.2f}  "
            f"{res.seconds:.1f}s  {res.report.summary()}"
        )
    for note in result.notes:
        typer.secho(f"Note: {note}", fg=typer.colors.YELLOW)
    typer.echo(f"Map folder: {result.map_dir}")
    if result.zip_path:
        typer.echo(f"Zip: {result.zip_path}")
    if install:
        try:
            dest = install_map(result.map_dir, game)
        except (OSError, FileNotFoundError) as e:
            typer.secho(f"Install failed: {e}", fg=typer.colors.RED, err=True)
            raise typer.Exit(1) from e
        typer.echo(f"Installed to {dest}")


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
def data_build(
    raw: Path = Path("data/raw"),
    out: Path = Path("data/processed"),
    workers: int | None = typer.Option(None, help="CPU worker processes (default: cores - 2)."),
    tracker: str = typer.Option(
        "beat_this",
        help="Beat tracker for the grid-drift stat: beat_this (GPU), librosa, auto or none.",
    ),
    limit: int | None = typer.Option(None, help="Only the first N maps of the manifest."),
    keys: str | None = typer.Option(None, help="Only these map keys, comma-separated."),
    retry_errors: bool = typer.Option(False, help="Process maps again that crashed before."),
    pack_only: bool = typer.Option(False, help="Skip processing; rewrite the shards only."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Parse, analyze and tokenize downloaded maps into training shards (WS3).

    Safe to rerun: maps already processed are skipped, then the shards, the
    manifest and the stats are rewritten from all processed maps.
    """
    from bsmapper.data.build import BuildConfig, run_build
    from bsmapper.data.pack import pack

    _setup_logging(verbose)
    if tracker not in ("beat_this", "librosa", "auto", "none"):
        raise typer.BadParameter(
            "expected beat_this, librosa, auto or none", param_hint="--tracker"
        )
    cfg = BuildConfig(
        raw_dir=raw,
        out_dir=out,
        tracker=tracker,
        limit=limit,
        keys=tuple(k.strip() for k in (keys or "").split(",") if k.strip()),
        retry_errors=retry_errors,
    )
    if workers:
        cfg.workers = workers
    if pack_only:
        from bsmapper.data.build import select_entries

        meta = pack(out, raw, keys=[k for k, _ in select_entries(cfg)])
    else:
        result = run_build(cfg)
        typer.echo(
            f"Processed {sum(result.processed.values())} maps "
            f"({result.skipped_done} were already done), beat-tracked {result.tracked}, "
            f"in {result.seconds / 60:.1f} min."
        )
        meta = result.pack
    if meta:
        from bsmapper.data.stats import summary_lines

        for line in summary_lines(meta["stats"]):
            typer.echo(line)


@lights_app.command("styles")
def lights_styles(
    dataset: Annotated[Path, typer.Argument(help="A built dataset (v1).")] = Path(
        "data/processed_v1"
    ),
    k: int = typer.Option(5, min=2, max=8, help="Number of styles."),
    workers: int = typer.Option(8, min=1),
    limit: int | None = typer.Option(None, help="Only the first N maps (for a quick test)."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Cluster the gold human lightshows into styles and write DATASET/light_styles.json.

    Light training (version 2 light prompt) reads the labels from it, and the trainer
    copies its style model (names, auto-pick, scene tables) into the checkpoint.
    """
    from bsmapper.lighting.style_build import STYLE_FILE, build

    _setup_logging(verbose)
    if not (dataset / "maps.parquet").is_file():
        raise typer.BadParameter(f"{dataset} is not a built dataset", param_hint="DATASET")
    out = build(dataset, k=k, workers=workers, limit=limit)
    stats = out["model"]["stats"]
    typer.echo(f"{stats['labelled']} of {stats['maps']} lightshows labelled; "
               f"excluded: {stats['excluded']}")  # fmt: skip
    for name, n in stats["sizes"].items():
        typer.echo(f"  {name:<8} {n:5d}  {stats['descriptions'].get(name, '')}")
    typer.echo(
        f"auto-pick val accuracy {stats['auto_pick_val_accuracy']:.3f} "
        f"(majority {stats['majority_val_accuracy']:.3f}, {stats['val_maps']} maps)"
    )
    typer.echo(f"wrote {dataset / STYLE_FILE}")


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


def _default_eval_data() -> Path:
    snapshot = Path("data/eval_v0_test")
    return snapshot if (snapshot / "dataset.json").is_file() else Path("data/processed")


@app.command()
def evaluate(
    model: Annotated[
        list[Path] | None,
        typer.Option("--model", "-m", help="Checkpoint to score; repeat for several."),
    ] = None,
    data: Annotated[
        Path | None,
        typer.Option(help="Built dataset; default data/eval_v0_test, else data/processed."),
    ] = None,
    split: str = typer.Option("test", help="Dataset split to score."),
    out: Annotated[
        Path | None, typer.Option(help="Output folder; default out/eval/<name>.")
    ] = None,
    name: str | None = typer.Option(None, help="Run name; default from the models and date."),
    stub: bool = typer.Option(True, "--stub/--no-stub", help="Also score the stub generator."),
    intensity: str | None = typer.Option(
        None, help="Model intensities, e.g. 0.5,0.8; 'none' for the default only."
    ),
    limit: int | None = typer.Option(None, help="Only the first N songs (by key)."),
    keys: str | None = typer.Option(None, help="Only these map keys, comma-separated."),
    modes: str | None = typer.Option(None, help="Only these modes, e.g. Standard."),
    difficulties: str | None = typer.Option(None, help="Only these difficulties."),
    seed: int = typer.Option(0, help="Generation seed."),
    device: str | None = typer.Option(None, help="cuda or cpu; default cuda if 4 GB are free."),
    lighting: bool = typer.Option(True, "--lighting/--no-lighting", help="Score lightshows."),
    style_songs: int = typer.Option(12, help="Songs for the style sweep (0 = off)."),
    batch_difficulties: int = typer.Option(
        2, help="Difficulties decoded per batch (0 = a whole mode, like generate)."
    ),
    resume: bool = typer.Option(False, help="Continue a run in the same output folder."),
    rescore: Annotated[
        Path | None,
        typer.Option(help="Score the saved maps of this earlier run folder again (no decoding)."),
    ] = None,
    compare: Annotated[
        tuple[Path, Path] | None,
        typer.Option(help="Diff two summary.json files instead of running: A B."),
    ] = None,
    system_a: str | None = typer.Option(None, help="With --compare: system of A."),
    system_b: str | None = typer.Option(None, help="With --compare: system of B."),
    snapshot: Annotated[
        Path | None,
        typer.Option(help="Copy --split of --data to this folder (a frozen test set) and exit."),
    ] = None,
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Score generated maps against held-out human maps (WS10). See docs/EVAL.md."""
    from bsmapper.eval import EvalConfig, format_compare, load_summary, run_eval, snapshot_split
    from bsmapper.eval.systems import checkpoint_label, intensity_supported
    from bsmapper.schema import Characteristic, Difficulty

    if compare:
        try:
            text = format_compare(
                load_summary(compare[0]), load_summary(compare[1]), system_a, system_b
            )
        except (OSError, ValueError, KeyError) as e:
            raise typer.BadParameter(str(e), param_hint="--compare") from e
        typer.echo(text)
        return
    _setup_logging(verbose)
    data = data or _default_eval_data()
    if not (data / "dataset.json").is_file():
        raise typer.BadParameter(f"{data} is not a built dataset", param_hint="--data")
    if snapshot:
        counts = snapshot_split(data, snapshot, split)
        typer.echo(f"Copied the {split} split of {data} to {snapshot}: {counts}")
        return
    models = list(model or [])
    for m in models:
        if not m.is_file():
            raise typer.BadParameter(f"{m} is not a file", param_hint="--model")
    if intensity is None:
        levels: list[float | None] = [0.5, 0.8] if intensity_supported() else [None]
    elif intensity.strip().lower() == "none":
        levels = [None]
    else:
        try:
            levels = [float(v) for v in intensity.split(",") if v.strip()]
        except ValueError as e:
            raise typer.BadParameter(str(e), param_hint="--intensity") from e
        if not intensity_supported():
            typer.secho(
                "Warning: this build's generator has no intensity option; scoring the default.",
                fg=typer.colors.YELLOW,
            )
            levels = [None]
    from datetime import date

    base = "-".join(checkpoint_label(m).replace("/", "_") for m in models) or "stub"
    if rescore:
        base = f"{rescore.name}-rescored"
    label = name or f"{base}-{date.today().isoformat()}"
    cfg = EvalConfig(
        data=data,
        out=out or Path("out/eval") / label,
        split=split,
        stub=stub,
        checkpoints=models,
        intensities=levels,
        keys=[k.strip() for k in keys.split(",")] if keys else None,
        limit=limit,
        modes=_parse_enum_list(modes, Characteristic, _MODE_ALIASES, "--modes") if modes else None,
        difficulties=_parse_enum_list(difficulties, Difficulty, _DIFF_ALIASES, "--difficulties")
        if difficulties
        else None,
        seed=seed,
        device=device,
        lighting=lighting,
        style_songs=style_songs,
        batch_difficulties=batch_difficulties,
        resume=resume,
        rescore=rescore,
        label=label,
    )
    summary = run_eval(cfg, progress=typer.echo)
    systems = summary["systems"]
    typer.echo("")
    typer.echo(f"{'system':<22} {'F1±2':>6} {'NPS/hum':>8} {'pct':>5} {'step':>6} {'err/100':>8}")
    for s, block in systems.items():
        g = block["groups"].get("all", {})

        def v(key, fmt, g=g):
            x = g.get(key)
            return format(x, fmt) if isinstance(x, int | float) else "–"

        typer.echo(
            f"{s:<22} {v('onset_f1_2', '6.3f'):>6} {v('nps_ratio', '8.2f'):>8} "
            f"{v('nps_percentile_median', '5.0f'):>5} {v('step_offset', '+6.2f'):>6} "
            f"{v('errors_per_100', '8.2f'):>8}"
        )
    typer.echo(f"\nSummary: {cfg.out / 'summary.json'}\nReport:  {cfg.out / 'report.html'}")


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    work_dir: Path = Path("out/server"),
    web_dist: str | None = typer.Option(None, help="Built web UI folder; default: web/dist."),
    mock: bool = typer.Option(False, help="Serve canned maps (UI development), not the pipeline."),
    mock_delay: float = typer.Option(0.4, help="Seconds per mock pipeline step."),
    generator: str = typer.Option("auto", help="Notes from: model, stub or auto."),
    model: str | None = typer.Option(None, help="Model checkpoint (see `generate --help`)."),
    device: str | None = typer.Option(None, help="cuda or cpu; default cuda when available."),
    tracker: str = typer.Option("auto", help="Beat tracker: auto, beat_this or librosa."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the local web UI and jobs API (WS9) on the generation pipeline (WS7)."""
    try:
        import uvicorn

        from bsmapper.server.app import create_app
        from bsmapper.server.mock import MockJobRunner
        from bsmapper.server.pipeline_runner import PipelineJobRunner
    except ImportError as e:
        raise typer.Exit(
            f"The server needs the server extra: pip install -e .[server] ({e})"
        ) from e

    _setup_logging(verbose)
    if mock:
        runner = MockJobRunner(step_delay=mock_delay)
    else:
        from bsmapper.generation import make_pipeline

        # The model loads once here and serves every job (one at a time, on the GPU).
        pipeline = make_pipeline(generator, "auto", model, device, tracker=tracker)
        setup = pipeline.setup
        typer.echo(
            f"Notes from {setup.generator}"
            + (f" ({setup.checkpoint})" if setup.checkpoint else f" ({setup.reason})")
        )
        runner = PipelineJobRunner(pipeline)
    application = create_app(runner, work_dir, Path(web_dist) if web_dist else None)
    typer.echo(f"bsmapper UI on http://{host}:{port}/  (API docs at /docs)")
    uvicorn.run(application, host=host, port=port, log_level="debug" if verbose else "info")
