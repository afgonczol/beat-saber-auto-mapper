"""Read a map folder (or zip) of any version into a ``schema.Beatmap``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bsmapper.formats import info as info_mod
from bsmapper.formats import v2, v3, v4
from bsmapper.formats._common import ParsedDifficulty, beat
from bsmapper.formats.report import MapFormatError, ReadReport
from bsmapper.formats.source import INFO_NAME, MapSource
from bsmapper.schema import Beatmap, BpmChange, DifficultyMap, Lightshow


def detect_major(data: Any) -> str:
    """Major format version ("2", "3" or "4") of a difficulty file."""
    if not isinstance(data, dict):
        return "2"
    version = data.get("version", data.get("_version"))
    if isinstance(version, str) and version[:1] in ("2", "3", "4"):
        return version[:1]
    if "colorNotesData" in data:
        return "4"
    if "colorNotes" in data:
        return "3"
    return "2"


def normalize_bpm_changes(base_bpm: float, changes: list[BpmChange]) -> list[BpmChange]:
    """Sorted changes with no-op entries removed (later entries win on the same beat)."""
    by_beat: dict[float, float] = {}
    for change in sorted(changes, key=lambda c: c.beat):
        by_beat[max(change.beat, 0.0)] = change.bpm
    result, current = [], base_bpm
    for b, bpm in sorted(by_beat.items()):
        if abs(bpm - current) > 1e-6:
            result.append(BpmChange(b, bpm))
            current = bpm
    return result


def editor_to_official(base_bpm: float, editor_changes: list[BpmChange]) -> list[BpmChange]:
    """Convert legacy ``_BPMChanges`` (placed in base-BPM beats) to tempo-warped beats.

    The game ignored these editor-only changes, so object times in such files are
    plain base-BPM beats. Expressing the changes in warped beats, and remapping the
    objects to match, puts notes back on the musical grid.
    """
    result = []
    prev_file = prev_warped = 0.0
    bpm = base_bpm
    for change in sorted(editor_changes, key=lambda c: c.beat):
        if change.beat <= 0:
            bpm = change.bpm
            result.append(BpmChange(0.0, bpm))
            continue
        warped = prev_warped + (change.beat - prev_file) * bpm / base_bpm
        result.append(BpmChange(beat(warped), change.bpm))
        prev_file, prev_warped, bpm = change.beat, warped, change.bpm
    return result


def _copy(ls: Lightshow, uses_chroma: bool) -> Lightshow:
    return Lightshow(
        list(ls.basic_events), list(ls.boost_events), list(ls.event_box_groups), uses_chroma
    )


def _most_common(items: list[Lightshow]) -> Lightshow:
    best, best_count = items[0], 0
    for item in items:
        count = sum(1 for other in items if other == item)
        if count > best_count:
            best, best_count = item, count
    return best


def read_map_report(path: Path | str) -> tuple[Beatmap, ReadReport]:
    """Read a map folder or zip. The report flags mods, dropped objects and warnings."""
    report = ReadReport()
    with MapSource(Path(path)) as source:
        if source.find(INFO_NAME) is None:
            raise MapFormatError(f"{path} has no Info.dat")
        try:
            info = info_mod.parse(source.load_json(INFO_NAME), report)
        except ValueError as exc:  # includes JSON errors
            raise MapFormatError(f"cannot parse Info.dat: {exc}") from exc
        report.info_version = info.version
        info_mod.collect_flags(info, report)
        base_bpm = info.beatmap.bpm

        audio_changes: list[BpmChange] = []
        if info.audio_data_filename and source.find(info.audio_data_filename):
            try:
                data = source.load_json(info.audio_data_filename)
                audio_changes = v4.parse_audio_data(data, base_bpm, report)
            except ValueError:
                report.warn(f"cannot parse {info.audio_data_filename}")

        lightshows: dict[str, Lightshow] = {}

        def v4_lightshow(name: str | None) -> Lightshow:
            if not name:
                return Lightshow()
            if name not in lightshows:
                try:
                    lightshows[name] = v4.parse_lightshow(source.load_json(name), report)
                except (FileNotFoundError, ValueError):
                    report.warn(f"cannot read lightshow file {name}")
                    lightshows[name] = Lightshow()
            return lightshows[name]

        # (ref, parsed, BPM changes the file's beats are in, (source, changes) it proposes)
        items: list[tuple[info_mod.DifficultyRef, ParsedDifficulty, list[BpmChange], Any]] = []
        for ref in info.refs:
            if source.find(ref.filename) is None:
                report.warn(f"missing difficulty file {ref.filename}")
                continue
            try:
                data = source.load_json(ref.filename)
            except ValueError:
                report.warn(f"cannot parse difficulty file {ref.filename}")
                continue
            major = detect_major(data)
            if major == "4":
                parsed = v4.parse_beatmap(data, report)
                parsed.lightshow = v4_lightshow(ref.lightshow_filename)
                own, proposal = (
                    audio_changes,
                    ("audio_data", audio_changes) if audio_changes else None,
                )
            else:
                parsed = (v3 if major == "3" else v2).parse(data, report)
                if parsed.bpm_changes:
                    own, proposal = parsed.bpm_changes, ("events", parsed.bpm_changes)
                elif parsed.editor_bpm_changes:
                    converted = editor_to_official(base_bpm, parsed.editor_bpm_changes)
                    own, proposal = [], ("custom", converted)
                else:
                    own, proposal = [], None
            report.difficulty_versions[f"{ref.characteristic}/{ref.difficulty}"] = parsed.version
            items.append((ref, parsed, own, proposal))

    # One tempo map for the whole song: official changes win over legacy editor ones.
    proposals = [p for *_, p in items if p is not None]
    chosen = next((p for p in proposals if p[0] != "custom"), proposals[0] if proposals else None)
    if chosen is None:
        shared: list[BpmChange] = []
    else:
        report.bpm_source = chosen[0]
        shared = normalize_bpm_changes(base_bpm, chosen[1])
    shared_timing = Beatmap(song_name="", bpm=base_bpm, bpm_changes=shared)

    difficulties = []
    for ref, parsed, own, proposal in items:
        own = normalize_bpm_changes(base_bpm, own)
        if own != shared:
            if proposal is None or proposal[0] != "custom":
                report.warn("difficulties disagree on BPM changes; times were remapped")
            own_timing = Beatmap(song_name="", bpm=base_bpm, bpm_changes=own)
            parsed.remap(
                lambda b, t=own_timing: shared_timing.seconds_to_beat(t.beat_to_seconds(b))
            )
        if parsed.rotations and ref.characteristic not in info_mod.ROTATING:
            report.drop("rotation_in_non_rotating_mode", len(parsed.rotations))
            parsed.rotations = []
        ls = parsed.lightshow or Lightshow()
        mods = ref.requirements + ref.suggestions
        uses_chroma = ls.uses_chroma or info_mod.CHROMA in mods
        difficulties.append(
            DifficultyMap(
                characteristic=ref.characteristic,
                difficulty=ref.difficulty,
                njs=ref.njs,
                njs_offset=ref.njs_offset,
                notes=parsed.notes,
                bombs=parsed.bombs,
                obstacles=parsed.obstacles,
                arcs=parsed.arcs,
                chains=parsed.chains,
                rotations=parsed.rotations,
                lightshow=_copy(ls, uses_chroma),
                label=ref.label,
            )
        )

    beatmap = info.beatmap
    beatmap.bpm_changes = shared
    beatmap.difficulties = difficulties
    # The most common lightshow becomes the shared one; difficulties that use it point to it.
    if difficulties:
        beatmap.lightshow = _most_common([d.lightshow for d in difficulties if d.lightshow])
        for dmap in difficulties:
            if dmap.lightshow == beatmap.lightshow:
                dmap.lightshow = None
    return beatmap, report


def read_map(path: Path | str) -> Beatmap:
    return read_map_report(path)[0]


class MapFolderReader:
    """``interfaces.MapReader`` for map folders and zips of any version."""

    def __init__(self) -> None:
        self.last_report: ReadReport | None = None

    def read(self, map_dir: Path) -> Beatmap:
        beatmap, self.last_report = read_map_report(map_dir)
        return beatmap
