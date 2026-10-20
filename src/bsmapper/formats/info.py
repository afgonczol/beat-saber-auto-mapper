"""Info.dat in the v2 (``_version`` 2.x) and v4 (``version`` 4.x) layouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bsmapper.formats._common import (
    as_dict,
    as_list,
    beat,
    out,
    string_list,
    strip_underscores,
)
from bsmapper.formats.report import MapFormatError, ReadReport
from bsmapper.schema import Beatmap, Characteristic, Difficulty, DifficultyMap

V2_VERSION = "2.0.0"
V4_VERSION = "4.0.0"
ROTATING = (Characteristic.DEGREE_90, Characteristic.DEGREE_360)
ALL_DIRECTIONS_ENVIRONMENT = "GlassDesertEnvironment"
CHROMA = "Chroma"


@dataclass(slots=True)
class DifficultyRef:
    """One difficulty entry of Info.dat."""

    characteristic: Characteristic
    difficulty: Difficulty
    filename: str
    njs: float = 16.0
    njs_offset: float = 0.0
    label: str | None = None
    lightshow_filename: str | None = None  # v4 only
    requirements: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class InfoData:
    version: str
    beatmap: Beatmap  # metadata only, no difficulties yet
    refs: list[DifficultyRef]
    audio_data_filename: str | None = None  # v4 BPMInfo.dat


@dataclass(slots=True)
class InfoEntry:
    """What a writer needs to list one difficulty in Info.dat."""

    dmap: DifficultyMap
    filename: str
    uses_chroma: bool = False
    lightshow_filename: str | None = None


def is_v4(data: dict[str, Any]) -> bool:
    return str(data.get("version", "")).startswith("4")


def _make_ref(
    report: ReadReport,
    characteristic: Any,
    difficulty: Any,
    filename: Any,
    njs: Any,
    offset: Any,
    custom: Any,
    lightshow: Any = None,
) -> DifficultyRef | None:
    try:
        char, diff = Characteristic(str(characteristic)), Difficulty(str(difficulty))
    except ValueError:
        report.warn(f"skipping unsupported difficulty {characteristic}/{difficulty}")
        return None
    if not filename:
        report.warn(f"{char}/{diff} has no beatmap file")
        return None
    data = strip_underscores(as_dict(custom))
    label = data.get("difficultyLabel")
    return DifficultyRef(
        char,
        diff,
        str(filename),
        njs=beat(njs or 0),
        njs_offset=beat(offset or 0),
        label=str(label) if label else None,
        lightshow_filename=str(lightshow) if lightshow else None,
        requirements=string_list(data.get("requirements")),
        suggestions=string_list(data.get("suggestions")),
    )


def parse(data: dict[str, Any], report: ReadReport) -> InfoData:
    if not isinstance(data, dict):
        raise MapFormatError("Info.dat is not a JSON object")
    return _parse_v4(data, report) if is_v4(data) else _parse_v2(data, report)


def _parse_v2(data: dict[str, Any], report: ReadReport) -> InfoData:
    try:
        bpm = float(data["_beatsPerMinute"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MapFormatError("Info.dat has no valid _beatsPerMinute") from exc
    beatmap = Beatmap(
        song_name=str(data.get("_songName", "")),
        bpm=bpm,
        audio_path=str(data.get("_songFilename", "song.egg")),
        song_sub_name=str(data.get("_songSubName", "")),
        song_author=str(data.get("_songAuthorName", "")),
        mapper=str(data.get("_levelAuthorName", "")),
        cover_path=str(data["_coverImageFilename"]) if data.get("_coverImageFilename") else None,
        preview_start=beat(data.get("_previewStartTime", 0)),
        preview_duration=beat(data.get("_previewDuration", 0)),
        environment=str(data.get("_environmentName", "DefaultEnvironment")),
        song_time_offset=beat(data.get("_songTimeOffset", 0)),
    )
    refs = []
    for beatmap_set in as_list(data.get("_difficultyBeatmapSets")):
        beatmap_set = as_dict(beatmap_set)
        for raw in as_list(beatmap_set.get("_difficultyBeatmaps")):
            raw = as_dict(raw)
            ref = _make_ref(
                report,
                beatmap_set.get("_beatmapCharacteristicName"),
                raw.get("_difficulty"),
                raw.get("_beatmapFilename"),
                raw.get("_noteJumpMovementSpeed"),
                raw.get("_noteJumpStartBeatOffset"),
                raw.get("_customData"),
            )
            if ref:
                refs.append(ref)
    return InfoData(str(data.get("_version", V2_VERSION)), beatmap, refs)


def _parse_v4(data: dict[str, Any], report: ReadReport) -> InfoData:
    song, audio = as_dict(data.get("song")), as_dict(data.get("audio"))
    try:
        bpm = float(audio["bpm"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MapFormatError("Info.dat has no valid audio.bpm") from exc
    environments = string_list(data.get("environmentNames"))
    entries = [as_dict(raw) for raw in as_list(data.get("difficultyBeatmaps"))]

    environment = environments[0] if environments else "DefaultEnvironment"
    for raw in entries:  # the environment of the first non-rotating difficulty
        idx = raw.get("environmentNameIdx", 0)
        if raw.get("characteristic") not in ("90Degree", "360Degree") and isinstance(idx, int):
            if 0 <= idx < len(environments):
                environment = environments[idx]
            break

    mappers = (
        string_list(as_dict(entries[0].get("beatmapAuthors")).get("mappers")) if entries else []
    )
    beatmap = Beatmap(
        song_name=str(song.get("title", "")),
        bpm=bpm,
        audio_path=str(audio.get("songFilename", "song.egg")),
        song_sub_name=str(song.get("subTitle", "")),
        song_author=str(song.get("author", "")),
        mapper=", ".join(mappers),
        cover_path=str(data["coverImageFilename"]) if data.get("coverImageFilename") else None,
        preview_start=beat(audio.get("previewStartTime", 0)),
        preview_duration=beat(audio.get("previewDuration", 0)),
        environment=environment,
    )
    refs = []
    for raw in entries:
        ref = _make_ref(
            report,
            raw.get("characteristic"),
            raw.get("difficulty"),
            raw.get("beatmapDataFilename"),
            raw.get("noteJumpMovementSpeed"),
            raw.get("noteJumpStartBeatOffset"),
            raw.get("customData"),
            raw.get("lightshowDataFilename"),
        )
        if ref:
            refs.append(ref)
    audio_data = audio.get("audioDataFilename")
    return InfoData(
        str(data.get("version", V4_VERSION)), beatmap, refs, str(audio_data) if audio_data else None
    )


def basename(path: str | None) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1] if path else ""


def write_v2(beatmap: Beatmap, entries: list[InfoEntry]) -> dict:
    sets: dict[Characteristic, list[dict]] = {}
    for entry in entries:
        dmap = entry.dmap
        item: dict[str, Any] = {
            "_difficulty": dmap.difficulty.value,
            "_difficultyRank": dmap.difficulty.rank,
            "_beatmapFilename": entry.filename,
            "_noteJumpMovementSpeed": out(dmap.njs),
            "_noteJumpStartBeatOffset": out(dmap.njs_offset),
        }
        custom: dict[str, Any] = {}
        if dmap.label:
            custom["_difficultyLabel"] = dmap.label
        if entry.uses_chroma:
            custom["_suggestions"] = [CHROMA]
        if custom:
            item["_customData"] = custom
        sets.setdefault(dmap.characteristic, []).append(item)

    return {
        "_version": V2_VERSION,
        "_songName": beatmap.song_name,
        "_songSubName": beatmap.song_sub_name,
        "_songAuthorName": beatmap.song_author,
        "_levelAuthorName": beatmap.mapper,
        "_beatsPerMinute": out(beatmap.bpm),
        "_songTimeOffset": out(beatmap.song_time_offset),
        "_shuffle": 0,
        "_shufflePeriod": 0.5,
        "_previewStartTime": out(beatmap.preview_start),
        "_previewDuration": out(beatmap.preview_duration),
        "_songFilename": basename(beatmap.audio_path),
        "_coverImageFilename": basename(beatmap.cover_path),
        "_environmentName": beatmap.environment,
        "_allDirectionsEnvironmentName": ALL_DIRECTIONS_ENVIRONMENT,
        "_difficultyBeatmapSets": [
            {"_beatmapCharacteristicName": char.value, "_difficultyBeatmaps": items}
            for char, items in sets.items()
        ],
    }


def write_v4(
    beatmap: Beatmap, entries: list[InfoEntry], duration_s: float, audio_data_filename: str
) -> dict:
    environments = [beatmap.environment]
    if any(e.dmap.characteristic in ROTATING for e in entries):
        environments.append(ALL_DIRECTIONS_ENVIRONMENT)
    audio_file = basename(beatmap.audio_path)

    items = []
    for entry in entries:
        dmap = entry.dmap
        item: dict[str, Any] = {
            "characteristic": dmap.characteristic.value,
            "difficulty": dmap.difficulty.value,
            "beatmapAuthors": {"mappers": [beatmap.mapper], "lighters": [beatmap.mapper]},
            "environmentNameIdx": 1 if dmap.characteristic in ROTATING else 0,
            "beatmapColorSchemeIdx": -1,
            "noteJumpMovementSpeed": out(dmap.njs),
            "noteJumpStartBeatOffset": out(dmap.njs_offset),
            "beatmapDataFilename": entry.filename,
            "lightshowDataFilename": entry.lightshow_filename,
        }
        custom: dict[str, Any] = {}
        if dmap.label:
            custom["difficultyLabel"] = dmap.label
        if entry.uses_chroma:
            custom["suggestions"] = [CHROMA]
        if custom:
            item["customData"] = custom
        items.append(item)

    return {
        "version": V4_VERSION,
        "song": {
            "title": beatmap.song_name,
            "subTitle": beatmap.song_sub_name,
            "author": beatmap.song_author,
        },
        "audio": {
            "songFilename": audio_file,
            "songDuration": out(duration_s),
            "audioDataFilename": audio_data_filename,
            "bpm": out(beatmap.bpm),
            "lufs": 0,
            "previewStartTime": out(beatmap.preview_start),
            "previewDuration": out(beatmap.preview_duration),
        },
        "songPreviewFilename": audio_file,
        "coverImageFilename": basename(beatmap.cover_path),
        "environmentNames": environments,
        "colorSchemes": [],
        "difficultyBeatmaps": items,
    }


def collect_flags(info: InfoData, report: ReadReport) -> None:
    """Fill requirement/suggestion flags of the report from Info.dat."""
    for ref in info.refs:
        for name in ref.requirements:
            if name not in report.requirements:
                report.requirements.append(name)
        for name in ref.suggestions:
            if name not in report.suggestions:
                report.suggestions.append(name)
    mods = report.requirements + report.suggestions
    if "Noodle Extensions" in report.requirements:
        report.noodle = True
    if "Mapping Extensions" in report.requirements:
        report.mapping_extensions = True
    if CHROMA in mods:
        report.chroma = True
