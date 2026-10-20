# Map formats (WS1)

`bsmapper.formats` reads v2, v3 and v4 maps (folders or zips) into `schema.Beatmap` and
writes them back as v3 (default), v2 or v4.

```python
from bsmapper.formats import read_map_report, write_map, package_map

beatmap, report = read_map_report("downloads/1a2b3.zip")  # folder or zip
if report.needs_mods:  # Noodle / Mapping Extensions: skip for training
    ...
write_map(beatmap, "out/Song", version="v3")
package_map(beatmap, Path("song.mp3"), Path("out/Song"), zip_path=Path("out/Song.zip"))
```

`MapFolderReader` / `MapFolderWriter` implement the `MapReader` / `MapWriter` protocols.

## What the reader normalises

- **Times** are beats, rounded to 6 decimals to remove float noise. Grid snapping is left to
  the dataset builder.
- **BPM changes** become one song-wide `Beatmap.bpm_changes`, from v2 type-100 events, v3
  `bpmEvents` or v4 `BPMInfo.dat` regions. No-op changes are dropped. Legacy MMA2/ChroMapper
  `_customData._BPMChanges` are editor-only (the game ignores them and the object times stay
  in base-BPM beats), so the reader converts them to real changes and remaps the objects
  onto the musical grid. If difficulties disagree, they are remapped through seconds onto the
  first one's tempo map (with a warning).
- **Lightshows**: the most common per-difficulty lightshow becomes `Beatmap.lightshow`, and
  difficulties that use it get `lightshow=None`. Others keep their own.
- **Event box groups**: `EventBoxGroup.boxes` holds v3.3-shaped box dicts with every key filled
  in (`f` filter, `w`, `d`, ..., and the events list `e` for colour, `l` otherwise). fx boxes keep
  their float events inline under `l` rather than as indices into `_fxEventsCollection`.
- **Chroma**: event `_color` / `color` goes to `BasicEvent.color` (RGBA). Other event custom data
  is kept with leading underscores stripped (v3 names). `Lightshow.uses_chroma` is set by
  coloured events or a Chroma requirement/suggestion. The writer then adds a Chroma
  *suggestion*, never a requirement.
- **Rotations** in characteristics other than 90/360 are dropped.

## ReadReport

`noodle` / `mapping_extensions` / `chroma` flags, Info requirements and suggestions,
`bpm_source`, per-difficulty file versions, `dropped` counts by reason (off-grid notes,
precision walls, object custom data, waypoints, v4 NJS events and so on) and `warnings`
(missing files, unsupported characteristics). Mod-only objects are removed from the `Beatmap`.

## Lossy conversions

| Target | Lost |
|---|---|
| v2 | chains, note angle offsets, event box groups; rotations over ±60° are split into several events |
| v4 | late rotations: v4 stores absolute lanes per object, so rotations come back as early events at the next object |
| v4 light colour events | easing other than none/linear (v3 has no per-event easing) |
| all | waypoints, special-event keywords, v4 NJS events, note/obstacle custom data (none of these are in the schema) |

## Writer output

- v2/v3: `Info.dat` (v2.0.0) plus `<Difficulty><Characteristic>.dat` (2.6.0 / 3.3.0). Each file
  carries the difficulty's effective lightshow.
- v4: `Info.dat` (4.0.0), beatmap files, a shared `Lightshow.dat`, `<Diff><Char>.lightshow.dat`
  for difficulties with their own lightshow, and `BPMInfo.dat`. Song duration comes from the audio
  file next to the map (mutagen). Without it, the duration is estimated from the last object.
- 90/360 difficulties use `GlassDesertEnvironment`.

## Packaging

`package_map` converts the audio to `song.egg` (OGG Vorbis via ffmpeg; Vorbis input is copied
as-is). It uses the cover you pass, else art embedded in the audio file, else a generated
gradient PNG. Missing song name and artist are filled from the tags (or an `Artist - Title`
filename). It then writes the map and can zip it flat.
