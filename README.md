# bsmapper: Beat Saber auto mapper

> **Status: early but usable.** `bsmapper generate` makes playable maps with the trained
> note_v1 model (weights are local in `checkpoints/`, not published). **New here? Start with
> [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).** See also [docs/PLAN.md](docs/PLAN.md)
> and [docs/GENERATION.md](docs/GENERATION.md).

bsmapper turns any song into a playable Beat Saber map using a transformer trained on curated
community maps:

- **Every difficulty**: Easy, Normal, Hard, Expert and Expert+ from one model, with density
  following the music
- **Every mode**: Standard, One Saber, 90° and 360°
- **Lightshows**: vanilla events, v3 group lighting or Chroma RGB, synced to song sections and onsets
- **Playable by construction**: parity, vision-block and density rules are enforced while decoding
- **CLI and local web UI** with an in-browser 3D preview and one-click install to `CustomLevels`

```bash
bsmapper generate song.mp3 --modes Standard,360Degree --lighting chroma --zip
bsmapper serve   # http://127.0.0.1:8765
```

## Development

```bash
pip install -e ".[dev,audio,ml,server]"
pytest
```

Agents and contributors: start with [CLAUDE.md](CLAUDE.md) and [docs/PLAN.md](docs/PLAN.md).

## Data and copyright

Training uses maps from [BeatSaver](https://beatsaver.com) and excludes automapped maps. The
dataset, audio and map files are kept locally and are **never** committed or redistributed.

## License

MIT
