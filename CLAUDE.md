# Agent guide

Read `docs/PLAN.md` first. It defines the workstreams (WS1–WS10), milestones and who owns what.

## Rules
- **Contracts**: `src/bsmapper/schema.py` and `src/bsmapper/interfaces.py` are shared by every
  workstream. Do not change them inside a feature PR. Open a separate small "contract change"
  PR, and say why in its description.
- **Ownership**: work only inside your workstream's package (`src/bsmapper/<pkg>/`), its tests
  (`tests/<pkg>/`) and its docs. If you need something from another package, open an issue
  or use a fixture/stub instead of editing that package.
- **Branches**: `ws/<n>-<short-name>`, one PR per work package into `main`. Reference the issue.
- **Never commit** data, map zips, audio, model checkpoints or generated output (`data/`,
  `checkpoints/`, `out/` are gitignored). Tiny hand-written JSON fixtures in
  `tests/fixtures/` are fine.
- **GPU**: there is one RTX 3090. Long training or dataset builds run in the background, one
  at a time. Unit tests must run on CPU in seconds.
- Tests needing audio files, network or a GPU are marked `@pytest.mark.slow` (skipped by default).

## Commands
```bash
pip install -e ".[dev,audio,ml,server]"
ruff check . && ruff format --check .
pytest
```

## Conventions
- Times inside the codebase are in **beats** (floats). Convert to seconds only at the audio boundary,
  via `Beatmap.beat_to_seconds`.
- The beat grid resolution is 24 ticks per beat everywhere (tokenizer, features, snapping).
- Use dataclasses from `schema.py`, not raw JSON dicts, outside of `formats/`.
