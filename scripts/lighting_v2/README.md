# Lighting v2 analysis scripts

These scripts reproduce the numbers in [docs/LIGHTING_V2.md](../../docs/LIGHTING_V2.md).

- They are analysis code, not part of the package.
- Run them from the repo root with the project's venv. The clustering also needs
  `scikit-learn`.
- `WORK` is any scratch folder; the scripts read and write it. Never commit it.

| Step | Command | Output |
|---|---|---|
| Human per-window stats and per-map features (gold maps, 12 workers, about 5 min) | `python scripts/lighting_v2/human_stats.py data/processed_v1 WORK/human.pkl` | `human.pkl` |
| Energy-binned quantiles | `python scripts/lighting_v2/human_bins.py WORK` | `human_bins.json` |
| Style clusters (no k: silhouette per k; with k: clusters) | `python scripts/lighting_v2/styles.py WORK 6` | `styles_k6.json` |
| Generate a song once and cache it | `python scripts/lighting_v2/gen_cache.py song.ogg WORK/stronger.pkl` | `stronger.pkl` |
| Re-decode the lights for seeds × decode masks | `python scripts/lighting_v2/sweep.py WORK/stronger.pkl WORK/toxic.pkl` | `sweep.pkl` |
| Section table of one cached lightshow vs. humans | `python scripts/lighting_v2/diagnose.py WORK/stronger.pkl WORK/human_bins.json default=base filtered=base+filter` | stdout |
| Before/after table (also assigns styles) | `python scripts/lighting_v2/summary.py WORK` | `summary.json` |

Notes:

- `gen_cache.py` and `sweep.py` load `checkpoints/note_v1/best.pt`. Set
  `BSMAPPER_CHECKPOINT` to use another checkpoint.
- The "before" rows need the renderer as it was before WS8b. Write it into `WORK` first:

```bash
git show 678f5cc:src/bsmapper/lighting/render.py > WORK/render_old.py
```
