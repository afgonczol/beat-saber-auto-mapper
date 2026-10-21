# Generation pipeline (WS7)

`bsmapper generate` and the web server (`bsmapper serve`) both run
`bsmapper.generation.Pipeline`:

| Stage | What happens |
| --- | --- |
| analyze | WS2 analysis. The song is padded with `BeatGrid.lead_in_s()` of silence so the bar before the first downbeat starts at 0 s (Beat Saber's beat 0). Map beat = grid beat + beats per bar. `timing.song_time_analysis` shifts the analysis the same way, so generators see `beat_times[i]` = map beat `i`. |
| generate | One `NoteGenerator.generate(analysis, request)` per mode x difficulty, hardest first. If a lower difficulty comes out at 95% or more of the next harder one's NPS, it is generated again at 80% of it. |
| rules | `postprocess`: snap to the 24-tick grid, dedupe, replay through `bsmapper.rules` and repair (turn or flip a note, move it to a nearby cell, soften a rotation) or drop anything that breaks a hard rule, then set NJS and offset (`njs.JUMP_TABLE`). Warnings are kept for the UI. |
| lighting | WS8 L0 lightshow (`light_beatmap`), v3 group lighting by default (WeaveEnvironment). |
| package | WS1 `package_map`: padded `song.egg`, cover (embedded art or a placeholder), v3 difficulty files with a v2 Info.dat, zip. Title and artist come from the tags, else from an `Artist - Title` filename. |

```bash
bsmapper generate song.mp3                                  # 5 Standard difficulties, v3 lights, zip
bsmapper generate song.mp3 --modes Standard,OneSaber,90,360 --difficulties Hard,Expert+ --style 0.7
bsmapper generate song.mp3 --lighting chroma --install      # warns if Plugins/Chroma.dll is missing
bsmapper serve            # web UI on the real pipeline; --mock for canned maps during UI work
```

## Generators

- **`StubGenerator`** is a placeholder, not the ML model. It picks note times from a
  per-difficulty grid by onset strength and section energy, up to a target NPS (Easy 1.0 to
  Expert+ 4.8, capped per beat so slow songs stay calm; One Saber gets 60%). Hands alternate,
  and each hand alternates down and up swings. Every candidate note is checked with
  `MapState.candidates_mask` before it is placed. Every few bars both hands may drift a
  column left or right (lean phrases), and from Normal up a full-height wall over the two
  columns on the other side often makes the player lean around it. For 90/360 it adds a rotation every few
  bars (90 stays within ±45°). Its maps say so: the mapper is `bsmapper (stub)` and the
  difficulty labels are `Stub <difficulty>`.
- **`ModelGenerator`** is the decoding loop for the WS6 model. It decodes window by window:
  the conditioning prefix plus the tail context, then token by token with the grammar mask,
  the request mask (forbidden object types and colours) and the rules mask (heads, cells and
  directions that `MapState` rejects at the current tick). Each step applies a density bias
  towards `target_nps`, then samples with temperature, top-p and a seeded RNG. WS6 only has
  to implement `NoteSampler`:

```python
class NoteSampler(Protocol):
    def encode(self, analysis: AudioAnalysis, start_beat: float, n_beats: int) -> Any: ...
    def next_logits(self, memory: Any, input_ids: Sequence[int]) -> np.ndarray: ...
```

## NJS and offset

NJS is fixed per difficulty (10, 10, 12, 16, 18). The offset brings the reaction time
(half jump duration in seconds, as the game computes it) to 1.0 / 0.9 / 0.8 / 0.7 / 0.6 s,
so a slow song does not give Expert+ two seconds of warning. `--njs` overrides the NJS;
the offset still follows the reaction-time target.

## Install

`--install` copies the map folder into `CustomLevels`. It replaces an earlier bsmapper
install of the same name and never touches a folder some other mapper made (it installs
next to it as `Name (2)`). Paths come from `~/.bsmapper/config.toml` (or `$BSMAPPER_CONFIG`):

```toml
[game]
beat_saber_dir = 'C:\Program Files (x86)\Steam\steamapps\common\Beat Saber'
# custom_levels = 'D:\Beat Saber\Beat Saber_Data\CustomLevels'  # optional override
```
