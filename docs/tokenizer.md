# Tokenizer (WS4)

`bsmapper.tokenizer` turns a `DifficultyMap` into 16-beat token windows for the note
model, and a `Lightshow` into windows in a separate lighting vocabulary. Decoding is
exact: after `snap_map`, `decode(encode(m)) == canonical_order(m)` for every map. The
tests check this on every fixture difficulty and on 150 random maps (plus 150 random
"messy" maps after snapping).

```python
from bsmapper.tokenizer import Conditioning, MapTokenizer, canonical_order, snap_map

dmap, report = snap_map(dmap)  # WS3b: reject if report.off_grid_ratio is high
tok = MapTokenizer(context_events=16)
cond = Conditioning.from_map(dmap, nps=nps, style=tech_score)
windows = tok.encode(dmap, cond, n_windows=song_windows)
ids = windows[k].input_ids()  # BOS prefix context SEP body EOS
loss_from = windows[k].body_offset  # train on body + EOS only
assert tok.decode(windows, meta=dmap) == canonical_order(dmap)
```

## Grid

* Times use **24 ticks per beat** (`TICKS_PER_BEAT`), the same grid as WS2's beat-synchronous
  features. Beats decode as `round(tick / 24, 6)`, which is how the WS1 readers write beats, so
  `7.333333` (tick 176) round-trips exactly.
* A window is 16 beats = 384 ticks. Window *k* covers ticks `[384k, 384(k+1))`. An object
  belongs to the window its start falls in. Its span (wall duration, arc or chain tail) may
  reach into later windows.

## Note vocabulary (655 tokens)

| Group | Tokens | Meaning |
|---|---|---|
| `PAD BOS EOS SEP` | 4 | ids 0..3 in both vocabularies |
| `CHAR_*` | 6 | `Characteristic` (Standard, OneSaber, NoArrows, 90Degree, 360Degree, Lightshow) |
| `DIFF_*` | 5 | `Difficulty` |
| `NPS_0..12`, `NPS_NONE` | 14 | notes/second buckets, edges 1,2,…,10,12,14 |
| `STYLE_0..4`, `STYLE_NONE` | 6 | `min(int(style*5), 4)` of the flowy (0) ↔ technical (1) slider |
| `BOMBS/WALLS/ARCS/CHAINS_ON/OFF` | 8 | mode flags: the map uses / may use this object type |
| `TIME_0..383` | 384 | tick within the window |
| `ROT_±15..±165`, `ROT_+180` | 23 | rotation event degrees |
| `WALL BOMB RED BLUE ARC CHAIN` | 6 | object heads |
| `POS_x_y` | 12 | grid cell (x 0..3, y 0..2) |
| `DIR_*` | 9 | `CutDirection` (`DIR_ANY` = dot) |
| `ANGLE_±5..±45` | 18 | note angle offset (only when nonzero) |
| `LATE` | 1 | late rotation (only when late) |
| `WX_0..3 WW_1..4 WH_y_h` | 4+4+15 | wall x, width, (y 0..2, height 1..5) |
| `DUR_1..16 DURT_0..23` | 16+24 | spans: whole beats, then remaining ticks |
| `FRAC_0..60` | 61 | k/20 in 0..3: arc multipliers and chain squish |
| `ANCHOR_0..2` | 3 | arc mid-anchor mode |
| `SLICE_1..32` | 32 | chain slice count |

## Sequence layout

```
BOS  CHAR DIFF NPS STYLE BOMBS WALLS ARCS CHAINS  <context>  SEP  <body>  EOS
```

* **Prefix**: fixed length (`PREFIX_LEN = 8`). Build it with `Conditioning.from_map` for
  training (the flags say which object types the map uses) or `Conditioning.from_request`
  for generation. `drop_style()` and `drop_nps()` swap in the `_NONE` tokens. WS6 calls them
  at random during training so the model also works with no preference and supports
  classifier-free guidance.
* **Context**: the last `context_events` objects (default 16) of window *k-1*, with TIME
  relative to window *k-1*. Window 0 has an empty context. At generation time use
  `tail_context(previous_body, NOTE_GRAMMAR, n)`, which gives exactly what training saw.
  Only the immediately previous window is used, so after an empty window the context is empty.
* **Body** grammar (`TIME` is emitted only when the tick changes and must strictly increase):

```
note   RED|BLUE POS DIR [ANGLE]
bomb   BOMB POS
wall   WALL WX WW WH <span>
arc    ARC RED|BLUE POS DIR FRAC(head mult) <span> POS DIR FRAC(tail mult) ANCHOR
chain  CHAIN RED|BLUE POS DIR <span> POS SLICE FRAC(squish)
rot    ROT_d [LATE]
<span> = DUR_16* [DUR_b] DURT_r      (b < 16; total = 16n + b beats + r ticks)
```

Objects on the same tick come in **canonical order**: sorted by token ids. That puts
rotations first, then walls, bombs, red notes, blue notes, arcs and chains, and within a type
it sorts by position, then direction. `decode` returns object lists in this order, and
`canonical_order(m)` sorts a map the same way. Gameplay objects on one tick don't depend on
their order, so the reorder is harmless. All fixture maps are already in canonical order.

### Masking during generation (WS7, WS5)

`NOTE_GRAMMAR.parser()` gives an incremental `Parser`. Feed it generated ids one at a time.
`parser.allowed_mask()` is a boolean mask of the grammatically valid next tokens. The tests
check that it allows every token the encoder emits. `parser.time`, `parser.events` (the
completed objects so far) and `parser.partial` (the object being read) are the token-level
state that WS5's rule masks can inspect. An object with an optional trailing modifier
(`ANGLE`, `LATE`) stays in `partial` until the next token shows the modifier is absent.

## Lighting vocabulary (500 tokens)

```
TIME_t  EVT_type VAL_value [FLOAT_k] [COLOR_i]      basic event
TIME_t  BOOST_ON | BOOST_OFF                        colour boost
```

`EVT` covers event types 0..19 and 40..43. `VAL` covers 0..31. `FLOAT_k` = k/20 in 0..2 and
is omitted when the float value is 1.0. `COLOR_i` indexes a 13-colour palette (12 hues and
white). Lighting windows have no prefix. Light events on the same tick keep their original
order (boosts first), because for lights the last event wins.

## What does not round-trip, and how it's handled

`encode` is strict and raises `OffGridError` for anything it could not decode identically.
`snap_map` (gameplay) and `prepare_lightshow` (lighting) make a map representable and report
what they changed in a `SnapReport`. `off_grid` counts objects moved by more than
`tolerance_ticks` (default 0.1). `dropped` and `quantized` count changes by reason.

| Input | Handling |
|---|---|
| Off-grid times (e.g. 1/5 or 1/7 beat, or precision noise) | snapped to the nearest 1/24 beat; counted in `off_grid` if moved > 0.1 tick. WS3b should reject maps with a high `off_grid_ratio`. |
| Wall durations, arc/chain tails off grid | the end time is snapped, so the span stays consistent |
| Objects before beat 0 | dropped (`before_start`) |
| Notes/bombs/arc or chain ends outside the 4×3 grid | dropped (`*_position`). Mapping Extensions maps are already flagged and dropped by WS1. |
| Walls with negative duration, or outside x 0..3 / width 1..4 / y 0..2 / height 1..5 | dropped (`wall_negative_duration`, `wall_shape`) |
| Arcs/chains whose tail comes before the head | dropped (`arc_reversed`, `chain_reversed`) |
| Angle offsets | rounded to 5° and clamped to ±45° (`angle_offset`) |
| Arc multipliers, chain squish | rounded to 0.05 and clamped to 0..3 |
| Arc mid-anchor outside 0..2 | set to 0 |
| Chain slice count | clamped to 1..32 |
| Rotations | rounded to 15° and wrapped into −165..+180. Those that become 0 are dropped (`rotation_zero`). |
| NJS, NJS offset, label | not tokenized. `decode(..., meta=dmap)` copies them back. WS7 picks NJS from tables. |
| `DifficultyMap.lightshow` | not part of the note tokens. Use `LightingTokenizer`. |
| Light event types outside the vocabulary, events before beat 0 | dropped |
| Light values > 31, float values off 0.05 or > 2 | clamped / rounded |
| Chroma colours | snapped to the nearest palette colour, alpha set to 1 (**lossy**) |
| Light `custom_data` (lightID, gradients, …) | dropped (**lossy**) |
| v3 event box groups | not tokenized (dropped and counted). WS8 renders group lighting from basic events. |
| `Lightshow.uses_chroma` | derived on decode as "some event has a colour" |

## Departures from PLAN §5 WS4

* Arcs and chains start with their own `ARC`/`CHAIN` head and repeat the head colour, position
  and direction. They don't attach to a preceding note, because real maps contain arcs with no
  note at the head. With the canonical order, the head note (when there is one) comes just
  before, so the model can copy it.
* Wall durations use exact spans (`DUR`/`DURT`) instead of a lossy `DUR_bucket`.
* `ROT` covers ±15..±180 (not only ±60), so v3/v4 rotations of any multiple of 15 survive.
  WS5 can still mask large rotations at decode time.
* `MODE_FLAGS` are four on/off tokens (bombs, walls, arcs, chains), matching
  `GenerationRequest.allow_*`.
