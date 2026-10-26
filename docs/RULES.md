# Playability rules (WS5)

`bsmapper.rules` checks a `DifficultyMap` for the classic automapper failures. The rules are
constraints and metrics, never the note generator. Each `Issue` has a severity: **error**
rules are hard constraints (decode masks), **warning** rules are only reported.

| Rule | Severity | What it catches |
|---|---|---|
| `parity` | error | Same-hand swings that need a reset: a repeated direction (turn < 60°), a cut the wrist can't reach with the expected forehand/backhand, or two same-hand notes at one time that point different ways |
| `stack_alignment` | error | Notes of one swing (a stack, window or slider) that don't lie along the cut: the line between them more than 30° off the cut axis, e.g. two notes side by side that both point up |
| `swing_interval` | error | Same-hand swings closer than the difficulty's minimum (Easy 180 ms … Expert+ 80 ms) |
| `swing_speed` | warning | Saber travel between swings above the difficulty's cells/s ceiling |
| `same_cell` | error | Two objects in one cell at the same time |
| `overlap` | error | Objects in one cell closer than 0.6 m along the track (NJS × Δt). Bomb–bomb pairs are exempt |
| `vision_block` | warning | A note 50–250 ms behind a centre (x 1–2, y 1) note or bomb on the same side |
| `crossover` | error | Red two or more columns right of blue at the same time |
| `note_in_wall` | error | A note inside an active wall |
| `nps_ceiling` | error | Notes/s over a 2 s window above the difficulty's ceiling (Easy 6 … Expert+ 20); reported once per dense passage |
| `wall_duration` | warning | Walls shorter than 13.8 ms |
| `dodge_gap` | error | Less than 150 ms between dodge walls that need the head in different places (left, right, crouch) |
| `wall_boxin` | error | Head-height walls of any length that close both centre columns (lean left and lean right) less than 300 ms apart, or one standing wall over both centre columns |
| `rotation_whiplash` | error | 360/90: total rotation swing above 90° within 1 s |
| `rotation_range` | error | 90Degree: cumulative rotation beyond ±45° |
| `rotation_step` | error | 360/90: a single rotation event over 60°. In 74k events of human 90/360 maps, none turns 75° or more, and 0.4–1% turn 60° |
| `arc_anchor` | warning | An arc whose head or tail is not on a note of its colour in that cell, with the same direction (a dot note fits any) |
| `arc_span` | warning | An arc whose tail is not after its head, that is longer than 3 s, or that skips a note of its hand |
| `arc_wall` | warning | An arc whose path (a straight line between its ends) runs through a wall |
| `chain_anchor` | error | A chain that does not start on a note with its direction, is longer than half a beat, has links that don't move along the cut, or skips a note of its hand |

## Parity model

Each hand alternates forehand and backhand. Forehand cuts must point within 115° of a
down-cut rotated 22.5° inward (DOWN_LEFT side for the right hand), backhand cuts within 115°
of the opposite direction. On top of that:

- A turn of 90° or less (DOWN → LEFT) may keep the parity as well as flip it. Human maps use
  such curves all the time, and strict alternation flags them as false positives.
- Stacks, windows and sliders are one swing: same-hand notes within 90 ms whose directions
  differ by 45° or less, each further along the cut.
- A slightly turned swing that starts further along the previous one (a slider with a gap)
  continues it.
- Dots are wildcards.
- Resets: a gap of 0.6 s or more, or a bomb within one column of the hand between two swings.

The checker tracks the *set* of parities each hand could be in, so it only reports a break
when no reading of the map avoids one.

## Stacks

A stack, window or slider is one swing, so its notes have to lie along the cut. A
playtest of a generated Maneater Expert had two reds side by side, both pointing up,
which no swing hits cleanly. `stack_alignment` measures the angle between the cut axis
and the line from one note of the swing to the next, and flags more than 30°. A knight
move under a straight cut is 26.6° and passes; a diagonal pair under a straight cut is 45°.

Calibrated on 38,123 same-hand pairs (both arrows, at most 45° apart, within 90 ms) in
1,530 human Standard Hard/Expert/Expert+ difficulties (800 maps): all but 4 lie within
30°, none between 30° and 90°. The 4 are side-by-side pairs in one Expert+ map (0.07% of
difficulties). In 69 generated difficulties, 6 (8.7%) had a pair over 30°. Generation
repairs the later note (another direction or cell) or drops it (`fix_with_rules`).

## Walls

`dodge_gap` leaves out walls shorter than 100 ms, which human maps use as flashes. Those
short walls still close a column, though. A playtest of a generated Stronger Expert had an
outer wall, then 1-tick walls in columns 2 and 1 within 145 ms, so no column was free.
`wall_boxin` covers that case. It counts every standing wall that reaches head height,
whatever its length, and flags one that closes the other centre column from the one a
wall closed less than 300 ms earlier (or at the same time). When `dodge_gap` already
flags a wall, `wall_boxin` doesn't report it again.

Calibrated on 205,367 walls in 1,483 human Hard/Expert/Expert+ difficulties (800 maps):

| Window | Human difficulties flagged |
|---|---|
| 0.15–0.2 s | 0 |
| 0.3 s | 2 (0.1%), both Expert+ side-switch flashes |
| 0.4 s | 23 (1.6%) |
| 0.5 s | 58 (3.9%) |

No human wall closes both centre columns on its own. Closing three or more of the four
columns within 0.3 s is common (5.6% of human difficulties: corridors with a centre wall),
so that is not a rule. Short centre walls are common too (23% of human difficulties), so
they are not banned either. Generation drops the later wall of a box-in (`fix_with_rules`).

## Arcs and chains

The span rules (`rules.spans`) look forward, since an arc's tail sits on a later note. So
they run over the whole map in `RuleChecker.check`, after the incremental rules, and are
not part of `MapState` or the decode masks. Generation repairs every span issue afterwards,
warnings included (`generation.postprocess.fix_spans`). The arc rules are warnings
because arcs are never hit, so a stray arc can't cause a miss; as errors they would put
modern human maps at 1.04 errors per 100 notes instead of 0.13. They are calibrated on 10,381 arcs and 1,381 chains
in 118 human v3 difficulties:

- 94% of human arcs pass every check. The rest are arc tails in empty air (5%), arcs
  through walls (1.7%, partly because the straight-line path ignores the arc's curve), and
  arcs longer than 3 s (0.5%).
- 46% of human arcs cross the centre cells, so arcs are not vision blocks.
- Human chains are at most a quarter beat long (two thirds are instant, length 0), and
  their links run along the cut.

The dataset build (`data.build`) leaves these rules out of its error filter, so datasets
v0 and v1 keep their meaning.

## Thresholds

All numbers live in `src/bsmapper/rules/thresholds.yaml`, in three layers merged in order:
`default` → `difficulties.<Difficulty>` → `characteristics.<Characteristic>`. Unknown keys
are an error. `severity.<rule>: off` disables a rule. Pass another file with
`RuleChecker(thresholds=path)`. `pyyaml` comes from the `ml` extra.

## API

```python
from bsmapper.rules import RuleChecker, MapState, summarize

issues = RuleChecker().check(dmap, beatmap)  # PlayabilityChecker
summarize(issues, len(dmap.notes))  # counts and rate per 100 notes, by rule
```

`MapState` is the incremental form. Every rule only looks back in time, and `check()` is
just `add()` over the time-sorted objects, so masks and the checker can't disagree.

```python
state = RuleChecker().state(dmap, beatmap)  # empty state; dmap only supplies mode, difficulty, NJS
state.check(obj)  # issues adding obj would cause; does not change the state
state.allowed(obj)  # no error-severity issue (pass severities=("error", "warning") to be stricter)
state.add(obj)  # commit; objects must come in time order
state.clone()  # independent copy (one per beam, or before a tentative add)

state.note_mask(beat, Color.RED)  # bool [x 4, y 3, direction 9], direction 8 = dot
state.bomb_mask(beat)  # bool [x 4, y 3]
state.candidates_mask(objs)  # bool per candidate; None entries are always allowed
```

For WS7: a factorized tokenizer (time, colour, x, y, direction tokens) can index
`note_mask` directly, e.g. `mask.any(axis=2)` for position tokens, then `mask[x, y]` for the
direction. Any other vocabulary can decode each next-token id into the object it would place
(or `None` for tokens that place nothing) and call `candidates_mask`. A mask can be all
false (for example at the density ceiling); the decoder must then emit a time shift instead.

## Calibration

Run against 79 unmodded maps in a local `CustomLevels` folder (228 difficulties, 150k notes):

| Maps | Difficulties | Errors per 100 notes | Difficulties at ≤ 0.5 |
|---|---|---|---|
| BeatSaver keys ≥ 0x10000 (2020+) | 143 | 0.13 | 133 |
| Older keys (2018–19) | 58 | 1.72 | 30 |
| Unkeyed (no mapper metadata, likely not hand-made) | 27 | 6.30 | 9 |

Most remaining modern-map errors are deliberate one-beat "down-down" resets at 120–180 BPM.
The 0.6 s reset gap stays strict on purpose, since repeated down-swings are a classic
automapper smell. `tests/rules/test_rules_customlevels.py` (marked slow) repeats this check
and requires < 0.5 errors per 100 notes on the modern maps.
