# Lighting (WS8)

`bsmapper.lighting` turns an `AudioAnalysis` into a lightshow. This first version is
**L0**, the rule baseline. It is also the fallback whenever no ML lighting checkpoint
exists. L1, the ML lightshow, will replace the rules but reuse the same renderers.

```python
from bsmapper.lighting import RuleLightingGenerator, light_beatmap

lightshow = RuleLightingGenerator().generate(analysis, beatmap, "v3")  # LightingGenerator
light_beatmap(beatmap, analysis, "v3")  # also sets beatmap.environment
```

There are two pipeline stages. First, `rule_lightshow` writes a vanilla lightshow, made of
basic events plus colour boost, and a `SectionPlan` for each section. Then a renderer turns
that into the requested style.

## Rules (L0)

Each section gets a **mood**: calm, mid or high. The mood comes from the section label
and its *intensity*. Intensity blends the analysed energy with its min–max rescaled value
across the song. Section energies are relative to the loudest section, so on a loud master
they bunch up between 0.9 and 1.0. Rescaling restores the contrast between sections.

| When | What |
|---|---|
| Section start | Ring spin, and laser speed set by mood (1 / 3 / 6). Centre lights switch to the section's lead colour, which alternates between red and blue from section to section. Calm sections start dark. |
| Section start, intensity jump ≥ 0.25 or a high-energy section | Ring zoom |
| Downbeats | Back lights fade every other bar in calm sections and flash every bar otherwise; white on 4-bar phrase starts when high. Ring lights fade every 4 bars in calm sections and flash on downbeats otherwise, plus a hit on the half bar when high. |
| Onsets | Left and right lasers alternate. Onsets are peaks of `onset_strength`, thinned to at least 1 / 0.5 / 0.25 beats apart by mood, with brightness from peak strength. Without onsets, lasers fall back to beats. |
| Drops | Colour boost for the whole section. A section counts as a drop if it is labelled "drop", or if it is high-energy and entered with an intensity jump of 0.3 or more. |
| Song end | All lights off, lasers stopped, boost off |

Each section also gets a **palette** chosen for its mood. Palettes never repeat
back-to-back, and returning choruses cycle through the high-energy palettes. Vanilla and
v3 output can only pick between the environment's red, blue and white, so they use the
alternating lead colour. Chroma uses the palette's RGB.

Times are converted from audio seconds to map beats with `Beatmap.seconds_to_beat`,
including BPM changes, then snapped to the 24-tick grid. If the packaged song has silence
prepended (`BeatGrid.lead_in_s()`), pass that amount as
`RuleLightingGenerator(time_offset_s=...)`.

Every threshold lives in `RuleConfig`.

## Styles

| Style | Output | Environment |
|---|---|---|
| `v3` (default) | Event box groups, plus the basic events the environment still reacts to, plus boost. Works in the unmodded game. | A profiled group-lighting environment: Weave (default), The Second or EDM |
| `vanilla` | Basic events only | A legacy environment (`DefaultEnvironment` by default) |
| `chroma` | Basic events with RGB from the section palettes, plus gradients on the centre lights. Info.dat lists Chroma under `_suggestions` only, never `_requirements`, so the map loads without the mod and shows vanilla colours. | Legacy |

`generate()` lights the map for `beatmap.environment`. If v3 is requested for an
environment with no profile, it writes vanilla events and logs a warning, because event
box groups do nothing in legacy environments. `light_beatmap()` and `environment_for()`
pick a suitable environment instead.

### v3 renderer

- **Light events** (types 0–4) become one colour box per group of the channel. The box
  filter covers all lights.
  - *On*: an instant event.
  - *Flash*: an instant event at 1.2× brightness that settles over 0.25 beat.
  - *Fade*: an instant event at 1.2× brightness that fades to black over 1 beat.
  - *Transition*: a transition event.
  - A per-channel `wave` (beat distribution) ripples a hit across the group's lights.
- **Ring spin** (type 8) turns the spin groups one full circle over 2 beats, then glides
  them back to the rest pose. The direction alternates between spins. A turn always
  takes the full 2 beats: spins and zooms that come while a turn is running are ignored.
  Squeezing turns into the gap whipped the inner square round in a playtest of Stronger
  (see [LIGHTING_V2.md](LIGHTING_V2.md)).
- **Ring zoom** (type 9) briefly fans the spin groups' lights out (3× the spread), then
  glides them back to rest. If an environment handles type 9 natively (The Second), the
  zoom event is passed through instead.
- **Laser speed** (types 12 and 13) becomes back-and-forth sweeps around the rest pose on
  the sweep groups. Faster speeds give a shorter period and a wider swing, capped at the
  motion's `max_angle` (30° by default). Speed 0 glides back to rest. The right side is
  mirrored.
- **Rest pose.** Rotated lights are never left parked: every move ends back at the motion's
  `rest` angle (by default 0, the environment's own pose) with no spread, or hands over
  directly to the next move. In a playtest on Weave, spins parked the inner square at
  45°/90°, and a zoom left it fanned 60° for a whole 80-beat drop. Those lights sat across
  the track and hid the notes.

### Environment profiles

Profiles are data files in `src/bsmapper/lighting/profiles/<Environment>.json`. To add an
environment, add a file; no code changes are needed.

- `groups`: group id → name, light count, colour support and rotation axes `[x, y, z]`.
  These come from KivalEvan/BeatSaber-JSMap.
- `channels`: vanilla light type → the colour groups that play it, plus `wave` (beats).
- `spin`, `left_sweep`, `right_sweep`: the groups to rotate, the `axis` (0 = x, 1 = y,
  2 = z) and the `spread` (degrees from the first light to the last). Optional: `rest`
  (default 0) and `max_angle` (default 30) to tune a motion after a playtest.
- `basic_types`: basic event types the environment still reacts to. They are passed
  through unchanged.

`EnvironmentProfile.validate()` checks that channel groups exist and support colour, and
that motion groups can rotate on the chosen axis. The tests run it on every bundled
profile.

The channel-to-group assignments started as a guess from the group names. Axes, waves and
spreads were then checked against human-made maps (below). They still need checking in-game.

### Calibration against human-made maps

The local dataset contains 20 maps (zips) for these environments: 8 Weave, 9 The Second
and 3 EDM. Each was read with `read_map_report`, using its fullest lightshow. Every
statistic is computed per map and then the median is taken across maps, so one dense map
(one Weave map has 39k box groups) cannot dominate. Axis counts only use rotation boxes
that move to a non-zero angle. Many mappers reset X and Y together at the same beat, and
those resets would otherwise show a fake 50/50 split. Single events on group ids beyond
the environment's range come from one template map and are ignored.

| | Weave (8) | The Second (9) | EDM (3) |
|---|---|---|---|
| Colour groups used | all 16 in 8/8 maps. Inner square 4–7 is busiest (~11% of colour box groups each for 4/5), distant 12–15 least (~4%) | all 14. Runway 2/3 busiest (~12% each), rings small 1 ~17%, top/bottom lasers 2.5–4% | all 12 colour groups. Circles 0, 4, 5 busiest (~13%), circles close 1 rarely (~2%) |
| Lit together | L/R pairs (4,5) (6,7) (14,15) (0,1) (2,3) (8,9); inner square 4–7 as a block | L/R pairs (2,3) (4,5) (10,11) (8,9) (6,7) (12,13) | (2,3), (0,2), single source circles 4/5 with 2/3 |
| Rotation axis `a` (moving boxes) | X dominates groups 0–13 (majority X in 7–8/8 maps, mostly 85–100%). Distant L/R 14/15 mixed (X 5/8, Y 3/8) | rings 0/1: X 100% in every map. Spotlights 4/5: X in 7/9, 50/50 in 2. Lasers 6–13 X ≥ 50% in every map | circle rotations 13–15: Y (89–100%) in 3/3. Single source lasers 6–11 X in 1 map, 50/50 in the other |
| Rotation spread `s` (mean abs per map, median) | inner 4–7: 5.5 (0–31); outer + side lasers: 7.9 (0–18) | rings: 20 (0–71); lasers 6–13: 16 (4–54); spotlights: 4.8 | always 0 on circle rotations; lasers 6–11: 0 and 7.8 |
| Colour wave `w` (d = 1) | share of boxes with a wave: inner 56%, outer 54%, side 42%, distant 19%. Median w 0.44 (inner), 0.45 (outer), 0.37 (side), 0.35 (distant) | rings: 58% of boxes, median w 0.75. Runway: 11%, w 0.5. Lasers and spotlights: ≤ 15% | rare (0% median, one map uses w 0.4–1 on circles) |
| Basic events | effectively none (1 type-4 event in 1 map) | types 0, 1, 4, 9 in 7–8/9 maps; 2/3 in 3/9; 12/13 none | 0, 1, 4 in 1/3 maps |

Changes made from this:

- **Weave**: the laser sweeps (outer and side squares) now rotate on X instead of Y. Channel
  waves are now 0.5 for rings (inner square) and 0.25 for the lasers
  (was 0.125 / 0). The back channel (distant) stays at 0.25, because only 19% of distant
  boxes use a wave. The laser wave stays below the 0.25-beat minimum onset spacing.
  The spin (inner square, X) and spreads already matched the data.
- **The Second**: the ring spin now turns on X instead of Y. The ring wave went from 0.25
  to 0.75. The sweeps now include the spotlights (4 left, 5 right), which are rotated in
  every map. They also get a 15° spread instead of 0. Basic types already matched.
- **EDM**: unchanged. Three maps support the Y axis for circle-rotation spins and zero spin
  spread, but they are too few to move the channels or waves.

Not changed despite a hint in the data:

- Mappers often light top and bottom pairs together. The laser channels keep the left/right
  split anyway, because L0 alternates left and right lasers.
- In the EDM maps, the single source circles 4/5 tend to light with circles 2/3 rather
  than with the lasers. Three maps are too few to move them.

## Dev script

```bash
python -m bsmapper.lighting.demo song.mp3 out/Song --style v3 --environment TheSecondEnvironment
```

The script analyses the song and pads it by the grid's lead-in, so the map sits on an
integer beat grid. It lights the map and adds one Expert Standard difficulty for
orientation, with alternating red and blue notes every 2 beats. It then packages
`song.egg`, a cover, Info.dat and the difficulty file. To watch the result, copy the folder
into `Beat Saber_Data/CustomLevels`. The script needs ffmpeg and the `audio` extra.
`tests/lighting/test_lighting_mapfolder.py` builds the same map folder from a synthetic
analysis and reads it back. Its slow test runs the script on `$BSMAPPER_DEMO_SONG`.

## Readability limits (WS8b)

`lighting.readability.limit_lightshow` bounds how fast a vanilla lightshow may change,
scaled by section intensity:

- per-channel minimum hold;
- colour hold;
- light-change moments per second;
- spin, zoom and laser speed gaps;
- boost toggle gaps.

It keeps the most important moments first: downbeats, then beats, then half beats.
`ReadabilityMask` is the decode-time counterpart for `LightSampler`. The model lighting
(L1) runs both by default; the L0 rules don't need them.

`lighting.metrics` measures a lightshow's readability: changes, colour flips, holds,
strobes, pattern repetition and v3 rotation speed. [LIGHTING_V2.md](LIGHTING_V2.md) has the
diagnosis, the human reference numbers and the design options for the next lighting model.

## Open points

- **In-game check.** Done means the lightshow is visibly synced in-game for all three
  styles. That still needs a playtest, along with checking the profile axes and groups.
- **Contract refinement (proposal).** `EventBoxGroup.boxes` could become typed
  dataclasses (`ColorBox`, `RotationBox`, `IndexFilter`, …). The v3.3 dict shape used
  here is the one `formats` already reads and writes.
- **L1.** The WS6 lighting decoder will output vanilla basic events. Feed those, with the
  section plans, to the same renderers.
