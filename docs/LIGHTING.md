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
  them back to the rest pose. The direction alternates between spins.
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

The channel-to-group assignments and rotation axes are a first guess from the group names.
They still need checking in-game.

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

## Open points

- **In-game check.** Done means the lightshow is visibly synced in-game for all three
  styles. That still needs a playtest, along with checking the profile axes and groups.
- **Contract refinement (proposal).** `EventBoxGroup.boxes` could become typed
  dataclasses (`ColorBox`, `RotationBox`, `IndexFilter`, …). The v3.3 dict shape used
  here is the one `formats` already reads and writes.
- **L1.** The WS6 lighting decoder will output vanilla basic events. Feed those, with the
  section plans, to the same renderers.
