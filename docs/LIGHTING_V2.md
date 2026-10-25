# Lighting v2: diagnosis, design and the style + scene model (WS8b)

Status:

- **Phase 1** (#36, merged): diagnosis, style analysis, design options and the readability
  quick win (§1–§4).
- **Phase 2**: the user chose (a) + (b2) after playtesting phase 1 ("lighting is way
  better"). §5 describes the light prompt v2 (style token + section scene plan + light
  references) and the lights-only fine-tune `light_v2`: trained 2026-10-24, adherence
  results and playtest maps in §5, awaiting the user's playtest.

The user played note_v1 maps and reported: the notes are "phenomenal", but the lights
"leave plenty to be desired". In Stronger, a section toward the middle is "very spastic
and constantly changing/jittering too much". The user also had an insight: each human map
has its own lighting style, so one model trained across all maps averages the styles into
mush.

This document covers four things:

1. The diagnosis of the Stronger jitter, measured.
2. The style analysis of 3,551 human lightshows.
3. The design options, with cost and risk.
4. The quick win in this PR, and its before/after numbers.

## How lighting works today

1. **L1**, note_v1's lighting decoder (26M parameters), writes vanilla basic events for
   one 16-beat window at a time.
   - It cross-attends to that window's audio and to the note track of the hardest
     difficulty.
   - Its only memory of the previous window is the last 16 events (`tail_context`).
   - It has no conditioning prefix, so nothing tells it which style to use.
   - Sampling uses temperature 0.9 and top-p 0.95.
2. The L0 planner (`lighting.rules`) supplies the section plans: palette, boost and
   intensity.
3. WS8's renderers turn the basic events into v3 event box groups (WeaveEnvironment by
   default), Chroma or vanilla.
   - On v3, ring spins and zooms become rotation boxes on the spin groups, and laser speeds
     become back-and-forth sweeps.
   - The #28 rest pose and the 30° cap still apply, so no light is parked across the track.

## 1. Diagnosis: the Stronger jitter

### Method

- Stronger (Kanye West, 104 BPM, 3:15) was regenerated with the current defaults: note_v1,
  intensity 0.8, seed 0, v3 on Weave.
- Every lightshow is measured with `bsmapper.lighting.metrics`, per L0 section and per
  16-beat window.
- The reference is the same statistics on the human lightshows: every *gold* map in
  `data/processed_v1` with a basic-event lightshow and a single tempo (3,566 maps, 105k
  windows). Each human window is binned by its song-relative loudness (window RMS ÷ the
  song's loudest window), and quantiles weight every map equally.

Metrics (seconds, so songs of different tempo compare):

| Metric | Meaning |
|---|---|
| changes/s | Visible light changes per second, summed over the 5 light channels. Repeating the current state is invisible and not counted. Several events on one channel at one tick count once, as in the game. |
| colour flips/s | A channel lighting up in a different colour from its previous lit event. Fades count: a red fade one beat after a blue fade reads as a flip. |
| moments/s | Distinct instants at which any light changes. |
| hold p10 | 10th percentile of the time a channel holds before its next change. |
| strobe/s | Changes less than 0.1 s after the channel's previous change. |
| whips/s, max °/s | On the rendered v3 map: rotation moves of a light group faster than 720°/s (two turns a second), per moving group per second, and the fastest move. |
| pattern repeat | How much each bar repeats one of the two bars before it, as (position, channel, colour) Jaccard overlap. |

Human reference, in the loudest windows (≥ 0.85 of the song's peak), as p10 / median /
p90:

| | p10 | median | p90 |
|---|---|---|---|
| changes/s | 3.7 | 9.6 | 25.2 |
| colour flips/s | 0 | 2.6 | 6.1 |
| moments/s | 2.4 | 4.8 | 15.4 |
| hold p10 (s) | 0.02 | 0.17 | 0.45 |
| laser speed events/s | 0 | 2.3 | 6.1 |
| ring spins/s | 0 | 0.7 | 3.2 |
| pattern repeat (whole map) | 0.12 | 0.33 | 0.63 |

### Findings

Stronger, default lightshow (seed 0), by L0 section:

| Section | Changes/s | Colour flips/s | Laser speed/s | Max rotation °/s | Whips/s |
|---|---|---|---|---|---|
| 0:02–0:25 intro | 0.17 | 0.09 | 0 | 0 | 0 |
| 0:25–1:12 chorus | 3.4 | 0.8 | 1.4 | 624 | 0 |
| 1:12–1:28 verse | 6.3 | 2.2 | 2.5 | 312 | 0 |
| 1:28–1:37 verse | 9.1 | 2.6 | **3.8** | 416 | 0 |
| **1:37–1:58 chorus** | 5.9 | **3.7** | 2.2 | **1248** | **0.06** |
| **1:58–2:16 verse** | 6.4 | **3.1** | 2.6 | **4991** | **0.04** |
| 2:16–2:35 verse | 7.0 | 2.0 | 2.9 | 416 | 0 |
| 2:35–3:15 outro | 6.5 | 2.3 | 2.7 | 1248 | 0.02 |

**The jittery section is 1:28–2:16** (beats 152–236), the middle of the song. It starts
with the densest laser-speed stretch and peaks in the second chorus (1:37–1:58) and the
verse after it (1:58–2:16, which is quieter, energy 0.62). Four things stack up there.

1. **The renderer turns spin rolls into whips.** From beat 175 the decoder writes ring
   spins in pairs a beat apart (175/176, 179/180, 183/184 …), each pair with a zoom. That
   is a common vanilla pattern, and it looks fine on rings that animate themselves. The v3
   renderer, though, gives every spin a full 360° turn and squeezed it into half the gap
   to the next spin. A turn in half a beat is 1,248°/s at 104 BPM. One spin at 2:00 was
   squeezed to a 12th of a beat, 4,991°/s. The inner square whips round every bar.
   **Cause: the renderer.**
2. **Laser sweeps restart every beat or two.** The decoder changes laser speed 2–4 times a
   second, as humans do (median 2.3/s). In vanilla, a speed event only changes the laser
   speed. The v3 renderer instead restarted the sweep at every speed event, always
   swinging first to the same side. With speeds alternating 3/5/7, left and right never
   in step, the lasers jerk back and forth. **Cause: the renderer**, triggered by a
   human-like event rate.
3. **The colours flip without a plan.** The light events themselves are tame:
   - 99.6% are fades;
   - the hold p10 is 0.29 s;
   - there are no strobes;
   - changes/s is *below* the human median.

   But each channel picks red or blue almost independently, beat by beat. Example: all
   red on beat 154, all blue on 155, then rings red and centre blue … Colour flips reach
   3.7/s in the chorus and 3.1/s in the verse after it. That is above the human median
   (2.6, and 1.6 at the verse's energy) and far above the 0.5/s per channel that humans
   average. Bar-to-bar pattern repetition is 0.25 for the whole song, against a human
   median of 0.33. Nothing ties a section to a palette or a motif. **Cause: the decoder.**
4. **Each window only remembers its last 16 events.** At this density that covers 1–2
   beats. Whatever palette or motif a section started with is forgotten within a bar, and
   each 16-beat window restarts from that short tail. **Cause: per-window resets.**

Temperature and the seed are **not** the cause:

- Seed 1 and temperatures 0.7, 0.5 and 0.3 give the same rates on Stronger: 6.3–6.7
  changes/s, 2.1 colour flips/s and 2.4–2.7 laser speed events/s.
- Lower temperatures only make the colour alternation rigid: exactly 2.17 flips/s in
  every section at 0.5 and 0.3.
- The whips depend on where spin pairs land, which does change with the seed (seed 1:
  max 461°/s).

Two more findings matter for the design.

- **The decoder's style is decided early, and by chance.**
  - Its look for a whole song (fade pulses vs on/off with transitions; lasers vs none)
    is set in the first windows and then carried through the tail context.
  - A single changed token can flip it. Masking one event on an unused light channel at
    beat 4 turned Stronger from fade pulses into on/transition pulses for the whole song.
  - GPU numeric noise between two runs flipped Toxic at beat 179, after which its lasers
    disappeared.
  - So the model holds several human styles and picks one per song by chance. It is not
    pure "mush", but it cannot be steered, and a section can drift.
- **Songs can start dark and stay dark.**
  - Stronger's first 25 s get 0.17 changes/s at 0.70 of peak loudness.
  - Toxic (143 BPM) is worse. For its first 89 s, including two full-loudness choruses,
    it gets 0–1.3 light changes per second. Its only "show" is ring spins every 0.4 s,
    each a 1,716°/s whip, with nothing lit.
  - An empty window gives the next window an empty tail, so darkness feeds itself.
  - The existing density check only falls back to L0 for the whole song.

## 2. Style analysis of human lightshows

### Features

Per map, computed in its loud windows (≥ 0.7 of peak):

- the rate of visible changes;
- the channel mix: laser share and ring share;
- colour flips per change;
- the shares of fade, flash and on events;
- the strobe share and the median hold;
- ring spins, zooms and laser speed events per second;
- timing: the share of changes on the beat and off the 1/4 grid, and the onset lift
  (onset strength at changes ÷ onset strength in loud frames);
- colour boost time and the share of white;
- energy following: Spearman correlation between window loudness and change rate;
- consistency: the coefficient of variation of the change rate across loud windows.

### Clusters

The 19 standardized features were clustered with k-means.

- The silhouette is low at every k (0.20 at k = 3–5, 0.12 at k = 6). Lighting styles are
  a continuum, not islands.
- k = 6 gives the most interpretable split, so that is the one used here:

| # | Style | Share | What defines it (medians) | Example maps |
|---|---|---|---|---|
| 3 | **Beat-pulsed fades + laser sweeps** | 36% | 100% fades, 10 changes/s, laser speed 3.8/s, 57% on the beat, follows energy (ρ 0.61) | Have A Great Life (Nolanimations), Angel Voices (Joetastic), Popular Monster (muffn) |
| 0 | **Calm fade pulses** | 19% | 100% fades, 5 changes/s, long holds (0.68 s), no laser speed, weak energy following (ρ 0.30) | Pet Cheetah (Kansas), Payphone (nolan121405), All Eyes On Me (Timbo) |
| 4 | **Sustained on/off scenes** | 18% | 49% plain on, 4% fades, few colour flips (0.15 per change), less consistent | Ding Dong Song (DaftMaple), Woman (Lekrkoekj), White Reflection (Joetastic) |
| 5 | **Strobe / sequenced** | 15% | 22 changes/s, median hold 0.05 s, 55% off the 1/4 grid, mostly Chroma maps | Monster (Dirty Alex), Crank (Dirty Alex & Rexxz), Otaku Boy (Ramen Noodle) |
| 1 | **Flash + zoom** | 7% | 50% flashes, 0.8 zooms/s, 15.5 changes/s, many colour flips | Cried Crime (Suiba x Cabbage), Feel So Gud (General Dumbass & Rexxz), In The Name Of Love (Duddzy) |
| 2 | **White + boost accents** | 5% | 27% white, boost on 39% of the time, ring-heavy, few laser sweeps | Toca Toca (Bellus), Genesis (symphonic), Will (Ge2toro) |

(Share = share of the 3,551 gold maps. The full example lists, eight per cluster, are in
the analysis output.)

Notes:

- **"Strobe / sequenced" (15%) is mostly an artefact for us.**
  - These maps light single lights in sequence with Chroma lightIDs and gradients. The
    tokenizer drops `customData`, so in our data they become plain channel events 1–3
    ticks apart, which is strobe noise in vanilla and v3.
  - This is the strongest data-hygiene finding: these maps should be excluded from light
    training or given their own style token.
- The model's own output falls into these clusters too, and which one depends on the seed.
  Each lightshow was assigned to its nearest centroid:
  - Toxic: seeds 0–3 land in styles 4, 4, 0, 3;
  - Stronger: 0, 0, 4, 0.

  So note_v1 picks a style per song by chance.
- Per-section consistency is high in human maps (coefficient of variation 0.23–0.38 of the
  change rate over loud windows). The style holds for the whole song.

## 3. Design options

Costs are for the RTX 3090.

- A full note_v1 run (notes + lights) is 80k steps at about 2.1 steps/s, roughly 11 h.
- A *lights-only* fine-tune freezes the audio encoder and the note decoder and trains the
  26M light decoder from note_v1's weights. Estimate: 3–4× faster per step, 30k steps in
  about 2–3 h.

### (a) Style-conditioned lighting model

- Add a `LIGHT_STYLE_k` token to each light window's prompt (lights have no prefix today),
  plus `LIGHT_STYLE_NONE` with 20% dropout so the unconditioned model still exists.
- Labels come from the clusters above. Recluster after dropping the strobe/sequenced
  maps, or keep them as an opt-in style.
- The user picks a style, or it is auto-picked from the song. The auto-pick would be a
  small classifier from song features (energy profile, onset density, tempo) to style,
  trained on the human maps in minutes.
- **Expected quality:** the whole song keeps one look, chosen on purpose instead of by
  chance. Re-rolls give variety within a style. It does *not* by itself stop colours
  flipping within a section, or tie repeated sections together.
- **Cost:**
  - data: labels exist after phase 1, and the tokenizer needs a lighting prefix (a few
    hours of code);
  - training: lights-only fine-tune, 2–3 h;
  - plus an auto-pick model, under 1 h.
- **Risk: medium.**
  - Styles overlap (low silhouette), so the token may steer only partly. Classifier-free
    guidance (as for STYLE on notes) can strengthen it.
  - Small clusters (5–7%) have about 180–260 maps each.

### (b) Hierarchical: a scene plan per section, then beat-level fill

- A *scene* per L0 section: palette (lead colour per channel), energy level, active
  channels, and a 1–2 bar motif.
- Repeated sections reuse their earlier scene, using note_v1's structure memory
  (`data.structure.find_references` already finds the repeated spans for the notes).
- Two ways to build it:
  - **(b1) rule plan + enforcement, no retrain.**
    - Extend the L0 planner to emit a scene per section.
    - Post-process the decoded events: recolour each channel to the scene palette, mute
      inactive channels.
    - Copy the light events of a section's reference span when the music repeats, as
      note_v1 does with notes.
    - Cost: 1–2 days of code, no GPU.
    - Risk: low to medium. It can look mechanical, and recolouring fights the decoder's
      own colour choices.
  - **(b2) learned fill with scene and reference conditioning.**
    - Give each light window a `REF … REF_END` block holding the light events of its
      reference span, exactly like note_v1 (the per-window `ref_start` is already in the
      v1 manifest).
    - Optionally add scene tokens extracted from human maps (dominant colours, active
      channels).
    - Cost:
      - data: about a day to extract the scenes and build light refs;
      - training: lights-only fine-tune with refs, 3–5 h (refs make the prompts longer).
    - Risk: medium. note_v1's refs cut note CE from 0.644 to 0.558, so the mechanism
      works. Lights are denser and less exactly repeated than notes, so the gain is
      uncertain.
- **Expected quality:** the biggest win against "constantly changing". Each section gets
  a stable palette and motif, and choruses look alike. This addresses findings 3 and 4.
  A scene plan also guarantees a lit scene in every section, which fixes the dark intro.

### (c) Readability constraints

Rules that bound how fast the show may change, tied to section energy:

- per-channel minimum hold;
- colour hold;
- a cap on light-change moments per second;
- motion gaps for spins, zooms and laser speeds;
- boost toggle gaps.

They can run as a post-filter or as decode masks.

- **Expected quality:**
  - Removes the worst failure modes: whips, sweep restarts, colour flipping above
    human-like rates, strobe bursts.
  - Cannot create structure. A filtered random show is calmer, but still not designed.
- **Cost:** none in data or GPU. **Done in this PR** as a post-filter (plus a per-window
  dark fill), renderer fixes and an opt-in decode mask.
- **Risk:**
  - Low as a post-filter.
  - **High as decode masks on note_v1.** Masking even rare tokens changes the sample, and
    blocking the model's quick "off" events or its laser speed events pushed it into
    another style for the rest of the song (§4). Masks only become safe once the model is
    trained with them (or with a planner that keeps it on-distribution).

### Combinations

| Option | Quality gain | Data work | GPU (3090) | Risk |
|---|---|---|---|---|
| (c) now | Removes whips, restarts, flip bursts | none | 0 | low |
| (c) + (b1) | + stable palettes and motifs, repeats, no dark sections | 1–2 days of code | 0 | low–medium |
| (c) + (a) | + one chosen style per song, user control | prefix + labels, ~½ day | 2–3 h | medium |
| (c) + (a) + (b2) | + learned motifs and repeats in the chosen style | ~1.5 days | 4–6 h (one fine-tune) | medium |
| Full retrain, all of the above | same as above, plus note re-tuning | ~1.5 days | ~11 h, and it blocks the GPU overnight | medium–high (notes could regress) |

**Recommendation: (c) now, then (a) + (b2) as one lights-only fine-tune**, with these data
fixes:

- drop or isolate the strobe/sequenced maps;
- keep only lightshows the vanilla vocabulary represents well.

(b1) is a cheap stopgap if the user wants calmer, structured lights before any training.
It also gives (b2) its scene plans.

## 4. Quick win: readability limits (this PR)

### What changed

- **Renderer** (`lighting/render.py`):
  - A ring spin always takes its full 2 beats. Spins and zooms that come while a turn is
    running are ignored, instead of squeezing the turn. That caps a turn at 312°/s at
    104 BPM and 429°/s at 143 BPM.
  - Laser sweeps no longer restart on every speed event:
    - a repeated speed continues the running sweep;
    - a new speed keeps the swing direction going, instead of jumping back to one side.
  - The #28 rest pose and the angle cap are unchanged.
- **Post-filter** (`lighting/readability.py`, `limit_lightshow`): it keeps the most
  important moments first (downbeats, then beats, half beats, the rest) and applies, from
  calm to peak section intensity:

  | Limit | Calm | Peak |
  |---|---|---|
  | minimum hold per light channel | 0.25 s | 0.12 s |
  | colour hold per channel (an earlier flip keeps the colour; the hit stays) | 4 beats | 2 beats |
  | light-change moments in any 1 s | 4 | 8 |
  | laser speed changes per side | 2 beats apart | 1 beat apart |
  | ring spins (a full turn of the inner lights on v3) | 4 beats apart | 4 beats apart |
  | ring zooms | 2 beats apart | 2 beats apart |
  | colour boost toggles (a blip cancels as a pair) | 4 beats apart | 4 beats apart |

  - Left and right laser speed changes less than half a beat apart move together.
  - **Dark-window fill.** A 16-beat window at section intensity ≥ 0.5 with fewer than 0.5
    visible changes per second takes the L0 rule lightshow's light events. Motion stays
    the model's. This is the existing whole-song L0 fallback, made per window.
  - The defaults are conservative. Every limit sits at or beyond the human median for its
    energy, so typical human-like stretches pass unchanged.
- **Decode-time limit** (`ReadabilityMask`, a `LightSampler` mask), **opt-in**
  (`ReadabilityConfig.decode_mask`). It forbids:
  - event types no renderer shows;
  - with `decode_light_holds`, light events inside the minimum hold;
  - with `decode_motion_gaps`, spins, zooms, laser speeds and boosts inside their gaps.

  It is off by default because in the sweep below no mask setting beat the post-filter
  alone, and every one changed the sample. Toxic often lost its lasers.
- **Metrics** (`lighting/metrics.py`): `span_stats`, `window_stats`, `pattern_repeat`,
  `together_share` and `motion_stats`, as used above.
- **Generation hook** (`generation/lighting.py`): `ModelLightingGenerator` runs the
  post-filter on the decoded events, with the L0 rule lightshow as the dark-window
  fallback, and the mask when `decode_mask` is set.
  - The section intensities come from the L0 planner.
  - `readability=None` turns both off.
  - Nothing else in `generation/` changed: a parallel session is working there (WS6b,
    arcs and modes).

### Before / after

#### Setup

- The same decoded lightshows are measured before and after.
- *Before* is main: no filter, the old renderer.
- *After* is this PR's default: the renderer fixes, plus the post-filter with the dark
  fill.
- Seed 0 is what is now installed in the game. The 4-seed mean (seeds 0–3) shows the
  change is not one lucky sample.
- Rotation figures come from the rendered v3 map. Whips are per moving light group per
  minute.

#### Stronger

The jittery section is 1:28–2:16.

| Metric | Before, seed 0 | After, seed 0 | Before, 4 seeds | After, 4 seeds |
|---|---|---|---|---|
| fastest rotation (°/s) | **4,991** | **416** | 4,991 | 416 |
| whips per group per min | 0.83 | **0** | 0.60 | **0** |
| colour flips/s, jittery section | 3.26 | **1.96** | 1.69 | 1.15 |
| colour flips/s, whole song | 1.87 | 1.56 | 1.21 | 1.06 |
| laser speed changes/s, jittery section | 2.64 | **1.77** | 1.30 | 0.90 |
| laser speed changes/s, whole song | 2.04 | 1.48 | 1.25 | 0.94 |
| ring spins/s | 0.41 | 0.36 | 0.37 | 0.33 |
| light changes/s | 5.09 | 5.11 | 7.63 | 6.46 |
| dark windows filled from L0 | – | 4 (the intro) | – | – |
| bar pattern repeat | 0.25 | 0.24 | 0.46 | 0.45 |

Filter actions, seed 0:

- dropped: 109 laser speed changes, 28 moments over the cap, 9 spins, 9 light events
  inside their hold, 1 zoom;
- recoloured: 65 colour flips;
- moved: 51 laser speed changes, to sync left and right.

#### Toxic

The jittery section is 1:29–1:41, the section with the most colour flips before.

| Metric | Before, seed 0 | After, seed 0 | Before, 4 seeds | After, 4 seeds |
|---|---|---|---|---|
| fastest rotation (°/s) | **1,716** | **429** | **41,184** | **429** |
| whips per group per min | 7.61 | **0** | 5.68 | **0** |
| colour flips/s, jittery section | 2.04 | 1.62 | 1.40 | 1.30 |
| laser speed changes/s | 0.81 | 0.62 | 1.74 | 1.26 |
| ring spins/s | 0.80 | 0.50 | 0.70 | 0.43 |
| light changes/s | 4.91 | 4.77 | 6.91 | 6.11 |
| dark windows filled from L0 | – | 8 | – | – |

- Seed 0: the two full-loudness choruses at 0:30–0:52 go from 0 light changes/s to
  3.8–4.3. Before, they had only spin whips.
- Filter actions, seed 0: dropped 207 light events inside their hold, 59 spins, 38 laser
  speed changes and 2 boost toggles (a one-beat blip); recoloured 30 flips.

Everything after the filter stays inside the human range for its loudness (§1). It sits
at or below the human median, except where the model was already above it.

#### Decode masks

These were tested in the same sweep, each with the post-filter after it. 4-seed means:

| Mask | Stronger laser speed/s | Toxic laser speed/s | Toxic light changes/s | Verdict |
|---|---|---|---|---|
| none (default) | 0.94 | 1.26 | 6.11 | — |
| unused types only | 0.87 | 1.28 | 5.84 | Same on average. Seed 0 of Toxic lost its lasers (0.03/s, 2.3 changes/s). |
| + motion gaps | 0.28 | 0.38 | 4.35 | Lasers mostly gone. |
| + light holds | 0.37 | 0.25 | 5.41 | Lasers mostly gone. Style shifted to on/off scenes. |

The masks don't calm what the post-filter leaves, and they can strip whole channels.
They stay opt-in until a model is trained with them.

### Playtest

Stronger and Toxic were regenerated with this branch (note_v1, intensity 0.8, `--arcs`,
seed 0) and installed with `--install`. To compare, regenerate one with
`--lighting-source rules` (L0) or check out main.

Please report:

1. Is the middle of Stronger (1:28–2:16) still jittery? If so, is it the colours, the
   lasers or the inner square?
2. Do Toxic's first choruses (0:30–0:52) now feel lit? They use L0 rule lights there,
   because the model left them dark.
3. Which direction to take next:
   - (b1) rule scene plans, no training;
   - (a) style token;
   - (a) + (b2) one fine-tune;
   - something else.

### Reproducing

The analysis scripts are in `scripts/lighting_v2/` (they need `scikit-learn` for the
clustering):

1. `human_stats.py` computes the per-window and per-map stats of the gold lightshows.
2. `human_bins.py` computes the energy-binned quantiles.
3. `styles.py` runs the clustering.
4. `gen_cache.py` and `diagnose.py` regenerate a song and compare it.

## 5. Phase 2: style token and section scene plan (light_v2)

Option (a) + (b2), keeping (c) as the safety net. The lighting decoder gets a prompt for
the first time:

```text
BOS  LSTYLE_s LEVEL_l PAL_p FLIPS_f  [REF <light events> REF_END]  <tail>  SEP  <body>  EOS
```

- `LSTYLE_s`: the map's style (below). `LSTYLE_NONE` = no preference.
- `LEVEL`, `PAL`, `FLIPS`: the window's *scene* (`tokenizer.lighting.window_scene`):
  - `LEVEL_0..6`: light events per beat. Edges 0.5 / 2 / 4 / 7 / 12 / 24.
  - `PAL`: the colour mix of lit events. `DARK`, `BLUE`, `BLUE_LEAD`, `EVEN`,
    `RED_LEAD`, `RED` (red share edges 0.15 / 0.42 / 0.58 / 0.85), `WHITE` (≥ 25% white).
  - `FLIPS_0..3`: the share of lit events that change their light's colour. Edges
    0.05 / 0.15 / 0.3. This is the "constantly changing" axis.

  Each has a `_NONE`. The edges come from the labelled gold windows of dataset v1: the
  first cut (flips per beat, red share 0.1 / 0.4 / 0.6 / 0.9) put nearly every loud
  window into one bucket.
- `REF … REF_END`: the human (training) or generated (generation) light events of the
  window's reference span: the same structure references note_v1 uses (`ref_threshold`
  0.85), at most 384 tokens.

The light vocabulary grows from 500 to 532 tokens. The new groups are appended, so v1
checkpoints (note_v1) keep working and decode as before (`LIGHT_VOCAB_SIZES`,
`light_version_for_vocab`).

### Styles (`bsmapper lights styles`)

`lighting.style_build` re-runs the phase 1 analysis inside the package and writes
`data/processed_v1/light_styles.json`: style labels, the style model and the scene
tables.

- **Candidates:** 3,566 gold maps with a basic-event lightshow on one tempo.
- **Left out:**
  - 15 lightshows too sparse to describe;
  - 1,023 **sequenced** ones (`styles.is_sequenced`).

  A Chroma lightID sequence, once its lightIDs are dropped, shows as several events of
  one light on the same tick. The share of such *stacked* events is bimodal on dataset
  v1 (median 0.1%, p90 66%), so a map counts as sequenced when at least 10% of its light
  events are stacked, or when 60% of its changes are strobes (under 0.1 s). Plain
  Chroma colours without lightIDs are kept. Phase 1's "any Chroma colour" rule would
  have dropped 23% of the maps, most of them fine.
- **k-means with k = 5** on the 19 standardised style features of §2. The clusters are
  named by matching them to prototype signatures (`styles.STYLE_PROTOTYPES`). They are
  phase 1's clusters 0–4, minus the sequenced one:

| Style | Maps | Look | Distinctive (z) |
|---|---|---|---|
| `pulse` | 1,163 (46%) | beat-pulsed fades with laser sweeps | laser speed +0.66, fade +0.54 |
| `calm` | 549 (22%) | slow fade pulses, long holds, little laser motion | rate −1.10, hold +0.96 |
| `flash` | 362 (14%) | fast flashes with ring zooms and spins | strobe +1.79, flash +1.15 |
| `scenes` | 313 (12%) | sustained on/off scenes, few colour flips | on +1.98, fade −1.63 |
| `accent` | 147 (6%) | white accents and colour boosts | white +2.87, boost +1.07 |

- **Auto-pick:** a softmax regression from 18 song features (tempo, length, loudness
  profile, onsets, mel profile) to style. It reaches 48% on the 250 val maps, against
  44% for always guessing `pulse`. The audio barely determines the style, as the low
  silhouette in §2 suggested. So `auto` *draws* a style from the predicted
  probabilities with the generation seed, instead of always taking the top one: another
  seed can give another look, and a fixed `--light-style` gives a fixed one.
- **Scene tables:** how often each scene occurs in the human windows, per style and
  per loudness bin (window RMS against the song's loudest window: edges 0.2 / 0.4 /
  0.6 / 0.8).

### Generation

`ModelLightingGenerator` with a v2 checkpoint:

1. **Style:** `light_style` is `auto`, `none` or a style name.
2. **Scene plan** (`lighting.scenes.plan_scenes`): one scene per L0 section, held by
   every window of the section.
   - A section whose music repeats an earlier one (most of its windows' references
     start there) reuses that section's scene.
   - Any other section draws from the style's human scenes at the section's loudness.
   - Loud sections (bin 2 and up) never draw a dark or near-dark scene. This fixes the
     dark choruses of §1.
   - Single-colour and lead palettes follow the L0 section's lead colour, so palettes
     alternate between sections.
   - Silent windows get a dark scene.
3. **References:** each window's reference block holds the *generated* lights of its
   reference span (`WindowRefs` on the light grammar).
4. The readability post-filter (§4) still runs afterwards, unchanged.

Options:

- the CLI: `bsmapper generate --light-style auto|none|pulse|calm|scenes|flash|accent`;
- the config: `[generate] light_style`;
- the jobs API and the web UI: `light_style`, the "Light style" menu under Lighting.

The default is `auto`. A v1 checkpoint ignores the option.

### Fine-tune (`configs/light_v2.yaml`)

- **Init:** note_v1's `best.pt`. The light embedding grows to 532 rows; the new rows
  start at the mean embedding (`models.checkpoint.load_weights`).
- **Frozen:** the audio encoder and the note decoder (`run.freeze`). The checkpoint
  therefore generates exactly note_v1's notes. The note memory and the lighting decoder
  train.
- **Data:**
  - labelled gold maps only (`light_labelled_only`);
  - dropout 0.2 on the style and 0.2 on the scene (all three tokens), 0.3 on the
    reference;
  - up to 1,536 tokens per example.
- **Optimiser:** lr 1e-4 with 1k warmup, cosine to 1e-5, batch 32, z-loss.
- **Run:** the divergence guard, and the detached run with a watchdog of note_v1
  (`checkpoints/light_v2/code/control.py start|stop|status`). `best.pt` is chosen on
  val light loss.

`light_v2` is **not** the default checkpoint until the user has playtested it. Use
`--model checkpoints/light_v2/best.pt` or `[model] checkpoint` in the config.

### Training run (2026-10-24)

- 30k steps in 2 h 54 min, with no guard events.
- Val light loss:
  - It reached its low within the first 1k steps (1.425), stayed flat to step 10k, then
    crept up to 1.438 by step 30k while light accuracy kept rising (80.2% → 80.5%).
  - `best.pt` is step 6000 (1.424).
  - This loss is not comparable with note_v1's 1.45: the v2 val windows include prompt
    tokens.
- Most of the gain comes from the prompt and the light references. Training past ~10k
  steps only sharpens the distribution.

### Adherence (`scripts/lighting_v2/phase2_eval.py`, 8 val maps, seed 0)

Each map's lights are decoded on its human audio and Expert+ notes, once per condition.

- **Measures:**
  - The generated show is labelled with the style model.
  - Changes/s and flips/s are medians over loud windows.
  - "Dark" counts loud windows with fewer than 0.5 changes/s.
  - The scene match is the share of windows within ±1 level bucket / with the same palette
    / within ±1 flip bucket of the map's human scenes.
- **No filter:** the readability filter of §4 is *not* applied here.

| Condition | Changes/s | Flips/s | Dark | Style hits | Scene match (level / palette / flips) |
|---|---|---|---|---|---|
| human | 9.8 | 3.0 | 0 | | |
| note_v1 | 10.6 | 1.3 | 3 | | 0.67 / 0.29 / 0.49 |
| light_v2, no prompt | 7.3 | 2.5 | 1 | | 0.66 / 0.38 / 0.75 |
| light_v2, `pulse` | 8.5 | 3.1 | 0 | 5/8 (none: 3/8) | 0.78 / 0.39 / 0.74 |
| light_v2, `flash` | 11.0 | 3.2 | 0 | 1/8 (none: 0/8) | 0.70 / 0.39 / 0.66 |
| light_v2, `scenes` | 6.9 | 2.4 | 5 | 4/8 (none: 1/8) | 0.56 / 0.43 / 0.66 |
| light_v2, `accent` | 9.0 | 3.1 | 0 | 1/8 (none: 1/8) | 0.73 / 0.53 / 0.79 |
| light_v2, `calm` | 6.2 | 2.4 | 16 | 4/8 (none: 3/8) | 0.78 / 0.42 / 0.72 |
| light_v2, human style + scenes | 9.9 | 3.2 | 0 | 4/8 (own label) | 0.77 / 0.44 / 0.78 |

- **The style token works, but weakly.** It moves the show in the intended direction:
  `flash` is the busiest, `calm` and `scenes` the sparsest. `pulse` and `scenes` clearly
  raise their own label.
  - `flash` and `accent`, the two rarest styles (362 and 147 maps), barely register.
  - Audio still decides most of the look: per map, the generated style often stays the
    same across all five requests.
- **Scenes help.** With the human scenes as the plan, every axis matches better than
  note_v1, the most on palette (0.29 → 0.44) and flips (0.49 → 0.78). Density and flips
  then come out at the human level (9.9 / 3.2 vs 9.8 / 3.0).
- **Style alone can go dark.** `calm` or `scenes` without a scene produce dark loud
  windows. In generation the scene plan forbids dark scenes in loud sections, and the
  readability filter fills dark windows, so the playtest maps had at most one filled
  window.
- **Colour flips:** note_v1 flips colours less than humans (1.3 vs 3.0 per second on loud
  windows). light_v2 follows the requested scene's flip bucket instead. A sustained,
  low-flip look therefore needs a low-flip scene or the `scenes` style; the §4 caps still
  apply on top.
- **Caveat:** these are 8 maps and one seed, so treat the hit rates as indicative only.
- **Possible follow-up:** classifier-free guidance on the style
  (`ModelLightingGenerator(style_guidance=…)`, already implemented, default 1.0) could
  strengthen the rare styles.

### Playtest maps

`out/light_v2_playtest/` has Stronger and Toxic, each with `--light-style` auto, calm,
scenes and flash (seed 0). The subtitle in-game shows `lights <style>`.

- Auto picked `accent` for Stronger and `flash` for Toxic.
- The notes are identical across variants; they come from the frozen note_v1 decoder.
