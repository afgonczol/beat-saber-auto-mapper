"""WS4: DifficultyMap <-> tokens, vocabularies, windowing and grid snapping.

    dmap, report = snap_map(dmap)                 # onto the 1/24-beat grid (WS3b reuses this)
    tok = MapTokenizer()
    windows = tok.encode(dmap, Conditioning.from_map(dmap, nps=..., style=...))
    ids = windows[0].input_ids()                  # BOS prefix context SEP body EOS
    assert tok.decode(windows, meta=dmap) == canonical_order(dmap)

See docs/tokenizer.md for the token grammar and what does not round-trip.
"""

from bsmapper.tokenizer.conditioning import (
    PREFIX_LEN,
    Conditioning,
    notes_per_second,
    nps_bucket,
    style_bucket,
)
from bsmapper.tokenizer.grammar import (
    Grammar,
    ObjectSpec,
    ParsedEvent,
    Parser,
    TokenGrammarError,
)
from bsmapper.tokenizer.grid import (
    TICKS_PER_BEAT,
    WINDOW_BEATS,
    WINDOW_TICKS,
    OffGridError,
    SnapReport,
    beat_to_tick,
    is_on_grid,
    snap_beat,
    snap_map,
    tick_error,
    tick_to_beat,
)
from bsmapper.tokenizer.lighting import (
    LIGHT_GRAMMAR,
    LightingTokenizer,
    light_events,
    prepare_lightshow,
)
from bsmapper.tokenizer.notes import (
    NOTE_GRAMMAR,
    MapTokenizer,
    canonical_order,
    events_to_map,
    map_events,
)
from bsmapper.tokenizer.vocab import BOS, EOS, LIGHT_VOCAB, NOTE_VOCAB, PAD, SEP, Vocab
from bsmapper.tokenizer.windows import TokenWindow, build_windows, tail_context

__all__ = [
    "BOS",
    "EOS",
    "LIGHT_GRAMMAR",
    "LIGHT_VOCAB",
    "NOTE_GRAMMAR",
    "NOTE_VOCAB",
    "PAD",
    "PREFIX_LEN",
    "SEP",
    "TICKS_PER_BEAT",
    "WINDOW_BEATS",
    "WINDOW_TICKS",
    "Conditioning",
    "Grammar",
    "LightingTokenizer",
    "MapTokenizer",
    "ObjectSpec",
    "OffGridError",
    "ParsedEvent",
    "Parser",
    "SnapReport",
    "TokenGrammarError",
    "TokenWindow",
    "Vocab",
    "beat_to_tick",
    "build_windows",
    "canonical_order",
    "events_to_map",
    "is_on_grid",
    "light_events",
    "map_events",
    "notes_per_second",
    "nps_bucket",
    "prepare_lightshow",
    "snap_beat",
    "snap_map",
    "style_bucket",
    "tail_context",
    "tick_error",
    "tick_to_beat",
]
