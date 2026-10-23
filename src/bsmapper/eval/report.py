"""A self-contained HTML report of one evaluation summary (no external assets)."""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any

from bsmapper.eval import repetition as rep
from bsmapper.eval.runner import ALL, HEADLINE, LIGHT_HEADLINE
from bsmapper.eval.systems import HUMAN

GROUP_COLUMNS = (
    ("n", "difficulties", ""),
    ("nps", "NPS", "human"),
    ("nps_ratio", "NPS / human", "human"),
    ("nps_percentile_median", "NPS pct", "human"),
    ("step_offset", "step offset", "human"),
    ("onset_f1_2", "F1 ±2", "high"),
    ("errors_per_100", "errors/100", "low"),
    ("jsd_pattern_3gram", "JSD 3-gram", "low"),
    ("rep_reuse", "reuse", "human"),
)

CSS = """
:root { --bg:#fbfbfa; --fg:#1d1d1b; --muted:#6b6b66; --line:#e2e1dc; --head:#f1f0ec;
  --best:#1f7a4d; --bad:#b3261e; --accent:#3557a8; }
@media (prefers-color-scheme: dark) { :root { --bg:#161615; --fg:#e8e7e3; --muted:#9a9993;
  --line:#33322f; --head:#21201e; --best:#5cc28f; --bad:#f08b83; --accent:#8fa8e8; } }
body { background:var(--bg); color:var(--fg); margin:0 auto; max-width:1100px; padding:24px 16px;
  font:14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:17px; margin:28px 0 8px; }
p, li { color:var(--fg); } .muted { color:var(--muted); }
.wrap { overflow-x:auto; }
table { border-collapse:collapse; margin:6px 0 12px; font-variant-numeric:tabular-nums; }
th, td { border-bottom:1px solid var(--line); padding:4px 10px; text-align:right;
  white-space:nowrap; }
th { background:var(--head); font-weight:600; } td:first-child, th:first-child { text-align:left; }
td.best { color:var(--best); font-weight:600; } td.ref { color:var(--muted); }
code { font-size:12px; } dl { display:grid; grid-template-columns:max-content 1fr; gap:4px 14px; }
dt { font-weight:600; } dd { margin:0; color:var(--muted); }
"""


def _fmt(v: Any, key: str = "") -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "–"
    if isinstance(v, int) or key == "n" or key.endswith("pairs"):
        return f"{int(v)}"
    if "percentile" in key:
        return f"{v:.0f}"
    if abs(v) >= 100:
        return f"{v:.0f}"
    return f"{v:.3f}" if abs(v) < 10 else f"{v:.2f}"


def _best(values: dict[str, Any], better: str, human: Any) -> set[str]:
    """Generated systems with the best value of a row."""
    ok = {
        s: v
        for s, v in values.items()
        if s != HUMAN and isinstance(v, int | float) and math.isfinite(v)
    }
    if len(ok) < 2 or better not in ("high", "low", "human"):
        return set()
    if better == "human":
        if not isinstance(human, int | float) or not math.isfinite(human):
            return set()
        score = {s: -abs(v - human) for s, v in ok.items()}
    else:
        score = {s: (v if better == "high" else -v) for s, v in ok.items()}
    top = max(score.values())
    return {s for s, v in score.items() if v == top}


def _table(header: list[str], rows: list[list[str]]) -> str:
    h = "".join(f"<th>{html.escape(c)}</th>" for c in header)
    body = "".join("<tr>" + "".join(rows_ for rows_ in r) + "</tr>" for r in rows)
    return f'<div class="wrap"><table><tr>{h}</tr>{body}</table></div>'


def _metric_rows(summary, systems, group: str, spec, section: str = "groups") -> list[list[str]]:
    rows = []
    for key, label, better in spec:
        vals = {}
        for s in systems:
            block = summary["systems"][s].get(section, {})
            stats = block.get(group, {}) if section == "groups" else block
            vals[s] = stats.get(key)
        if all(v is None for v in vals.values()):
            continue
        best = _best(vals, better, vals.get(HUMAN))
        cells = [f"<td>{html.escape(label)}</td>"]
        for s in systems:
            cls = "ref" if s == HUMAN else ("best" if s in best else "")
            cells.append(f'<td class="{cls}">{_fmt(vals[s], key)}</td>')
        rows.append(cells)
    return rows


def render(summary: dict[str, Any]) -> str:
    meta = summary.get("meta", {})
    systems = list(summary["systems"])
    ds = meta.get("dataset", {})
    cfg = meta.get("config", {})
    title = f"bsmapper eval: {meta.get('label', '')}"
    parts = [
        f"<h1>{html.escape(title)}</h1>",
        f'<p class="muted">{html.escape(meta.get("created", ""))} · commit '
        f"{html.escape(meta.get('commit', '') or '?')} · {ds.get('songs', '?')} songs, "
        f"{ds.get('difficulties', '?')} human difficulties from the "
        f"<code>{html.escape(str(ds.get('split', '')))}</code> split of "
        f"<code>{html.escape(str(ds.get('path', '')))}</code>"
        + (f" (limit {cfg['limit']})" if cfg.get("limit") else "")
        + "</p>",
        "<p>Generated maps are made on each human map's own beat grid from the dataset "
        "features (no beat tracking), through the pipeline's generate and post-process "
        "stages. Green marks the best generated system in a row; for rows without a "
        "direction the best is the one closest to the human maps. The human column of the "
        "JSD rows is a split-half floor (one half of the songs against the other).</p>",
    ]
    sys_desc = []
    for d in meta.get("systems", []):
        extra = ", ".join(
            f"{k} {v}" for k, v in d.items() if k != "name" and v not in (None, "", [])
        )
        sys_desc.append(
            f"<li><b>{html.escape(d['name'])}</b> "
            f"<span class=muted>{html.escape(extra)}</span></li>"
        )
    if sys_desc:
        parts.append("<ul>" + "".join(sys_desc) + "</ul>")

    header = ["metric", *systems]
    parts.append("<h2>Headline (all difficulties)</h2>")
    parts.append(_table(header, _metric_rows(summary, systems, ALL, HEADLINE)))

    parts.append("<h2>By mode and difficulty</h2>")
    for key, label, better in GROUP_COLUMNS:
        rows = []
        for g in summary.get("groups", []):
            vals = {s: summary["systems"][s]["groups"].get(g, {}).get(key) for s in systems}
            if all(v is None for v in vals.values()):
                continue
            best = _best(vals, better, vals.get(HUMAN))
            cells = [f"<td>{html.escape(g)}</td>"]
            for s in systems:
                cls = "ref" if s == HUMAN else ("best" if s in best else "")
                cells.append(f'<td class="{cls}">{_fmt(vals[s], key)}</td>')
            rows.append(cells)
        parts.append(f"<h3>{html.escape(label)}</h3>")
        parts.append(_table(["group", *systems], rows))

    rules = sorted(
        {
            k
            for s in systems
            for k in summary["systems"][s]["groups"].get(ALL, {})
            if k.startswith("rule_")
            and k.endswith("_per_100")
            and (summary["systems"][s]["groups"][ALL][k] or 0) > 0
        }
    )
    if rules:
        spec = tuple((k, k[5:-8].replace("_", " "), "low") for k in rules)
        parts.append("<h2>Rule issues per 100 notes, by rule</h2>")
        parts.append(_table(header, _metric_rows(summary, systems, ALL, spec)))

    parts.append("<h2>Repetition on repeated audio</h2>")
    parts.append(
        f"<p class=muted>{rep.SEGMENT_BEATS}-beat segments on bar lines whose audio "
        f"matches an earlier one (cosine ≥ {rep.SIMILARITY}). Share of those pairs where "
        "the map repeats the notes exactly, mirrors them, keeps only the rhythm, or does "
        "something different.</p>"
    )
    spec = (
        ("rep_pairs", "pairs with notes", ""),
        *((f"rep_{c}", c.replace("_", " "), "human") for c in rep.CLASSES),
        ("rep_distance", "distance to human", "low"),
    )
    rows = []
    for g in [ALL, *[g for g in summary.get("groups", []) if g.startswith("Standard/E")]]:
        if g not in summary["systems"].get(HUMAN, {}).get("groups", {}):
            continue
        rows.append([f'<td colspan="{len(systems) + 1}"><b>{html.escape(g)}</b></td>'])
        rows += _metric_rows(summary, systems, g, spec)
    parts.append(_table(header, rows))

    styled = [s for s in systems if "style" in summary["systems"][s]]
    if styled:
        parts.append("<h2>Style: does the tech score follow the requested style?</h2>")
        diff = cfg.get("style_difficulty", "")
        parts.append(
            f"<p class=muted>Standard {html.escape(str(diff))} on "
            f"{cfg.get('style_songs', '?')} songs at each style. Tech score in techRating "
            "z units; percentile among human maps of the difficulty.</p>"
        )
        rows = []
        for s in styled:
            st = summary["systems"][s]["style"]
            for style, v in st["by_style"].items():
                rows.append(
                    [
                        f"<td>{html.escape(s)}</td>",
                        f"<td>{html.escape(style)}</td>",
                        f"<td>{_fmt(v['tech_score'])}</td>",
                        f"<td>{_fmt(v['style_percentile'], 'percentile')}</td>",
                        f"<td>{_fmt(v['nps'])}</td>",
                        f"<td>{v['n']}</td>",
                    ]
                )
            rows.append(
                [
                    f"<td>{html.escape(s)}</td>",
                    "<td><b>high − low</b></td>",
                    f"<td><b>{_fmt(st['tech_delta'])}</b></td>",
                    f"<td><b>{_fmt(st['percentile_delta'], 'percentile')}</b></td>",
                    f"<td>ρ {_fmt(st['spearman'])}</td>",
                    "<td></td>",
                ]
            )
        parts.append(_table(["system", "style", "tech score", "percentile", "NPS", "n"], rows))

    lit = [s for s in systems if "lighting" in summary["systems"][s]]
    if lit:
        parts.append("<h2>Lighting (vanilla basic events)</h2>")
        spec = (("n", "songs", ""), *LIGHT_HEADLINE)
        parts.append(_table(["metric", *lit], _metric_rows(summary, lit, "", spec, "lighting")))

    rows = []
    for s in systems:
        r = summary["systems"][s].get("run", {})
        notes = "; ".join(r.get("notes", []))[:300]
        rows.append(
            [
                f"<td>{html.escape(s)}</td>",
                f"<td>{r.get('songs', 0)}</td>",
                f"<td>{r.get('failed', 0)}</td>",
                f"<td>{_fmt(r.get('seconds_per_song'))}</td>",
                f'<td style="text-align:left;white-space:normal">{html.escape(notes)}</td>',
            ]
        )
    parts.append("<h2>Run</h2>")
    parts.append(_table(["system", "songs", "failed", "s / song", "notes"], rows))
    parts.append(GLOSSARY)
    body = "\n".join(parts)
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width, initial-scale=1">'
        f"<title>bsmapper eval</title><style>{CSS}</style></head><body>{body}</body></html>"
    )


GLOSSARY = """
<h2>Metrics</h2>
<dl>
<dt>onset F1 ±k</dt><dd>note onset ticks (either colour) matched one-to-one to the human
map's within k ticks (1 tick = 1/24 beat)</dd>
<dt>NPS / human</dt><dd>colour notes per second (first to last note) over the same song's
human difficulty</dd>
<dt>NPS percentile</dt><dd>percentile among dataset train/val human maps of the same mode
and difficulty within ±15% BPM; 50 = typical human</dd>
<dt>step offset</dt><dd>NPS placed on the ladder of human median NPS per difficulty at the
song's BPM, minus the map's own level; −1 = as dense as a typical map one difficulty lower</dd>
<dt>rule errors / warnings</dt><dd><code>bsmapper.rules</code> issues per 100 notes</dd>
<dt>JSD</dt><dd>Jensen-Shannon divergence (base 2) of pooled counts against the human maps
of the same group: cut direction, cell, per-swing pattern (direction + cell, per hand) and
its 2-/3-grams, and onset gaps (rhythm)</dd>
<dt>red share, crossover rate</dt><dd>share of red notes; share of swings landing past the
other hand's latest swing within one beat (dataset tech feature)</dd>
<dt>tech score</dt><dd>dataset tech-score calibration (techRating z units within the
difficulty)</dd>
<dt>repetition</dt><dd>see the repetition section; distance is total variation between
class shares</dd>
<dt>lighting</dt><dd>light events (types 0–4) per bar; share of event ticks within 2 ticks
of an onset peak; share of strong onsets lit; share on beats and bar lines; share of
section boundaries with a spin/zoom, boost or colour change within a beat; Spearman r of
light density and section energy</dd>
</dl>
"""


def write_report(summary: dict[str, Any], path: Path) -> Path:
    path.write_text(render(summary), encoding="utf-8")
    return path
