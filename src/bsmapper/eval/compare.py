"""``bsmapper evaluate --compare A.json B.json``: one system of each run, side by side.

By default each file's first model system is compared (its stub if it has none), so
``note_v0``'s run against ``note_v1``'s just works. Each row shows A, B, B - A and
whether B is better: higher / lower / closer to the human maps, per metric. Human
values come from B's run (A's when B has none).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from bsmapper.eval.runner import ALL, HEADLINE, LIGHT_HEADLINE
from bsmapper.eval.systems import HUMAN, STUB

COMPARE_GROUPS = (ALL, "Standard/Hard", "Standard/Expert", "Standard/ExpertPlus")


def load_summary(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def default_system(summary: dict[str, Any]) -> str:
    names = list(summary["systems"])
    models = [n for n in names if n not in (HUMAN, STUB)]
    if models:
        return models[0]
    if STUB in names:
        return STUB
    raise ValueError("the summary has no generated system")


def _num(v: Any) -> float | None:
    return v if isinstance(v, int | float) and math.isfinite(v) else None


def verdict(a: float | None, b: float | None, better: str, human: float | None) -> str:
    """'better', 'worse' or '' for B against A."""
    if a is None or b is None or a == b:
        return ""
    if better == "high":
        return "better" if b > a else "worse"
    if better == "low":
        return "better" if b < a else "worse"
    if better == "human" and human is not None:
        da, db = abs(a - human), abs(b - human)
        return "" if da == db else ("better" if db < da else "worse")
    return ""


def compare_rows(
    a: dict[str, Any],
    b: dict[str, Any],
    sys_a: str | None = None,
    sys_b: str | None = None,
    groups: tuple[str, ...] = COMPARE_GROUPS,
) -> tuple[str, str, list[dict[str, Any]]]:
    sys_a = sys_a or default_system(a)
    sys_b = sys_b or default_system(b)
    for s, summ, name in ((sys_a, a, "A"), (sys_b, b, "B")):
        if s not in summ["systems"]:
            raise ValueError(f"{name} has no system {s!r}; it has {', '.join(summ['systems'])}")
    rows = []
    sa, sb = a["systems"][sys_a], b["systems"][sys_b]
    human_src = b if HUMAN in b["systems"] else a
    human = human_src["systems"].get(HUMAN, {})
    for g in groups:
        ga, gb = sa["groups"].get(g), sb["groups"].get(g)
        if ga is None or gb is None:
            continue
        gh = human.get("groups", {}).get(g, {})
        for key, label, better in HEADLINE:
            rows.append(_row(g, key, label, better, ga, gb, gh))
    la, lb = sa.get("lighting"), sb.get("lighting")
    if la and lb:
        lh = human.get("lighting", {})
        for key, label, better in LIGHT_HEADLINE:
            rows.append(_row("lighting", key, label, better, la, lb, lh))
    ta, tb = sa.get("style"), sb.get("style")
    if ta and tb:
        for key, label in (("tech_delta", "tech score high-low style"), ("spearman", "style ρ")):
            rows.append(_row("style", key, label, "high", ta, tb, {}))
    return sys_a, sys_b, [r for r in rows if r["a"] is not None or r["b"] is not None]


def _row(group, key, label, better, ga, gb, gh) -> dict[str, Any]:
    va, vb, vh = _num(ga.get(key)), _num(gb.get(key)), _num(gh.get(key))
    return {
        "group": group,
        "metric": key,
        "label": label,
        "a": va,
        "b": vb,
        "human": vh,
        "delta": vb - va if va is not None and vb is not None else None,
        "verdict": verdict(va, vb, better, vh),
    }


def format_compare(
    a: dict[str, Any], b: dict[str, Any], sys_a: str | None = None, sys_b: str | None = None
) -> str:
    sys_a, sys_b, rows = compare_rows(a, b, sys_a, sys_b)
    la = a.get("meta", {}).get("label", "A")
    lb = b.get("meta", {}).get("label", "B")
    lines = [f"A = {sys_a} ({la})   B = {sys_b} ({lb})", ""]

    def f(v):
        return "–" if v is None else f"{v:9.3f}"

    head = f"{'metric':<32} {'A':>9} {'B':>9} {'B-A':>9} {'human':>9}"
    group = None
    better = worse = 0
    for r in rows:
        if r["group"] != group:
            group = r["group"]
            lines += ["", f"[{group}]", head]
        mark = {"better": "  + better", "worse": "  - worse"}.get(r["verdict"], "")
        better += r["verdict"] == "better"
        worse += r["verdict"] == "worse"
        lines.append(
            f"{r['label'][:32]:<32} {f(r['a']):>9} {f(r['b']):>9} {f(r['delta']):>9} "
            f"{f(r['human']):>9}{mark}"
        )
    sa = set(a.get("meta", {}).get("dataset", {}).get("keys", []))
    sb = set(b.get("meta", {}).get("dataset", {}).get("keys", []))
    lines += ["", f"B is better on {better} rows and worse on {worse}."]
    if sa and sb and sa != sb:
        lines.append(
            f"Warning: the runs scored different songs ({len(sa)} vs {len(sb)}, "
            f"{len(sa & sb)} shared); differences may come from the song sets."
        )
    return "\n".join(lines)
