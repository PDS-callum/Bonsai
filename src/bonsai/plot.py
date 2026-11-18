"""The frontier plot: quality against speed in the pot, as a standalone SVG.

Each panel sets one quality measure against generation speed in the pot, at
the pot's own settings. Every tree is a point, with its interval as a
whisker; the frontier (trees no other tree beats on both) is a dashed
staircase. Series are seed families, with pruned trees as their own series;
each has a colour and a marker shape, so colour is never the only cue, and
the trees on the frontier are labelled directly.

The SVG is written by hand (no plotting library) and carries its own light and
dark colours, so it reads well embedded in a README in either theme. Points
have native tooltips when the file is opened on its own.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape

from bonsai.journal import FrontierPoint

# The reference palette's first three categorical slots, which stay distinct
# for every pair under colour-vision deficiency in both themes; any further
# series fold into a neutral "Other".
LIGHT = ("#2a78d6", "#eb6834", "#1baf7a")
DARK = ("#3987e5", "#d95926", "#199e70")
SHAPES = ("circle", "square", "triangle")

STYLE = """
.surface { fill: #fcfcfb; }
.title { font: 600 15px system-ui, sans-serif; fill: #0b0b0b; }
.text { font: 12px system-ui, sans-serif; fill: #52514e; }
.label { font: 11px system-ui, sans-serif; fill: #52514e; }
.grid { stroke: #e7e6e2; stroke-width: 1; }
.axis { stroke: #b9b8b2; stroke-width: 1; }
.frontier { stroke: #52514e; stroke-width: 1.5; stroke-dasharray: 5 4; fill: none; }
.ring { stroke: #fcfcfb; stroke-width: 2; }
.other { fill: #8a8984; stroke: #8a8984; }
%(light)s
@media (prefers-color-scheme: dark) {
  .surface { fill: #1a1a19; }
  .title { fill: #f0efec; }
  .text, .label { fill: #c3c2b7; }
  .grid { stroke: #2e2e2c; }
  .axis { stroke: #55554f; }
  .frontier { stroke: #c3c2b7; }
  .ring { stroke: #1a1a19; }
  .other { fill: #8f8e88; stroke: #8f8e88; }
  %(dark)s
}
"""


@dataclass(frozen=True)
class Panel:
    title: str
    points: Sequence[FrontierPoint]
    low: float
    high: float
    step: float
    unit: str = "%"


@dataclass(frozen=True)
class Series:
    key: tuple[str, bool]
    label: str
    slot: int | None
    """Palette slot, or None for "Other"."""


def family(point: FrontierPoint) -> str:
    """``qwen2.5-0.5b-instruct-weights`` -> ``qwen2.5-0.5b-instruct``."""
    return point.origin.removesuffix("-weights") or point.seed


def pretty(name: str) -> str:
    """``qwen2.5-0.5b-instruct`` -> ``Qwen2.5-0.5B``, ``smollm2-360m-instruct``
    -> ``SmolLM2-360M``: a short display name for a seed family."""
    known = {"qwen": "Qwen", "smollm": "SmolLM", "llama": "Llama"}
    base = name.removesuffix("-instruct")
    head, _, size = base.rpartition("-")
    for prefix, proper in known.items():
        if head.startswith(prefix):
            head = proper + head[len(prefix) :]
            break
    return f"{head}-{size.upper()}" if head else base


def short_label(point: FrontierPoint) -> str:
    """How a point is labelled on the plot, e.g. ``block6 Q4_K_M``."""
    cut = point.seed.removeprefix(family(point)).strip("-")
    quant = point.quant.upper()
    return f"{cut} {quant}" if point.pruned and cut else quant


def series_of(points: Sequence[FrontierPoint]) -> dict[tuple[str, bool], Series]:
    keys = sorted({(family(p), p.pruned) for p in points})
    result = {}
    for index, key in enumerate(keys):
        slot = index if index < len(LIGHT) else None
        name = pretty(key[0]) + (", pruned" if key[1] else "")
        result[key] = Series(key, name if slot is not None else "Other", slot)
    return result


def _marker(shape: str, x: float, y: float, css: str, tip: str) -> str:
    r = 4.5
    if shape == "circle":
        body = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}"'
    elif shape == "square":
        body = (
            f'<rect x="{x - r:.1f}" y="{y - r:.1f}" width="{2 * r}" height="{2 * r}" '
            'rx="1"'
        )
    else:
        h = r * 1.25
        body = (
            f'<path d="M{x:.1f},{y - h:.1f} L{x + h:.1f},{y + h * 0.75:.1f} '
            f'L{x - h:.1f},{y + h * 0.75:.1f} Z"'
        )
    tag = body[1:].split()[0]
    return f'{body} class="{css} ring"><title>{escape(tip)}</title></{tag}>'


def _place(
    text: str, x: float, y: float, placed: list[tuple[float, float, float, float]]
) -> tuple[float, float]:
    """Where to put a point's label: above right, then below right, then above
    left, then further up, avoiding labels already placed."""
    width, height = 6.2 * len(text), 12.0
    candidates = [
        (x + 8, y - 7), (x + 8, y + 15), (x - 8 - width, y - 7),
        (x - 8 - width, y + 15), (x + 8, y - 21), (x - 8 - width, y - 21),
    ]  # fmt: skip

    def clear(lx: float, ly: float) -> bool:
        return all(
            lx + width < px or px + pw < lx or ly < py - ph or py < ly - height
            for px, py, pw, ph in placed
        )

    lx, ly = next((c for c in candidates if clear(*c)), candidates[0])
    placed.append((lx, ly, width, height))
    return lx, ly


def _ticks(low: float, high: float, step: float) -> list[float]:
    count = round((high - low) / step)
    return [low + i * step for i in range(count + 1)]


def render(panels: Sequence[Panel], *, title: str, x_label: str) -> str:
    """The plot as an SVG document."""
    every = [p for panel in panels for p in panel.points]
    series = series_of(every)
    width, panel_h = 920, 330
    left, gap, right = 56, 56, 20
    top = 92
    plot_w = (width - left - right - gap * (len(panels) - 1)) / len(panels)
    plot_h = panel_h - 70
    height = top + panel_h
    # Headroom past the fastest tree, so its label fits.
    fastest = max((p.generation_tps for p in every), default=1.0)
    x_max = max(10.0, math.ceil((fastest + 6) / 10) * 10)

    light = "\n".join(
        f".s{i} {{ fill: {c}; stroke: {c}; }}" for i, c in enumerate(LIGHT)
    )
    dark = "\n  ".join(
        f".s{i} {{ fill: {c}; stroke: {c}; }}" for i, c in enumerate(DARK)
    )
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="{escape(title)}">',
        f"<title>{escape(title)}</title>",
        f"<style>{STYLE % {'light': light, 'dark': dark}}</style>",
        f'<rect class="surface" width="{width}" height="{height}" rx="8"/>',
        f'<text class="title" x="{left}" y="30">{escape(title)}</text>',
    ]
    # Legend: one row, marker then name, in a fixed order.
    key_x = float(left)
    shown = {s.label: s for s in series.values()}  # "Other" appears once
    for s in sorted(shown.values(), key=lambda s: (s.slot is None, s.slot or 0)):
        shape = SHAPES[s.slot] if s.slot is not None else "circle"
        css = f"s{s.slot}" if s.slot is not None else "other"
        out.append(_marker(shape, key_x + 5, 56, css, s.label))
        out.append(
            f'<text class="text" x="{key_x + 16}" y="60">{escape(s.label)}</text>'
        )
        key_x += 30 + 7 * len(s.label)
    out.append(
        f'<line class="frontier" x1="{key_x}" y1="56" x2="{key_x + 22}" y2="56"/>'
        f'<text class="text" x="{key_x + 28}" y="60">Frontier</text>'
    )

    for index, panel in enumerate(panels):
        x0 = left + index * (plot_w + gap)
        y0 = top + 20

        def sx(v: float, x0: float = x0) -> float:
            return x0 + v / x_max * plot_w

        def sy(v: float, y0: float = y0, panel: Panel = panel) -> float:
            v = min(max(v, panel.low), panel.high)
            return y0 + plot_h - (v - panel.low) / (panel.high - panel.low) * plot_h

        out.append(
            f'<text class="text" x="{x0}" y="{y0 - 8}">{escape(panel.title)}</text>'
        )
        for v in _ticks(panel.low, panel.high, panel.step):
            out.append(
                f'<line class="grid" x1="{x0}" y1="{sy(v):.1f}" '
                f'x2="{x0 + plot_w:.1f}" y2="{sy(v):.1f}"/>'
                f'<text class="label" x="{x0 - 6}" y="{sy(v) + 4:.1f}" '
                f'text-anchor="end">{v:g}{panel.unit}</text>'
            )
        for v in _ticks(0, x_max, 10):
            out.append(
                f'<text class="label" x="{sx(v):.1f}" y="{y0 + plot_h + 16:.1f}" '
                f'text-anchor="middle">{v:g}</text>'
            )
        out.append(
            f'<line class="axis" x1="{x0}" y1="{y0 + plot_h}" '
            f'x2="{x0 + plot_w:.1f}" y2="{y0 + plot_h}"/>'
        )
        out.append(
            f'<text class="text" x="{x0 + plot_w / 2:.1f}" y="{y0 + plot_h + 36:.1f}" '
            f'text-anchor="middle">{escape(x_label)}</text>'
        )
        # The frontier: a staircase from the fastest tree to the best.
        best = sorted((p for p in panel.points if p.on_frontier),
                      key=lambda p: -p.generation_tps)  # fmt: skip
        if best:
            first = best[0]
            path = [
                f"M{sx(first.generation_tps):.1f},{y0 + plot_h:.1f}",
                f"V{sy(first.quality):.1f}",
            ]
            path += [
                f"H{sx(p.generation_tps):.1f} V{sy(p.quality):.1f}" for p in best[1:]
            ]
            path.append(f"H{x0:.1f}")
            out.append(f'<path class="frontier" d="{" ".join(path)}"/>')
        placed: list[tuple[float, float, float, float]] = []
        for p in sorted(panel.points, key=lambda p: p.on_frontier):
            s = series[(family(p), p.pruned)]
            css = f"s{s.slot}" if s.slot is not None else "other"
            shape = SHAPES[s.slot] if s.slot is not None else "circle"
            x, y = sx(p.generation_tps), sy(p.quality)
            if p.low is not None and p.high is not None:
                out.append(
                    f'<line class="{css}" x1="{x:.1f}" y1="{sy(p.low):.1f}" '
                    f'x2="{x:.1f}" y2="{sy(p.high):.1f}" stroke-width="1.5" '
                    'opacity="0.45"/>'
                )
            tip = (
                f"{p.file}: {p.generation_tps:.1f} tokens/s, "
                f"{p.quality:.1f}{panel.unit}"
            )
            out.append(_marker(shape, x, y, css, tip))
            if p.on_frontier:
                text = short_label(p)
                lx, ly = _place(text, x, y, placed)
                out.append(
                    f'<text class="label" x="{lx:.1f}" y="{ly:.1f}">'
                    f"{escape(text)}</text>"
                )
    out.append("</svg>")
    return "\n".join(out) + "\n"
