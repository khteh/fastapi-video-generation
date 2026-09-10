"""Renders `Slide` objects (see slides.py) into PNG images with Pillow.

All layout values (font sizes, offsets, line spacing, icon radii) are
defined relative to a 1280x720 reference canvas and scaled by the actual
requested resolution. Without this, rendering at 4K (3840x2160) would
just stretch a 720p-sized layout across a much bigger canvas — tiny text
in the corner of an otherwise empty frame — rather than a properly
scaled, crisp 4K slide.

Heading/body text is also fit to the available width (wrapped to
multiple lines for headings, truncated with an ellipsis for body lines)
rather than letting it overflow off the edge of the frame. This is a
safety net independent of `script_providers.py`'s `_short_title()`
(which pre-shortens long topics for its own headings) — this module has
no way to know whether the text it's given has already been shortened,
so it fits whatever it receives, from any source.
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.generation.script_providers import Slide

_BG = (18, 28, 48)
_HEADER_BG = (33, 92, 168)
_ACCENT = (255, 196, 79)
_TEXT = (240, 244, 250)
_MUTED = (176, 196, 222)

_KIND_ACCENT = {
    "title": (255, 196, 79),
    "content": (110, 200, 255),
    "diagram": (140, 230, 170),
    "summary": (255, 150, 150),
}

# Reference canvas every layout constant below is expressed in terms of.
_REFERENCE_WIDTH = 1280
_REFERENCE_HEIGHT = 720


def _font(size: int) -> ImageFont.ImageFont:
    # Pillow's built-in default font avoids any dependency on system font
    # files being present; `size` is honored on Pillow >= 9.2 via the
    # optional argument, with a silent fallback on older versions.
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


class PillowSlideRenderer:
    def __init__(self, width: int, height: int) -> None:
        self._width = width
        self._height = height
        # Scale uniformly off width so aspect-ratio changes don't distort
        # text/icons; height differences beyond that are handled by the
        # caller choosing a sane width/height pair (e.g. 16:9 at any size).
        self._scale = width / _REFERENCE_WIDTH

    def _s(self, value: float) -> int:
        """Scale a reference-canvas (1280-wide) pixel value to the actual
        output resolution, with a floor of 1px so nothing disappears."""
        return max(1, round(value * self._scale))

    @staticmethod
    def _wrap_to_lines(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
        """Greedily wrap `text` into lines that each fit within
        `max_width` pixels, breaking at word boundaries. A single "word"
        that alone is wider than `max_width` (e.g. a pathologically long
        string with no spaces at all) is hard-broken into width-fitting
        chunks instead — otherwise no amount of wrapping around it would
        ever bring it under the limit, and it would overflow regardless
        of how many lines `_fit_lines` is allowed."""
        words = text.split()
        if not words:
            return [text]

        lines: list[str] = []
        current = ""
        for word in words:
            while draw.textlength(word, font=font) > max_width:
                # Binary-search the longest prefix of `word` that fits on
                # its own line, hard-break there, and continue with the
                # remainder as if it were a fresh "word".
                lo, hi, fit = 1, len(word), 1
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if draw.textlength(word[:mid], font=font) <= max_width:
                        fit = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1
                if current:
                    lines.append(current)
                    current = ""
                lines.append(word[:fit])
                word = word[fit:]
            candidate = f"{current} {word}".strip() if current else word
            if draw.textlength(candidate, font=font) <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    @classmethod
    def _fit_lines(
        cls,
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.ImageFont,
        max_width: int,
        max_lines: int,
    ) -> list[str]:
        """Wrap `text` to at most `max_lines` lines that each fit within
        `max_width`. If wrapping alone doesn't fit everything in
        `max_lines`, the last line is truncated with an ellipsis rather
        than letting anything overflow the frame."""
        lines = cls._wrap_to_lines(draw, text, font, max_width)
        if len(lines) <= max_lines:
            return lines

        kept = lines[:max_lines]
        last = kept[-1]
        ellipsis = "…"
        while last and draw.textlength(last + ellipsis, font=font) > max_width:
            last = last[:-1].rstrip()
        kept[-1] = last + ellipsis if last else ellipsis
        return kept

    @classmethod
    def _fit_single_line(
        cls, draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int
    ) -> str:
        """Truncate `text` with an ellipsis if it doesn't fit on one line
        within `max_width` — used for body_lines, which are meant to stay
        single lines (multiple bullet entries already provide multi-line
        structure)."""
        if draw.textlength(text, font=font) <= max_width:
            return text
        truncated = text
        ellipsis = "…"
        while truncated and draw.textlength(truncated + ellipsis, font=font) > max_width:
            truncated = truncated[:-1].rstrip()
        return truncated + ellipsis if truncated else ellipsis

    def render(self, slide: Slide, slide_index: int, total_slides: int) -> Image.Image:
        img = Image.new("RGB", (self._width, self._height), _BG)
        draw = ImageDraw.Draw(img)
        accent = _KIND_ACCENT.get(slide.kind, _ACCENT)

        heading_font = _font(self._s(44))
        heading_margin = self._s(50)
        heading_max_width = self._width - 2 * heading_margin
        heading_lines = self._fit_lines(draw, slide.heading, heading_font, heading_max_width, max_lines=2)

        heading_top_pad = self._s(40)
        heading_line_height = self._s(50)
        heading_bottom_pad = self._s(40)
        header_h = heading_top_pad + len(heading_lines) * heading_line_height + heading_bottom_pad

        draw.rectangle([0, 0, self._width, header_h], fill=_HEADER_BG)
        draw.rectangle([0, header_h, self._width, header_h + self._s(6)], fill=accent)

        for i, line in enumerate(heading_lines):
            draw.text(
                (heading_margin, heading_top_pad + i * heading_line_height),
                line,
                fill=_TEXT,
                font=heading_font,
            )

        body_font = _font(self._s(32))
        body_margin = self._s(60)
        # Leave enough clearance on the right that body text can't run
        # into the diagram illustration, which sits around
        # x = width - s(320), regardless of a line's vertical position.
        body_max_width = self._width - body_margin - self._s(420)
        y = header_h + self._s(70)
        line_height = self._s(48)
        for line in slide.body_lines:
            fitted = self._fit_single_line(draw, line, body_font, body_max_width)
            draw.text((body_margin, y), fitted, fill=_TEXT, font=body_font)
            y += line_height

        self._draw_illustration(draw, slide, accent)

        # Progress footer: small dots showing position in the deck.
        dot_radius = self._s(8)
        gap = self._s(28)
        total_width = (total_slides - 1) * gap
        start_x = self._width // 2 - total_width // 2
        dot_y = self._height - self._s(40)
        for i in range(total_slides):
            x = start_x + i * gap
            fill = accent if i == slide_index else (70, 80, 100)
            draw.ellipse(
                [x - dot_radius, dot_y - dot_radius, x + dot_radius, dot_y + dot_radius],
                fill=fill,
            )

        draw.text(
            (self._width - self._s(260), self._height - self._s(40)),
            "STEM Explainer",
            fill=_MUTED,
            font=_font(self._s(20)),
        )
        return img

    def _draw_illustration(self, draw: ImageDraw.ImageDraw, slide: Slide, accent) -> None:
        """A simple generic molecule/bond sketch as illustrative visual
        interest — not concept-specific chemical/physical accuracy, just
        something that reads as generically 'STEM' on screen alongside the
        narration."""
        cx, cy = self._width - self._s(320), self._height // 2 + self._s(40)
        if slide.kind == "title":
            self._draw_molecule(draw, cx, cy, accent, atoms=3)
        elif slide.kind == "diagram":
            self._draw_equilibrium_arrows(draw, cx, cy, accent)
        elif slide.kind == "summary":
            self._draw_molecule(draw, cx, cy, accent, atoms=4)
        else:
            self._draw_molecule(draw, cx, cy, accent, atoms=2)

    def _draw_molecule(self, draw: ImageDraw.ImageDraw, cx: int, cy: int, accent, atoms: int) -> None:
        radius = self._s(46)
        orbit = self._s(120)
        line_width = self._s(5)
        outline_width = self._s(6)

        centers = []
        for i in range(atoms):
            angle = (2 * math.pi / atoms) * i - math.pi / 2
            x = cx + int(orbit * math.cos(angle))
            y = cy + int(orbit * math.sin(angle)) if atoms > 1 else cy
            centers.append((x, y))
        if atoms == 1:
            centers = [(cx, cy)]

        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                if atoms <= 3 or (j == i + 1) or (i == 0 and j == len(centers) - 1):
                    draw.line([centers[i], centers[j]], fill=(200, 210, 225), width=line_width)

        palette = [accent, (255, 255, 255), (140, 230, 170), (255, 150, 150)]
        for idx, (x, y) in enumerate(centers):
            color = palette[idx % len(palette)]
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline=color, width=outline_width)

    def _draw_equilibrium_arrows(self, draw: ImageDraw.ImageDraw, cx: int, cy: int, accent) -> None:
        width = self._s(220)
        offset_y = self._s(15)
        line_width = self._s(5)
        head_len = self._s(20)

        draw.line([cx - width // 2, cy - offset_y, cx + width // 2, cy - offset_y], fill=accent, width=line_width)
        draw.polygon(
            [
                (cx + width // 2, cy - offset_y),
                (cx + width // 2 - head_len, cy - offset_y - self._s(10)),
                (cx + width // 2 - head_len, cy - offset_y + self._s(10)),
            ],
            fill=accent,
        )
        draw.line(
            [cx - width // 2, cy + offset_y, cx + width // 2, cy + offset_y],
            fill=(200, 210, 225),
            width=line_width,
        )
        draw.polygon(
            [
                (cx - width // 2, cy + offset_y),
                (cx - width // 2 + head_len, cy + offset_y - self._s(10)),
                (cx - width // 2 + head_len, cy + offset_y + self._s(10)),
            ],
            fill=(200, 210, 225),
        )
