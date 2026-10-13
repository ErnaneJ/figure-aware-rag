"""Match an embedded image to its printed caption using the page layout."""
import re

import fitz

# A caption starts with a figure label and a number followed by punctuation, e.g. "Figure 3." or
# "Fig. 2:". Body text that mentions a figure ("Figure 13 shows ...") does not match.
CAPTION_START = re.compile(r"^\s*(fig\.?|figure|image|photo|graph|chart|plate)\s*\d+[a-z]?\s*[.:)\-–]",
                           re.IGNORECASE)
MAX_GAP = 60  # points between the image edge and the caption block


def image_rect(page: fitz.Page, xref: int) -> fitz.Rect | None:
    """Position of the image on the page, or None when PyMuPDF cannot place it."""
    rects = page.get_image_rects(xref)
    return rects[0] if rects else None


def caption_blocks(page: fitz.Page) -> list[tuple[fitz.Rect, str]]:
    """Text blocks of the page that start like a caption."""
    blocks = []
    for b in page.get_text("blocks"):
        text = b[4].strip()
        if b[6] == 0 and CAPTION_START.match(text):
            blocks.append((fitz.Rect(b[:4]), " ".join(text.split())))
    return blocks


def layout_caption(page: fitz.Page, rect: fitz.Rect, blocks=None) -> str | None:
    """Caption printed nearest to the image, below it first, then above it.

    A candidate must overlap the image horizontally (same column) and lie within MAX_GAP points
    of its bottom or top edge. Returns None when no candidate exists."""
    blocks = caption_blocks(page) if blocks is None else blocks
    below, above = [], []
    for brect, text in blocks:
        if min(rect.x1, brect.x1) - max(rect.x0, brect.x0) <= 0:
            continue
        gap_below = brect.y0 - rect.y1
        gap_above = rect.y0 - brect.y1
        if -5 <= gap_below <= MAX_GAP:
            below.append((gap_below, text))
        elif -5 <= gap_above <= MAX_GAP:
            above.append((gap_above, text))
    for candidates in (below, above):
        if candidates:
            return min(candidates)[1]
    return None


def describe_position(page: fitz.Page, rect: fitz.Rect | None) -> str:
    """Plain description of where the image sits on the page, for the LLM fallback."""
    if rect is None:
        return "an unknown position"
    width, height = page.rect.width, page.rect.height
    cy = (rect.y0 + rect.y1) / 2
    vertical = "top third" if cy < height / 3 else "middle third" if cy < 2 * height / 3 else "bottom third"
    if rect.width > width * 0.6:
        horizontal = "full width"
    elif (rect.x0 + rect.x1) / 2 < width / 2:
        horizontal = "left column"
    else:
        horizontal = "right column"
    return f"{vertical}, {horizontal}"
