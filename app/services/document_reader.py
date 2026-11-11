"""Read supplied CV text and render scanned PDF pages for vision, without OCR."""

import base64
import mimetypes
from pathlib import Path

import fitz


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def read_cv(path: Path) -> str:
    if path.suffix.lower() != ".txt":
        raise ValueError("CV input must be a .txt file")
    return path.read_text(encoding="utf-8")


def image_data_uri(path: Path, *, page: int = 0, render_scale: float = 1.5) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        with fitz.open(path) as pdf:
            if page < 0 or page >= len(pdf):
                raise ValueError(f"PDF page index {page} is out of range")
            pixmap = pdf[page].get_pixmap(
                matrix=fitz.Matrix(render_scale, render_scale), alpha=False
            )
            raw = pixmap.tobytes("png")
            mime = "image/png"
    elif suffix in IMAGE_SUFFIXES:
        raw = path.read_bytes()
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
    else:
        raise ValueError("Supported invoice formats: PNG, JPEG, WebP and PDF")
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
