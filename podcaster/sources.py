"""Fetching and parsing the source material."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

try:  # optional, only needed for .docx sources
    import docx  # type: ignore
except Exception:  # pragma: no cover
    docx = None

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover
    BeautifulSoup = None


SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown", ".txt", ".text", ".html", ".htm", ".docx", ".rtf"}
MAX_UPLOAD_BYTES = 60 * 1024 * 1024


@dataclass
class SourceDoc:
    text: str
    title: str = ""
    origin: str = ""
    kind: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


def _decode(data: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "gb18030", "big5", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _strip_markdown(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"^\s{0,3}>\s?", "", text, flags=re.M)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"(\*\*\*|\*\*|\*|__|_|~~)", "", text)
    text = re.sub(r"^\s{0,3}([-*+]|\d+\.)\s+", "", text, flags=re.M)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _html_to_text(html: str) -> tuple[str, str]:
    if BeautifulSoup is None:
        raise RuntimeError("beautifulsoup4 is required to parse HTML sources")
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "iframe", "svg", "form", "nav", "footer", "header"]):
        tag.decompose()
    title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
    main = soup.find("article") or soup.find("main") or soup.body or soup
    text = main.get_text("\n", strip=True)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), title


def _pdf_to_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        text = "\n\n".join(pages)
        if len(text.strip()) < 40:  # scanned PDF → OCR-ish fallback via pdfplumber
            try:
                import pdfplumber

                with pdfplumber.open(io.BytesIO(data)) as pdf:
                    text = "\n\n".join((p.extract_text() or "") for p in pdf.pages)
            except Exception:
                pass
        return text.strip()
    except Exception:
        try:
            import pdfplumber

            with pdfplumber.open(io.BytesIO(data)) as pdf:
                return "\n\n".join((p.extract_text() or "") for p in pdf.pages).strip()
        except Exception as exc:
            raise RuntimeError(f"Could not read PDF: {exc}") from exc


def _docx_to_text(data: bytes) -> str:
    if docx is None:
        raise RuntimeError("python-docx is not installed; cannot read .docx sources")
    document = docx.Document(io.BytesIO(data))
    parts = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells))
    return "\n".join(parts).strip()


def parse_upload(filename: str, data: bytes) -> SourceDoc:
    suffix = Path(filename).suffix.lower()
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    if suffix == ".pdf":
        text = _pdf_to_text(data)
        kind = "pdf"
    elif suffix == ".docx":
        text = _docx_to_text(data)
        kind = "docx"
    elif suffix in (".html", ".htm"):
        text, title = _html_to_text(_decode(data))
        kind = "html"
    elif suffix in (".md", ".markdown"):
        text = _strip_markdown(_decode(data))
        kind = "markdown"
    elif suffix == ".rtf":
        text = re.sub(r"\\[a-z]+\d*\s?", " ", _decode(data, errors="replace"))
        text = re.sub(r"[{}]", "", text)
        kind = "rtf"
    elif suffix in (".txt", ".text", ""):
        text = _decode(data)
        kind = "text"
    else:
        raise ValueError(f"Unsupported file type '{suffix}'. Use pdf, docx, md, txt or html.")

    if not text.strip():
        raise ValueError("No readable text found in the file (it may be a scanned PDF).")
    title = Path(filename).stem
    return SourceDoc(text=text, title=title, origin=filename, kind=kind,
                     meta={"bytes": len(data), "chars": len(text)})


ACCEPT_HTML = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"

# Some sites (Wikipedia among them) reject generic browser user-agents when
# they can tell the client is not a browser, so try a descriptive agent first
# and fall back to the browser one.
USER_AGENTS = [
    "Podcaster/1.0 (local podcast script generator; https://localhost)",
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
     "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
]


def fetch_url(url: str, timeout: float = 45.0) -> SourceDoc:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    raw = b""
    content_type = ""
    response = None
    last_error: Exception | None = None
    for agent in USER_AGENTS:
        headers = {
            "User-Agent": agent,
            "Accept": ACCEPT_HTML,
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        try:
            with httpx.Client(follow_redirects=True, timeout=timeout, headers=headers) as client:
                response = client.get(url)
            if response.status_code < 400:
                raw = response.content
                content_type = response.headers.get("content-type", "").lower()
                break
            last_error = httpx.HTTPStatusError(
                f"HTTP {response.status_code}", request=response.request, response=response)
        except httpx.HTTPError as exc:
            last_error = exc
    if response is None or not raw:
        raise ValueError(f"Could not fetch {url}: {last_error}")

    if "pdf" in content_type or raw[:4] == b"%PDF":
        text = _pdf_to_text(raw)
        kind = "pdf"
        title = url
    elif "json" in content_type:
        text = _decode(raw)
        kind = "json"
        title = url
    else:
        text, title = _html_to_text(_decode(raw))
        kind = "html"

    if not text.strip():
        raise ValueError("The page returned no extractable text.")
    return SourceDoc(text=text, title=title or url, origin=url, kind=kind,
                     meta={"chars": len(text), "content_type": content_type})


def load_local_file(path: str) -> SourceDoc:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    doc = parse_upload(p.name, p.read_bytes())
    doc.origin = str(p)
    doc.title = p.stem
    return doc
