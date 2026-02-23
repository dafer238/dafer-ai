"""Markdown + Notebook documentation server with Ayu Mirage theme,
sidebar navigation, and sequential prev/next page arrows.

Reads the table of contents from ``myst.yml`` so the page ordering is
defined in a single place (shared with a local Jupyter Book build).
"""

from __future__ import annotations

import bisect
import re
import unicodedata
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
import markdown

app = FastAPI()

BASE_DIR = Path(__file__).parent.resolve()

# ── Load static assets at startup ────────────────────────────────────────────

_STYLE = (BASE_DIR / "static" / "style.css").read_text(encoding="utf-8")
_SCRIPT = (BASE_DIR / "static" / "script.js").read_text(encoding="utf-8")
_TEMPLATE = (BASE_DIR / "templates" / "page.html").read_text(encoding="utf-8")

# ── Table of contents: flat ordered list from myst.yml ───────────────────────

PageEntry = dict  # {"file": str, "title": str}


def _parse_toc() -> list[PageEntry]:
    """Read myst.yml and flatten the toc into an ordered page list."""
    myst_path = BASE_DIR / "myst.yml"
    if not myst_path.exists():
        return []
    cfg = yaml.safe_load(myst_path.read_text(encoding="utf-8"))
    toc = cfg.get("project", {}).get("toc", [])
    pages: list[PageEntry] = []

    def _walk(nodes: list[dict]) -> None:
        for node in nodes:
            if "file" in node:
                pages.append({"file": node["file"], "title": node.get("title", "")})
            if "children" in node:
                _walk(node["children"])

    _walk(toc)
    return pages


PAGES: list[PageEntry] = _parse_toc()

# Map normalised URL path → index for O(1) prev/next lookup
_PAGE_INDEX: dict[str, int] = {}
for _i, _p in enumerate(PAGES):
    _norm = _p["file"].replace("\\", "/")
    _PAGE_INDEX[_norm] = _i
    # Also index without extension so /path/theory resolves
    if _norm.endswith(".md"):
        _PAGE_INDEX[_norm[:-3]] = _i
    elif _norm.endswith(".ipynb"):
        _PAGE_INDEX[_norm[:-6]] = _i


def _neighbours(url_path: str) -> tuple[Optional[PageEntry], Optional[PageEntry]]:
    """Return (prev_page, next_page) for the given URL path."""
    key = url_path.lstrip("/").replace("\\", "/")
    idx = _PAGE_INDEX.get(key)
    if idx is None:
        idx = _PAGE_INDEX.get(key + ".md")
    if idx is None:
        idx = _PAGE_INDEX.get(key + ".ipynb")
    if idx is None:
        return None, None
    prev_p = PAGES[idx - 1] if idx > 0 else None
    next_p = PAGES[idx + 1] if idx < len(PAGES) - 1 else None
    return prev_p, next_p


def _page_url(entry: PageEntry) -> str:
    """Turn a toc entry into the server URL."""
    f = entry["file"].replace("\\", "/")
    if f.endswith(".md"):
        f = f[:-3]
    return "/" + f


# ── Folders / files to skip entirely in the sidebar tree ─────────────────────

SKIP_NAMES = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".ipynb_checkpoints",
    "_build",
    "static",
    "templates",
    "data",
    "models",
    "checkpoints",
    "src",
    "environment",
    "scripts",
    "notes",
}
SKIP_PREFIXES = ("cache_",)

# ── Sidebar tree builder ──────────────────────────────────────────────────────


def _label(name: str) -> str:
    """Turn a directory/file name into a readable label."""
    parts = name.split("_", 1)
    label = (
        parts[-1]
        if len(parts) == 2 and parts[0].lstrip("0123456789").replace("week", "") == ""
        else name
    )
    return label.replace("_", " ").title()


def _build_tree(directory: Path, current_url: str, depth: int = 0) -> str:
    """Recursively build the sidebar HTML tree for *directory*."""
    entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    items: list[str] = []
    for entry in entries:
        if entry.name.startswith(".") or entry.name in SKIP_NAMES:
            continue
        if any(entry.name.startswith(pf) for pf in SKIP_PREFIXES):
            continue
        if entry.is_dir():
            inner = _build_tree(entry, current_url, depth + 1)
            if not inner:
                continue
            items.append(
                f'<div class="tree-folder">'
                f'<div class="folder-label"><i class="arrow">▶</i>{_label(entry.name)}</div>'
                f'<div class="folder-children">{inner}</div>'
                f"</div>"
            )
        elif entry.suffix in (".md", ".ipynb"):
            rel = entry.relative_to(BASE_DIR).as_posix()
            url = "/" + (rel[:-3] if rel.endswith(".md") else rel)
            label = _label(entry.stem)
            if entry.suffix == ".ipynb":
                label = f"📓 {label}"
            active = "active" if url == current_url else ""
            items.append(f'<a class="tree-file {active}" href="{url}">{label}</a>')
    return "".join(items)


# ── Prev / Next navigation HTML ──────────────────────────────────────────────


def _nav_html(current_url: str) -> str:
    prev_p, next_p = _neighbours(current_url)
    if not prev_p and not next_p:
        return ""
    parts: list[str] = ['<nav class="page-nav">']
    if prev_p:
        parts.append(
            f'<a class="page-nav-prev" href="{_page_url(prev_p)}">'
            f'<span class="nav-dir">← Previous</span>'
            f'<span class="nav-title">{prev_p["title"]}</span></a>'
        )
    else:
        parts.append("<span></span>")
    if next_p:
        parts.append(
            f'<a class="page-nav-next" href="{_page_url(next_p)}">'
            f'<span class="nav-dir">Next →</span>'
            f'<span class="nav-title">{next_p["title"]}</span></a>'
        )
    else:
        parts.append("<span></span>")
    parts.append("</nav>")
    return "\n".join(parts)


# ── HTML page assembly ────────────────────────────────────────────────────────


def _page(content_html: str, current_url: str) -> str:
    """Fill the page template with sidebar, nav, and content."""
    sidebar = _build_tree(BASE_DIR, current_url)
    nav = _nav_html(current_url)
    body = content_html + nav
    return (
        _TEMPLATE.replace("__STYLE__", _STYLE)
        .replace("__SCRIPT__", _SCRIPT)
        .replace("__SIDEBAR__", sidebar)
        .replace("__CONTENT__", body)
    )


# ── Math protection (keep $…$ / $$…$$ intact through markdown) ────────────────

_DISPLAY_RE = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
_INLINE_RE = re.compile(r"(?<!\\)(?<!\$)\$(?!\$)([^\n\$]+?)(?<!\\)\$(?!\$)")


def _protect_math(text: str) -> tuple[str, dict]:
    """Replace math spans with unique placeholders before markdown processing."""
    store: dict[str, str] = {}
    counter = [0]

    def _save(expr: str) -> str:
        key = f"ZZMATH{counter[0]}ZZ"
        counter[0] += 1
        store[key] = expr
        return key

    text = _DISPLAY_RE.sub(lambda m: _save(f"$${m.group(1)}$$"), text)
    text = _INLINE_RE.sub(lambda m: _save(f"${m.group(1)}$"), text)
    return text, store


def _restore_math(html: str, store: dict) -> str:
    for key, value in store.items():
        html = html.replace(key, value)
    return html


# ── Cross-link rewriting ─────────────────────────────────────────────────────
# Theory files contain relative links like ../../03_probability/week07_likelihood/theory.md
# These need to be resolved to absolute server URLs.

_HREF_RE = re.compile(r'href="([^"]+)"')


def _rewrite_links(html: str, md_dir: Path) -> str:
    """Resolve relative .md/.ipynb links to absolute server URLs."""

    def _fix(m: re.Match) -> str:
        href = m.group(1)
        # Skip external URLs, anchors, and mailto
        if href.startswith(("http://", "https://", "//", "#", "mailto:")):
            return m.group(0)
        # Split off any fragment (#section)
        fragment = ""
        if "#" in href:
            href, fragment = href.rsplit("#", 1)
            fragment = "#" + fragment
        if not href:
            # Pure anchor link like #section
            return m.group(0)
        # Resolve relative path against the markdown file's directory
        resolved = (md_dir / href).resolve()
        try:
            rel = resolved.relative_to(BASE_DIR).as_posix()
        except ValueError:
            return m.group(0)
        # Strip .md extension for clean URLs
        if rel.endswith(".md"):
            rel = rel[:-3]
        return f'href="/{rel}{fragment}"'

    return _HREF_RE.sub(_fix, html)


# ── Markdown rendering ────────────────────────────────────────────────────────


def _render_md(md_path: Path, current_url: str) -> str:
    if not md_path.exists() or not md_path.is_file():
        raise HTTPException(status_code=404, detail=f"{md_path.relative_to(BASE_DIR)} not found")
    text = md_path.read_text(encoding="utf-8")
    text, math_store = _protect_math(text)
    body = markdown.markdown(
        text,
        extensions=["fenced_code", "tables", "toc"],
    )
    body = _restore_math(body, math_store)
    body = _rewrite_links(body, md_path.parent)
    return _page(body, current_url)


# ── Notebook rendering ────────────────────────────────────────────────────────

_nbconvert_available = False
try:
    from nbconvert import HTMLExporter
    from nbformat import read as nb_read

    _nbconvert_available = True
except ImportError:
    pass


def _render_nb(nb_path: Path, current_url: str) -> str:
    if not nb_path.exists() or not nb_path.is_file():
        raise HTTPException(status_code=404, detail=f"{nb_path.relative_to(BASE_DIR)} not found")
    if not _nbconvert_available:
        raise HTTPException(
            status_code=500,
            detail="nbconvert is not installed — run: pip install nbconvert",
        )
    with open(nb_path, encoding="utf-8") as f:
        notebook = nb_read(f, as_version=4)
    exporter = HTMLExporter(template_name="basic")
    body, _ = exporter.from_notebook_node(notebook)
    return _page(body, current_url)


# ── Full-text search index ────────────────────────────────────────────────────

_SEARCH_INDEX: list[dict] = []
# Each entry: {"file", "url", "title", "raw", "raw_lower",
#              "text", "head_offsets": list[int], "head_slugs": list[str]}

# strip markdown for snippet display (not for searching — we search raw)
_MD_STRIP_RE = re.compile(
    r"(?:"
    r"\[([^\]]*)\]\([^)]*\)"  # [text](url) → text
    r"|```[\s\S]*?```"  # fenced code blocks
    r"|`[^`]+`"  # inline code
    r"|\$\$[\s\S]*?\$\$"  # display math
    r"|\$[^$\n]+\$"  # inline math
    r"|[#*_~>|`]"  # leftover md punctuation
    r")"
)

_HEADING_RE = re.compile(r"^#{1,6}[ \t]+(.+)$", re.MULTILINE)


def _slugify(value: str) -> str:
    """Reproduce python-markdown toc slugification."""
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    return re.sub(r"[-\s]+", "-", value)


def _build_search_index() -> None:
    """Walk all theory.md files and build an anchor-aware search index."""
    _SEARCH_INDEX.clear()
    for md_path in sorted(BASE_DIR.rglob("theory.md")):
        rel = md_path.relative_to(BASE_DIR).as_posix()
        parts = rel.split("/")
        if any(p.startswith(".") or p.startswith("_") or p in SKIP_NAMES for p in parts):
            continue
        url = "/" + rel[:-3]
        raw = md_path.read_text(encoding="utf-8")
        # Heading positions: char offset of the '#' in raw text → slug
        head_offsets: list[int] = []
        head_slugs: list[str] = []
        for m in _HEADING_RE.finditer(raw):
            head_offsets.append(m.start())
            head_slugs.append(_slugify(m.group(1).strip()))
        # Title from first H1
        title_m = re.search(r"^#\s+(.+)$", raw, re.MULTILINE)
        title = title_m.group(1).strip() if title_m else rel
        # Strip markdown to plain text for human-readable snippets
        text = _MD_STRIP_RE.sub(lambda m: m.group(1) or " ", raw)
        text = re.sub(r"\s+", " ", text).strip()
        _SEARCH_INDEX.append(
            {
                "file": rel,
                "url": url,
                "title": title,
                "raw": raw,
                "raw_lower": raw.lower(),
                "text": text,
                "head_offsets": head_offsets,
                "head_slugs": head_slugs,
            }
        )


_build_search_index()


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/api/search", response_class=JSONResponse)
def api_search(q: str = Query("", min_length=0)):
    """Full-text search across all theory.md files.
    Returns results with snippets and URLs anchored to the matching section."""
    query = q.strip().lower()
    if len(query) < 2:
        return []

    results: list[dict] = []
    for doc in _SEARCH_INDEX:
        raw_lower = doc["raw_lower"]
        idx = raw_lower.find(query)
        if idx == -1:
            continue

        count = raw_lower.count(query)

        # ── Find the best heading anchor for this match ─────────────────────
        # Walk occurrences: skip any that land inside the Table of Contents
        # (heading slug == "table-of-contents") and use the next real-content hit.
        ho = doc["head_offsets"]
        hs = doc["head_slugs"]
        anchor: str = ""
        search_start = 0
        while True:
            idx = raw_lower.find(query, search_start)
            if idx == -1:
                break
            pos = bisect.bisect_right(ho, idx)
            slug = hs[pos - 1] if pos > 0 else ""
            if slug in {"table-of-contents", "tableofcontents", "toc"}:
                # skip — this occurrence is just the ToC listing
                search_start = idx + 1
                continue
            anchor = slug
            break
        url = doc["url"] + ("#" + anchor if anchor else "")

        # ── Build snippet from stripped text (easier to read) ─────────────
        # Search `text` (markdown-stripped) directly so the term is always
        # inside the window and `highlightSnippet` can wrap it in <mark>.
        text = doc["text"]
        text_lower = text.lower()
        snip_idx = -1
        snip_start = 0
        while True:
            i = text_lower.find(query, snip_start)
            if i == -1:
                break
            # Skip occurrences that are very early in the text — the
            # Table-of-Contents listing is always at the top of the page.
            toc_boundary = min(len(text) // 5, 800)
            if i < toc_boundary and snip_start == 0:
                snip_start = i + 1
                continue
            snip_idx = i
            break
        if snip_idx == -1:
            # Fallback: first occurrence anywhere in text
            snip_idx = max(0, text_lower.find(query))
        start = max(0, snip_idx - 80)
        end = min(len(text), snip_idx + len(query) + 80)
        snippet = text[start:end]
        if start > 0:
            snippet = "\u2026" + snippet
        if end < len(text):
            snippet = snippet + "\u2026"

        results.append({"url": url, "title": doc["title"], "snippet": snippet, "count": count})

    results.sort(key=lambda r: r["count"], reverse=True)
    return results[:20]


@app.get("/", response_class=HTMLResponse)
def index():
    intro = BASE_DIR / "intro.md"
    if intro.exists():
        return _render_md(intro, "/intro")
    return _render_md(BASE_DIR / "README.md", "/README")


@app.get("/{full_path:path}", response_class=HTMLResponse)
def serve(full_path: str):
    target = BASE_DIR / full_path
    current_url = "/" + full_path

    # Directory → look for theory.md, then README.md
    if target.is_dir():
        for fallback in ("theory.md", "README.md"):
            candidate = target / fallback
            if candidate.exists():
                rel = candidate.relative_to(BASE_DIR).as_posix()
                url = "/" + (rel[:-3] if rel.endswith(".md") else rel)
                return _render_md(candidate, url)
        raise HTTPException(status_code=404, detail=f"{full_path} not found")

    # Explicit .ipynb
    if target.suffix == ".ipynb" and target.exists():
        return _render_nb(target, current_url)

    # Explicit .md
    if target.suffix == ".md" and target.exists():
        return _render_md(target, current_url.removesuffix(".md"))

    # Extensionless → try .md then .ipynb
    md_alt = target.with_suffix(".md")
    if md_alt.exists():
        return _render_md(md_alt, current_url)

    nb_alt = target.with_suffix(".ipynb")
    if nb_alt.exists():
        return _render_nb(nb_alt, current_url)

    raise HTTPException(status_code=404, detail=f"{full_path} not found")
