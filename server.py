"""Markdown documentation server with Ayu Mirage theme and sidebar navigation."""

import re
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import markdown

app = FastAPI()

BASE_DIR = Path(__file__).parent.resolve()

# Folders / files to skip entirely in the sidebar tree
SKIP_NAMES = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".ipynb_checkpoints",
    "cache_week01",
    "cache_week02",
    "cache_week03",
    "data",
    "models",
}

# ── Ayu Mirage colour palette ────────────────────────────────────────────────
STYLE = """
:root {
  --bg:          #1f2430;
  --panel:       #242936;
  --panel-alt:   #1a1f2e;
  --border:      #343d4a;
  --text:        #cccac2;
  --muted:       #607080;
  --accent:      #ffcc66;
  --link:        #5ccfe6;
  --green:       #bae67e;
  --orange:      #ffa759;
  --red:         #ff3333;
  --sidebar-w:   280px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body {
  height: 100%;
  font-family: 'Segoe UI', system-ui, sans-serif;
  font-size: 15px;
  background: var(--bg);
  color: var(--text);
}

/* ── Layout ── */
#layout { display: flex; height: 100vh; overflow: hidden; }

/* ── Sidebar ── */
#sidebar {
  width: var(--sidebar-w);
  min-width: var(--sidebar-w);
  background: var(--panel);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  transition: width .22s ease, min-width .22s ease;
}
/* Collapsed: shrink to just the toggle button width */
#sidebar.collapsed {
  width: 40px !important;
  min-width: 40px !important;
}
#sidebar.collapsed #sidebar-header a,
#sidebar.collapsed #sidebar-search,
#sidebar.collapsed #sidebar-tree { display: none; }
#sidebar.collapsed #sidebar-header { justify-content: center; padding: 8px 0; }

#sidebar-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 14px 10px;
  border-bottom: 1px solid var(--border);
  gap: 8px;
}
#sidebar-header a {
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--accent);
  text-decoration: none;
  letter-spacing: .02em;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
#sidebar-header a:hover { color: var(--link); }

/* ── Sidebar toggle button ── */
#sidebar-toggle {
  flex-shrink: 0;
  background: transparent;
  border: 1px solid var(--border);
  border-radius: 5px;
  color: var(--accent);
  cursor: pointer;
  width: 28px;
  height: 28px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1rem;
  line-height: 1;
  transition: background .15s, color .15s;
  user-select: none;
  padding: 0;
}
#sidebar-toggle:hover { background: var(--panel-alt); color: var(--link); }

#sidebar-search {
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
}
#sidebar-search input {
  width: 100%;
  background: var(--panel-alt);
  border: 1px solid var(--border);
  border-radius: 5px;
  color: var(--text);
  padding: 5px 10px;
  font-size: .85rem;
  outline: none;
}
#sidebar-search input::placeholder { color: var(--muted); }
#sidebar-search input:focus { border-color: var(--accent); }

#sidebar-tree {
  flex: 1;
  overflow-y: auto;
  padding: 8px 0 16px;
  scrollbar-width: thin;
  scrollbar-color: var(--border) transparent;
}
#sidebar-tree::-webkit-scrollbar { width: 5px; }
#sidebar-tree::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* tree nodes */
.tree-folder { user-select: none; }
.tree-folder > .folder-label {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 18px;
  cursor: pointer;
  color: var(--orange);
  font-size: .82rem;
  font-weight: 600;
  letter-spacing: .04em;
  text-transform: uppercase;
}
.tree-folder > .folder-label:hover { background: var(--panel-alt); }
.tree-folder > .folder-label .arrow {
  display: inline-block;
  transition: transform .18s;
  color: var(--muted);
  font-style: normal;
  font-size: .75rem;
}
.tree-folder.open > .folder-label .arrow { transform: rotate(90deg); }

.folder-children { display: none; }
.tree-folder.open > .folder-children { display: block; }

.tree-file {
  display: block;
  padding: 4px 18px 4px 32px;
  font-size: .84rem;
  color: var(--link);
  text-decoration: none;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.tree-file:hover { background: var(--panel-alt); color: var(--accent); }
.tree-file.active { background: var(--panel-alt); color: var(--accent); font-weight: 600; }

/* nested folders indent */
.folder-children .tree-folder > .folder-label { padding-left: 30px; }
.folder-children .tree-file                   { padding-left: 44px; }
.folder-children .folder-children .tree-folder > .folder-label { padding-left: 44px; }
.folder-children .folder-children .tree-file                   { padding-left: 58px; }

/* ── Main content ── */
#main {
  flex: 1;
  overflow-y: auto;
  padding: 36px 48px;
  max-width: 860px;
  scrollbar-width: thin;
  scrollbar-color: var(--border) transparent;
}
#main::-webkit-scrollbar { width: 6px; }
#main::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

#main h1 { color: var(--accent);  font-size: 1.9rem; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 1px solid var(--border); }
#main h2 { color: var(--orange);  font-size: 1.4rem; margin: 28px 0 12px; }
#main h3 { color: var(--green);   font-size: 1.15rem; margin: 22px 0 8px; }
#main h4, #main h5, #main h6 { color: var(--link); margin: 16px 0 6px; }

#main p  { line-height: 1.75; margin-bottom: 14px; }
#main ul, #main ol { padding-left: 24px; margin-bottom: 14px; line-height: 1.75; }
#main li { margin-bottom: 4px; }

#main a              { color: var(--link); }
#main a:hover        { color: var(--accent); }

#main blockquote {
  border-left: 3px solid var(--accent);
  margin: 16px 0;
  padding: 8px 18px;
  background: var(--panel);
  border-radius: 0 6px 6px 0;
  color: var(--muted);
}

#main table {
  width: 100%;
  border-collapse: collapse;
  margin: 18px 0;
  font-size: .9rem;
}
#main th {
  background: var(--panel);
  color: var(--orange);
  border: 1px solid var(--border);
  padding: 8px 14px;
  text-align: left;
}
#main td {
  border: 1px solid var(--border);
  padding: 7px 14px;
}
#main tr:nth-child(even) td { background: #222838; }

#main hr { border: none; border-top: 1px solid var(--border); margin: 24px 0; }

#main code {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 1px 6px;
  font-family: 'Cascadia Code', 'Fira Code', monospace;
  font-size: .88em;
  color: var(--green);
}

#main pre {
  background: var(--panel-alt) !important;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 18px 20px;
  overflow-x: auto;
  margin: 18px 0;
}
#main pre code {
  background: none !important;
  border: none;
  padding: 0;
  font-size: .88rem;
  line-height: 1.6;
  color: var(--text);
}

/* hidden search results */
.search-hidden { display: none !important; }

/* ── KaTeX overrides (keep math colour readable) ── */
.katex { color: var(--text); font-size: 1.05em; }
.katex-display { overflow-x: auto; padding: 6px 0; }
"""

# ── JavaScript (sidebar toggle + search filter + active link) ────────────────
SCRIPT = """
document.addEventListener('DOMContentLoaded', () => {
  const sidebar = document.getElementById('sidebar');
  const toggleBtn = document.getElementById('sidebar-toggle');
  const STORAGE_KEY = 'sidebar-collapsed';

  // Restore previous state without animation flash
  sidebar.style.transition = 'none';
  if (localStorage.getItem(STORAGE_KEY) === '1') {
    sidebar.classList.add('collapsed');
  }
  requestAnimationFrame(() => { sidebar.style.transition = ''; });

  toggleBtn.addEventListener('click', () => {
    const collapsed = sidebar.classList.toggle('collapsed');
    localStorage.setItem(STORAGE_KEY, collapsed ? '1' : '0');
  });

  // Open folders that contain the active link
  const active = document.querySelector('.tree-file.active');
  if (active) {
    let el = active.parentElement;
    while (el) {
      if (el.classList.contains('tree-folder')) { el.classList.add('open'); }
      el = el.parentElement;
    }
  } else {
    // Open top-level folders by default
    document.querySelectorAll('#sidebar-tree > .tree-folder').forEach(f => f.classList.add('open'));
  }

  // Toggle folders on click
  document.querySelectorAll('.folder-label').forEach(lbl => {
    lbl.addEventListener('click', () => {
      lbl.closest('.tree-folder').classList.toggle('open');
    });
  });

  // Live search / filter
  document.getElementById('tree-search').addEventListener('input', function () {
    const q = this.value.toLowerCase().trim();

    // First pass: show/hide individual files
    document.querySelectorAll('.tree-file').forEach(link => {
      const match = !q ||
        link.textContent.toLowerCase().includes(q) ||
        link.getAttribute('href').toLowerCase().includes(q);
      link.classList.toggle('search-hidden', !match);
    });

    // Second pass: hide folders that have no visible files under them
    // Process deepest first by reversing querySelectorAll order
    const folders = [...document.querySelectorAll('.tree-folder')].reverse();
    folders.forEach(folder => {
      if (!q) {
        folder.classList.remove('search-hidden');
        return;
      }
      folder.classList.add('open');
      const hasVisible = [...folder.querySelectorAll('.tree-file')]
        .some(f => !f.classList.contains('search-hidden'));
      folder.classList.toggle('search-hidden', !hasVisible);
    });
  });
});
"""


# ── Sidebar tree builder ──────────────────────────────────────────────────────


def _label(name: str) -> str:
    """Turn a directory name like 'week03_linear_models' into a readable label."""
    # strip leading order prefix (e.g. "01_", "week03_")
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
    items = []
    for entry in entries:
        if entry.name.startswith(".") or entry.name in SKIP_NAMES:
            continue
        if entry.is_dir():
            inner = _build_tree(entry, current_url, depth + 1)
            if not inner:
                continue
            rel = entry.relative_to(BASE_DIR).as_posix()
            items.append(
                f'<div class="tree-folder">'
                f'<div class="folder-label"><i class="arrow">▶</i>{_label(entry.name)}</div>'
                f'<div class="folder-children">{inner}</div>'
                f"</div>"
            )
        elif entry.suffix == ".md":
            rel = entry.relative_to(BASE_DIR).as_posix()
            url = "/" + rel
            label = _label(entry.stem)
            active = "active" if url == current_url else ""
            items.append(f'<a class="tree-file {active}" href="{url}">{label}</a>')
    return "".join(items)


# ── HTML shell ────────────────────────────────────────────────────────────────


def _page(content_html: str, current_url: str) -> str:
    sidebar = _build_tree(BASE_DIR, current_url)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>dafer-ai</title>
<link rel="stylesheet"
  href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/base16/monokai.min.css">
<link rel="stylesheet"
  href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
<script>
document.addEventListener('DOMContentLoaded', () => {{
  hljs.highlightAll();
  renderMathInElement(document.getElementById('main'), {{
    delimiters: [
      {{left: '$$',  right: '$$',  display: true}},
      {{left: '\\[', right: '\\]', display: true}},
      {{left: '$',   right: '$',   display: false}},
      {{left: '\\(', right: '\\)', display: false}}
    ],
    throwOnError: false
  }});
}});
</script>
<style>{STYLE}</style>
</head>
<body>
<div id="layout">
  <nav id="sidebar">
    <div id="sidebar-header">
      <a href="/">dafer-ai</a>
      <button id="sidebar-toggle" title="Toggle sidebar">☰</button>
    </div>
    <div id="sidebar-search"><input id="tree-search" type="search" placeholder="Filter files…"></div>
    <div id="sidebar-tree">{sidebar}</div>
  </nav>
  <main id="main">{content_html}</main>
</div>
<script>{SCRIPT}</script>
</body>
</html>"""


# ── Math protection (keep $…$ / $$…$$ intact through markdown) ────────────────

_DISPLAY_RE = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
# Inline: do NOT use DOTALL — inline math must not cross newlines.
# Handles \} inside expressions; stops at first unescaped $ on same line.
_INLINE_RE = re.compile(r"(?<!\\)(?<!\$)\$(?!\$)([^\n\$]+?)(?<!\\)\$(?!\$)")


def _protect_math(text: str) -> tuple[str, dict]:
    """Replace math spans with unique placeholders before markdown processing.

    Markdown can mangle underscores, asterisks and backslashes inside math
    expressions; we extract them first and put them back afterwards.
    """
    store: dict[str, str] = {}
    counter = [0]

    def _save(expr: str) -> str:
        key = f"ZZMATH{counter[0]}ZZ"
        counter[0] += 1
        store[key] = expr
        return key

    # Process display math first ($$…$$)
    def _sub_display(m: re.Match) -> str:
        return _save(f"$${m.group(1)}$$")

    def _sub_inline(m: re.Match) -> str:
        return _save(f"${m.group(1)}$")

    text = _DISPLAY_RE.sub(_sub_display, text)
    text = _INLINE_RE.sub(_sub_inline, text)
    return text, store


def _restore_math(html: str, store: dict) -> str:
    for key, value in store.items():
        html = html.replace(key, value)
    return html


# ── Markdown rendering ────────────────────────────────────────────────────────


def _render(md_path: Path, current_url: str) -> str:
    if not md_path.exists() or not md_path.is_file():
        raise HTTPException(status_code=404, detail=f"{md_path.relative_to(BASE_DIR)} not found")
    text = md_path.read_text(encoding="utf-8")
    text, math_store = _protect_math(text)
    body = markdown.markdown(
        text,
        extensions=["fenced_code", "tables", "toc"],
    )
    body = _restore_math(body, math_store)
    return _page(body, current_url)


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
def index():
    return _render(BASE_DIR / "README.md", "/README.md")


@app.get("/{full_path:path}", response_class=HTMLResponse)
def serve(full_path: str):
    target = BASE_DIR / full_path
    if target.is_dir():
        target = target / "README.md"
        full_path = full_path.rstrip("/") + "/README.md"
    if target.suffix != ".md":
        alt = target.with_suffix(".md")
        if alt.exists():
            target = alt
            full_path = full_path + ".md"
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"{full_path} not found")
    return _render(target, "/" + full_path)
