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

  function toggleSidebar() {
    const collapsed = sidebar.classList.toggle('collapsed');
    localStorage.setItem(STORAGE_KEY, collapsed ? '1' : '0');
  }

  toggleBtn.addEventListener('click', toggleSidebar);

  const floatBtn = document.getElementById('sidebar-float-btn');
  if (floatBtn) { floatBtn.addEventListener('click', toggleSidebar); }

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

  // ── Concept search (full-text via /api/search) ──────────────────────────
  const conceptInput = document.getElementById('concept-search');
  const resultsPane = document.getElementById('search-results');
  const treePane = document.getElementById('sidebar-tree');
  let searchTimer = null;

  function highlightSnippet(snippet, query) {
    const ql = query.toLowerCase();
    const parts = [];
    let remaining = snippet;
    while (remaining.length) {
      const idx = remaining.toLowerCase().indexOf(ql);
      if (idx === -1) { parts.push(escHtml(remaining)); break; }
      parts.push(escHtml(remaining.slice(0, idx)));
      parts.push('<mark>' + escHtml(remaining.slice(idx, idx + query.length)) + '</mark>');
      remaining = remaining.slice(idx + query.length);
    }
    return parts.join('');
  }

  function escHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  conceptInput.addEventListener('input', function () {
    clearTimeout(searchTimer);
    const q = this.value.trim();

    if (q.length < 2) {
      resultsPane.innerHTML = '';
      resultsPane.style.display = 'none';
      treePane.style.display = '';
      return;
    }

    searchTimer = setTimeout(async () => {
      try {
        const resp = await fetch('/api/search?q=' + encodeURIComponent(q));
        const data = await resp.json();

        if (!data.length) {
          resultsPane.innerHTML = '<div class="sr-empty">No results found</div>';
        } else {
          resultsPane.innerHTML = data.map(r =>
            '<a class="sr-item" href="' + r.url + '">' +
            '<span class="sr-title">' + escHtml(r.title) + '</span>' +
            '<span class="sr-count">' + r.count + ' match' + (r.count > 1 ? 'es' : '') + '</span>' +
            '<span class="sr-snippet">' + highlightSnippet(r.snippet, q) + '</span>' +
            '</a>'
          ).join('');
        }
        resultsPane.style.display = 'block';
        treePane.style.display = 'none';
      } catch (e) {
        resultsPane.innerHTML = '<div class="sr-empty">Search error</div>';
        resultsPane.style.display = 'block';
        treePane.style.display = 'none';
      }
    }, 250);
  });

  // When concept search is cleared, restore tree
  conceptInput.addEventListener('search', function () {
    if (!this.value) {
      resultsPane.innerHTML = '';
      resultsPane.style.display = 'none';
      treePane.style.display = '';
    }
  });
});
