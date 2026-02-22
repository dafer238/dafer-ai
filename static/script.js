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
