"""The hub page's own stylesheet, head links, and site header."""

#: The site logo, colored the same way the landing page and dashboard style
#: the wordmark: the "spike" half in the shared accent color.
_WORDMARK = '<span class="wordmark-spike">spike</span>forge'

_HEAD = """
<link rel="stylesheet" href="../assets/fonts/inter.css">
<script async src="https://stats.capsize.online/capsize-count.js" data-site="spikeforge"></script>
<link rel="stylesheet" href="../assets/capsize-landing.css">
"""

#: This page's own layout and components. Only ``var(--bg)`` etc. come from
#: the linked stylesheet; nothing here depends on its selectors.
_STYLE = """
* { box-sizing: border-box; }
body { font-family: "Inter", ui-sans-serif, system-ui, sans-serif;
  max-width: 1180px; margin: 0 auto; padding: 0 20px 3rem; color: var(--text);
  background: var(--bg); line-height: 1.55; }
a { color: var(--accent); }
code { font-family: ui-monospace, "SFMono-Regular", Menlo, Consolas,
  monospace; background: var(--panel-alt); padding: 0.1rem 0.35rem;
  border-radius: 4px; font-size: 0.92em; }
.hub-header { display: flex; align-items: center; justify-content:
  space-between; flex-wrap: wrap; gap: 10px; padding-block: 18px;
  border-bottom: 1px solid var(--border); margin-bottom: 2rem; }
.hub-brand { display: inline-flex; align-items: center; gap: 8px;
  font-weight: 800; font-size: 17px; color: var(--text); text-decoration:
  none; }
.wordmark-spike { color: var(--accent); }
.hub-nav { display: flex; gap: 20px; font-size: 13px; flex-wrap: wrap; }
.hub-nav a { color: var(--muted); text-decoration: none; }
.hub-nav a:hover { color: var(--accent); }
h1 { margin: 0 0 0.25rem; font-size: 28px; letter-spacing: -0.01em; }
.sub { color: var(--muted); margin-top: 0; font-size: 14px; max-width: 68ch; }
.notice { background: var(--panel); border: 1px solid var(--border);
  border-left: 3px solid var(--accent); border-radius: 0 6px 6px 0;
  padding: 0.9rem 1.1rem; margin: 1.5rem 0; font-size: 13.5px;
  color: var(--muted); }
.notice strong { color: var(--text); }
.notice a { color: var(--accent); }
/* Eight columns do not fit a phone; the table scrolls inside its own box
   rather than making the whole page scroll sideways. */
.table-scroll { overflow-x: auto; margin-top: 1.5rem; border: 1px solid
  var(--border); border-radius: 8px; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; }
th, td { text-align: left; padding: 0.6rem 0.7rem; border-bottom: 1px solid
  var(--border); vertical-align: top; }
th { font-family: ui-monospace, monospace; font-size: 11px; letter-spacing:
  0.04em; text-transform: uppercase; color: var(--muted); background:
  var(--panel-head); }
tbody tr:hover { background: var(--panel-alt); }
tbody tr:last-child td { border-bottom: none; }
.badge { display: inline-block; padding: 0.1rem 0.55rem; border-radius: 999px;
  font-size: 0.75rem; font-weight: 600; white-space: nowrap; }
.badge-verified { background: rgba(63,185,80,0.16); color: #7ee2a8; }
.badge-unverified { background: rgba(248,81,73,0.16); color: #ff9d97; }
.badge-trained { background: rgba(165,184,255,0.18); color: var(--accent); }
.badge-untrained { background: rgba(160,177,204,0.16); color: var(--muted); }
.score { font-variant-numeric: tabular-nums; font-weight: 600; }
/* The requested credit line is often a full citation; keep it legible but
   subordinate to the licence id above it. */
.attribution { color: var(--muted); font-size: 0.8rem; line-height: 1.4;
  display: inline-block; margin-top: 0.15rem; }
.muted { color: var(--muted); }
/* The weights column is the one a visitor scans first; give it room so the
   score does not wrap one word per line. */
th:nth-child(2), td:nth-child(2) { min-width: 12rem; }
td:nth-child(2) { font-size: 0.86rem; line-height: 1.45; }
/* Training data: the citation needs width, or every row grows a tall
   one-word-per-line column. */
th:nth-child(7), td:nth-child(7) { min-width: 13rem; }
.hub-footer { margin-top: 2rem; color: var(--muted); font-size: 0.85rem; }
"""


def _header() -> str:
    """Return the site header: the wordmark and a link back to the site."""
    return f"""<header class="hub-header">
  <a class="hub-brand" href="https://spikeforge.net/">{_WORDMARK}</a>
  <nav class="hub-nav">
    <a href="https://spikeforge.net/">Home</a>
    <a href="https://docs.spikeforge.net/">Documentation</a>
    <a href="https://dash.spikeforge.net/">Dashboard</a>
    <a href="https://github.com/Capsize-Games/spikeforge">GitHub</a>
  </nav>
</header>"""
