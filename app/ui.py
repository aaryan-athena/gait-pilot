"""Visual design layer: theme CSS and the small components every page uses.

Kept apart from the page logic so the look can change without touching what a
page does. Components return nothing and render straight into the page; every
piece of user-supplied text goes through :func:`esc` first, because these are
written as raw HTML.

Colour carries meaning here (green good, amber watch, red attention), so every
coloured element also carries a word or a symbol: the meaning has to survive a
colour-blind reader and a greyscale printout.
"""
from __future__ import annotations

import html
from typing import Iterable, Optional, Sequence

import streamlit as st

#: Page names, in sidebar order. The first is the landing page.
PAGES = ["Trends", "Analyse a walk", "Camera setup", "Limitations"]

PAGE_ICONS = {
    "Trends": ":material/monitoring:",
    "Analyse a walk": ":material/directions_walk:",
    "Camera setup": ":material/straighten:",
    "Limitations": ":material/info:",
}

NAV_KEY = "nav_page"

#: tone -> (css class, symbol). Symbol repeats the colour's meaning in a shape.
TONES = {
    "good": ("good", "✓"),
    "watch": ("watch", "●"),
    "attention": ("attention", "▲"),
    "insufficient": ("neutral", "–"),
    "neutral": ("neutral", "–"),
    "info": ("info", "i"),
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root {
  --cc-primary: #0B6E63; --cc-primary-dark: #08564D; --cc-primary-soft: #E5F2EF;
  --cc-ink: #13232B; --cc-muted: #5B6B73; --cc-faint: #8A989F;
  --cc-border: #E3E8EA; --cc-surface: #FFFFFF; --cc-bg: #F5F7F7;
  --cc-good: #1E7A4C; --cc-good-soft: #E7F4EC;
  --cc-watch: #A86207; --cc-watch-soft: #FDF3E1;
  --cc-attention: #B42318; --cc-attention-soft: #FDECEA;
  --cc-info: #1D5C8C; --cc-info-soft: #E8F1F8;
  --cc-neutral-soft: #EEF1F2;
  --cc-radius: 14px;
  --cc-shadow: 0 1px 2px rgba(19,35,43,.04), 0 4px 16px rgba(19,35,43,.05);
}
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select {
  font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
}
/* The header is kept, not hidden: it holds the button that reopens a
   collapsed sidebar, and hiding the header left no way back to the menu.
   It is made transparent and click-through instead, with only the clutter in
   it (deploy, main menu, the running indicator) removed. */
[data-testid="stStatusWidget"], [data-testid="stDecoration"],
[data-testid="stToolbarActions"], [data-testid="stAppDeployButton"],
[data-testid="stMainMenu"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; pointer-events: none; }
header[data-testid="stHeader"] [data-testid="stExpandSidebarButton"] { pointer-events: auto; }
[data-testid="stExpandSidebarButton"] {
  background: var(--cc-surface) !important; border: 1px solid var(--cc-border) !important;
  border-radius: 10px !important; box-shadow: var(--cc-shadow); color: var(--cc-primary) !important;
}
.main .block-container, [data-testid="stMainBlockContainer"] {
  padding-top: 1.6rem !important; padding-bottom: 72px !important; max-width: 1180px;
}
h1, h2, h3, h4 { color: var(--cc-ink); letter-spacing: -0.01em; }

/* ---- sidebar ------------------------------------------------------------ */
section[data-testid="stSidebar"] { border-right: 1px solid var(--cc-border); }
section[data-testid="stSidebar"] > div { padding-top: 0.6rem !important; }
.cc-brand { display:flex; align-items:center; gap:.7rem; padding:.4rem .2rem 1rem; }
.cc-brand-mark { width:40px; height:40px; border-radius:12px; display:flex;
  align-items:center; justify-content:center; font-size:1.25rem; color:#fff;
  background: linear-gradient(135deg, var(--cc-primary), #14A38F); box-shadow: var(--cc-shadow); }
.cc-brand-name { font-weight:800; font-size:1.12rem; color:var(--cc-ink); line-height:1.1; }
.cc-brand-tag { font-size:.74rem; color:var(--cc-muted); }
.cc-nav-label { font-size:.7rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase;
  color: var(--cc-faint); margin: .2rem 0 .3rem .2rem; }
section[data-testid="stSidebar"] [role="radiogroup"] { gap: 2px; width: 100%; }
section[data-testid="stSidebar"] [data-testid="stRadioOption"] {
  display: flex; width: 100%; padding: .55rem .75rem; border-radius: 10px; margin: 0;
  transition: background .12s ease; cursor: pointer;
}
section[data-testid="stSidebar"] [data-testid="stRadioOption"]:hover { background: var(--cc-neutral-soft); }
section[data-testid="stSidebar"] [data-testid="stRadioOption"] div:has(> [data-testid="stMarkdownContainer"])
  > div:not([data-testid="stMarkdownContainer"]) { display: none; }
section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {
  background: var(--cc-primary-soft); box-shadow: inset 3px 0 0 var(--cc-primary);
}
section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] p {
  color: var(--cc-primary-dark); font-weight: 650;
}
section[data-testid="stSidebar"] [data-testid="stRadioOption"] p { font-size: .95rem; color: var(--cc-ink); }
section[data-testid="stSidebar"] [data-testid="stRadioOption"] [role="img"] { margin-right: .35rem; color: var(--cc-muted); }
section[data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] [role="img"] { color: var(--cc-primary); }

/* ---- hero --------------------------------------------------------------- */
.cc-hero { border-radius: 18px; padding: 1.5rem 1.7rem; margin-bottom: 1rem; color: #fff;
  background: radial-gradient(120% 140% at 100% 0%, #18A58F 0%, var(--cc-primary) 45%, #074C45 100%);
  box-shadow: 0 10px 30px rgba(11,110,99,.18); position: relative; overflow: hidden; }
.cc-hero:after { content:""; position:absolute; right:-60px; top:-60px; width:220px; height:220px;
  border-radius:50%; background: rgba(255,255,255,.07); }
.cc-hero-eyebrow { font-size:.74rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase;
  opacity:.8; margin-bottom:.35rem; }
.cc-hero-title { font-size:1.85rem; font-weight:800; line-height:1.15; margin:0; color:#fff; }
.cc-hero-sub { font-size:.98rem; opacity:.9; margin-top:.45rem; max-width: 720px; line-height:1.5; }

/* ---- cards and tiles ---------------------------------------------------- */
.cc-tiles { display:grid; grid-template-columns: repeat(auto-fit, minmax(148px, 1fr)); gap:.8rem; margin:.3rem 0 1rem; }
.cc-tile { background: var(--cc-surface); border:1px solid var(--cc-border); border-radius: var(--cc-radius);
  padding: .95rem 1.05rem; box-shadow: var(--cc-shadow); min-height: 104px; }
.cc-tile-label { font-size:.76rem; font-weight:600; color: var(--cc-muted); text-transform: uppercase; letter-spacing:.05em; }
.cc-tile-value { font-size:1.65rem; font-weight:750; color: var(--cc-ink); margin-top:.25rem; line-height:1.15; }
.cc-tile-value small { font-size:.85rem; font-weight:600; color: var(--cc-muted); margin-left:.2rem; }
.cc-tile-sub { font-size:.8rem; color: var(--cc-muted); margin-top:.35rem; line-height:1.35; }
.cc-tile.good { border-top: 3px solid var(--cc-good); }
.cc-tile.watch { border-top: 3px solid var(--cc-watch); }
.cc-tile.attention { border-top: 3px solid var(--cc-attention); }
.cc-tile.accent { border-top: 3px solid var(--cc-primary); }

.cc-pill { display:inline-flex; align-items:center; gap:.3rem; font-size:.74rem; font-weight:650;
  padding:.18rem .55rem; border-radius:999px; white-space:nowrap; }
.cc-pill.good { background: var(--cc-good-soft); color: var(--cc-good); }
.cc-pill.watch { background: var(--cc-watch-soft); color: var(--cc-watch); }
.cc-pill.attention { background: var(--cc-attention-soft); color: var(--cc-attention); }
.cc-pill.neutral { background: var(--cc-neutral-soft); color: var(--cc-muted); }
.cc-pill.info { background: var(--cc-info-soft); color: var(--cc-info); }

.cc-banner { border-radius: var(--cc-radius); padding: 1.05rem 1.25rem; margin: .2rem 0 1rem;
  border: 1px solid var(--cc-border); display:flex; gap: .9rem; align-items:flex-start; background: var(--cc-surface); }
.cc-banner-icon { flex: 0 0 34px; height:34px; border-radius:10px; display:flex; align-items:center;
  justify-content:center; font-weight:800; font-size:1rem; }
.cc-banner-title { font-weight:750; font-size:1.08rem; color: var(--cc-ink); }
.cc-banner-body { font-size:.9rem; color: var(--cc-muted); margin-top:.2rem; line-height:1.5; }
.cc-banner.good { background: linear-gradient(0deg, var(--cc-good-soft), var(--cc-good-soft)); border-color:#CDE8D7; }
.cc-banner.good .cc-banner-icon { background: var(--cc-good); color:#fff; }
.cc-banner.watch { background: var(--cc-watch-soft); border-color:#F3DDB4; }
.cc-banner.watch .cc-banner-icon { background: var(--cc-watch); color:#fff; }
.cc-banner.attention { background: var(--cc-attention-soft); border-color:#F6C9C4; }
.cc-banner.attention .cc-banner-icon { background: var(--cc-attention); color:#fff; }
.cc-banner.neutral, .cc-banner.info { background: var(--cc-info-soft); border-color:#CFE0EE; }
.cc-banner.neutral .cc-banner-icon, .cc-banner.info .cc-banner-icon { background: var(--cc-info); color:#fff; }

.cc-section { display:flex; align-items:baseline; justify-content:space-between; margin: 1.4rem 0 .5rem; }
.cc-section-title { font-size:1.15rem; font-weight:750; color: var(--cc-ink); }
.cc-section-caption { font-size:.85rem; color: var(--cc-muted); margin-top:.1rem; }

.cc-step { display:flex; align-items:center; gap:.65rem; margin-bottom:.35rem; }
.cc-step-num { width:28px; height:28px; border-radius:50%; display:flex; align-items:center;
  justify-content:center; font-weight:750; font-size:.85rem; background: var(--cc-neutral-soft); color: var(--cc-muted); }
.cc-step.done .cc-step-num { background: var(--cc-good); color:#fff; }
.cc-step.current .cc-step-num { background: var(--cc-primary); color:#fff; }
.cc-step-title { font-weight:700; font-size:1.02rem; color: var(--cc-ink); }
.cc-step-hint { font-size:.84rem; color: var(--cc-muted); margin: 0 0 .6rem 2.4rem; }

.cc-concern { background: var(--cc-surface); border:1px solid var(--cc-border); border-left-width: 4px;
  border-radius: 12px; padding: .85rem 1rem; margin-bottom: .6rem; }
.cc-concern.attention { border-left-color: var(--cc-attention); }
.cc-concern.watch { border-left-color: var(--cc-watch); }
.cc-concern.muted { border-left-color: var(--cc-border); opacity: .85; }
.cc-concern-head { display:flex; align-items:center; justify-content:space-between; gap:.6rem; flex-wrap: wrap; }
.cc-concern-name { font-weight:700; color: var(--cc-ink); }
.cc-concern-msg { font-size:.88rem; color: var(--cc-ink); margin-top:.35rem; line-height:1.45; }
.cc-concern-meta { font-size:.78rem; color: var(--cc-muted); margin-top:.35rem; }

.cc-empty { text-align:center; padding: 2.2rem 1.5rem; border:1px dashed #C9D3D6; border-radius: 18px;
  background: var(--cc-surface); }
.cc-empty-title { font-size:1.2rem; font-weight:750; color: var(--cc-ink); }
.cc-empty-body { font-size:.92rem; color: var(--cc-muted); margin-top:.35rem; }

.cc-kv { display:flex; justify-content:space-between; font-size:.88rem; padding:.35rem 0;
  border-bottom: 1px solid var(--cc-border); }
.cc-kv span:first-child { color: var(--cc-muted); }
.cc-kv span:last-child { font-weight: 650; color: var(--cc-ink); }

/* ---- native widgets ----------------------------------------------------- */
[data-testid="stFileUploaderDropzone"] { border: 2px dashed #9CC9C1 !important; border-radius: 14px !important;
  background: #FBFDFC !important; padding: 1.6rem 1.4rem !important; }
[data-testid="stFileUploaderDropzone"] button { background: var(--cc-primary) !important; color:#fff !important;
  border:none !important; border-radius: 9px !important; font-weight: 600 !important; }
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button { border-radius: 10px; font-weight: 600; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: var(--cc-radius); }
[data-testid="stExpander"] details { border-radius: 12px; background: var(--cc-surface); }
[data-testid="stTabs"] button p { font-weight: 600; }

.sticky-footer { position: fixed; bottom: 0; left: 0; right: 0; background: rgba(255,255,255,.94);
  backdrop-filter: blur(6px); border-top: 1px solid var(--cc-border); padding: 8px 24px;
  font-size: .76rem; color: var(--cc-muted); z-index: 999; text-align: center; }
</style>
"""


def esc(text) -> str:
    """Escape anything that came from data or a person before it goes into HTML."""
    return html.escape(str(text), quote=True)


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def _html(markup: str) -> None:
    # One line: markdown treats indented lines as code, and st.html does not
    # need the whitespace.
    st.html(" ".join(line.strip() for line in markup.splitlines()))


# --------------------------------------------------------------------------
# navigation
# --------------------------------------------------------------------------
def go_to(page: str) -> None:
    """Button callback: switch page on the next run."""
    st.session_state[NAV_KEY] = page


def nav_button(label: str, page: str, *, key: str, primary: bool = False,
               icon: Optional[str] = None, disabled: bool = False) -> None:
    st.button(label, key=key, on_click=go_to, args=(page,), icon=icon,
              type="primary" if primary else "secondary", disabled=disabled)


def brand() -> None:
    _html(
        '<div class="cc-brand"><div class="cc-brand-mark">🚶</div>'
        '<div><div class="cc-brand-name">CadenceCare</div>'
        '<div class="cc-brand-tag">Gait screening &amp; trends</div></div></div>'
    )


def nav_label(text: str) -> None:
    _html(f'<div class="cc-nav-label">{esc(text)}</div>')


# --------------------------------------------------------------------------
# page furniture
# --------------------------------------------------------------------------
def hero(title: str, subtitle: str = "", eyebrow: str = "CadenceCare") -> None:
    _html(
        f'<div class="cc-hero"><div class="cc-hero-eyebrow">{esc(eyebrow)}</div>'
        f'<div class="cc-hero-title">{esc(title)}</div>'
        + (f'<div class="cc-hero-sub">{esc(subtitle)}</div>' if subtitle else "")
        + "</div>"
    )


def section(title: str, caption: str = "") -> None:
    _html(
        f'<div class="cc-section"><div><div class="cc-section-title">{esc(title)}</div>'
        + (f'<div class="cc-section-caption">{esc(caption)}</div>' if caption else "")
        + "</div></div>"
    )


def step(number: int, title: str, *, state: str = "current", hint: str = "") -> None:
    """A numbered step header; ``state`` is done, current or todo."""
    mark = "✓" if state == "done" else str(number)
    _html(
        f'<div class="cc-step {esc(state)}"><div class="cc-step-num">{mark}</div>'
        f'<div class="cc-step-title">{esc(title)}</div></div>'
        + (f'<div class="cc-step-hint">{esc(hint)}</div>' if hint else "")
    )


def pill(text: str, tone: str = "neutral") -> str:
    css, symbol = TONES.get(tone, TONES["neutral"])
    return f'<span class="cc-pill {css}">{symbol} {esc(text)}</span>'


def banner(title: str, body: str = "", tone: str = "info") -> None:
    css, symbol = TONES.get(tone, TONES["info"])
    _html(
        f'<div class="cc-banner {css}"><div class="cc-banner-icon">{symbol}</div>'
        f'<div><div class="cc-banner-title">{esc(title)}</div>'
        + (f'<div class="cc-banner-body">{esc(body)}</div>' if body else "")
        + "</div></div>"
    )


def tile(label: str, value: str, *, unit: str = "", sub: str = "",
         tone: str = "", sub_html: str = "") -> str:
    """One stat tile, as HTML for :func:`tiles`."""
    unit_html = f"<small>{esc(unit)}</small>" if unit else ""
    sub_part = sub_html or (esc(sub) if sub else "")
    return (
        f'<div class="cc-tile {esc(tone)}"><div class="cc-tile-label">{esc(label)}</div>'
        f'<div class="cc-tile-value">{esc(value)}{unit_html}</div>'
        + (f'<div class="cc-tile-sub">{sub_part}</div>' if sub_part else "")
        + "</div>"
    )


def tiles(items: Iterable[str]) -> None:
    _html('<div class="cc-tiles">' + "".join(items) + "</div>")


def concern_card(name: str, message: str, meta: str, *, tone: str,
                 badges: Sequence[str] = ()) -> None:
    _html(
        f'<div class="cc-concern {esc(tone)}"><div class="cc-concern-head">'
        f'<span class="cc-concern-name">{esc(name)}</span>'
        f'<span>{" ".join(badges)}</span></div>'
        f'<div class="cc-concern-msg">{esc(message)}</div>'
        f'<div class="cc-concern-meta">{esc(meta)}</div></div>'
    )


def empty_state(title: str, body: str) -> None:
    _html(
        f'<div class="cc-empty"><div style="font-size:2.2rem">🚶</div>'
        f'<div class="cc-empty-title">{esc(title)}</div>'
        f'<div class="cc-empty-body">{esc(body)}</div></div>'
    )


def key_values(rows: Sequence[tuple[str, str]]) -> None:
    _html("".join(
        f'<div class="cc-kv"><span>{esc(k)}</span><span>{esc(v)}</span></div>'
        for k, v in rows
    ))
