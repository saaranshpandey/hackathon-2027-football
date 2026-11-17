"""Build the technical report from SOLUTION.md: figure, then HTML, then a paper-style PDF through Microsoft Edge.

    uv run --no-project --with markdown --with latex2mathml --with matplotlib --with pandas python report/build.py

Needs submissions/T_DBC.csv (python -m possession.run --approach T_DBC --device cuda --post 50 --name T_DBC).
Writes report/fig_timeline.svg and out/submission/NumpyWarriors.pdf. Only the public five-minute window is plotted.
"""

import re
import subprocess
import tempfile
from pathlib import Path

import markdown
import matplotlib
from latex2mathml.converter import convert as to_mathml

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "report"
OUT = ROOT / "out" / "submission"
TEAM = "NumpyWarriors"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
FPS = 25

INK, INK2, GRID = "#0b0b0b", "#3a3936", "#e4e3df"
LABEL_COLOURS = {"H": "#2a78d6", "A": "#eb6834", "D": "#1baf7a"}
LABEL_NAMES = {"H": "Home (Barcelona)", "A": "Away (Athletic)", "D": "Dead ball"}

# Possession counts in the public window, from evaluation/run_evaluation.py.
LANES = [("Ground truth", "evaluation/gt/BAR-ATH_possession_gt_public.csv", 17),
         ("Ours", "submissions/T_DBC.csv", 18)]

plt.rcParams.update({
    "svg.fonttype": "none", "font.family": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
    "text.color": INK, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
})


def runs(labels):
    out, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.append((start, i, labels[start]))
            start = i
    return out


def timeline_figure(path):
    truth = pd.read_csv(ROOT / LANES[0][1])
    window = (truth.label != "X").to_numpy()
    fig, ax = plt.subplots(figsize=(6.0, 1.45), layout="constrained")
    for row, (name, csv, possessions) in enumerate(LANES):
        y = len(LANES) - 1 - row
        csv = ROOT / csv
        if not csv.exists() and csv.name == "T_DBC.csv":
            csv = OUT / f"{TEAM}.csv"  # the upload copy of the same file
        labels = pd.read_csv(csv).label.to_numpy()[window]
        for label, colour in LABEL_COLOURS.items():
            spans = [(a / FPS / 60, (b - a) / FPS / 60) for a, b, k in runs(labels) if k == label]
            ax.broken_barh(spans, (y - 0.33, 0.66), facecolors=colour, edgecolor="white", linewidth=0.7)
        ax.text(5.06, y, f"{possessions} possessions", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(LANES)), [n for n, _, _ in LANES][::-1])
    ax.set_xlim(0, 5)
    ax.set_xlabel("Minutes since kickoff")
    ax.tick_params(length=0)
    ax.spines["bottom"].set_visible(False)
    ax.legend(handles=[Patch(color=LABEL_COLOURS[k], label=LABEL_NAMES[k]) for k in LABEL_COLOURS],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=8.5, handlelength=1.2)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


CSS = """
@page { size: Letter; margin: 1in 1.25in 0.9in 1.25in; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: "Times New Roman", Times, serif; font-size: 10pt; line-height: 1.3; color: #000; background: #fff;
       margin: 0; text-align: justify; hyphens: auto; }
h1 { font-size: 17pt; line-height: 1.2; font-weight: 700; text-align: left; margin: 0 0 14pt; }
h1 + p { text-align: left; margin: 0 0 18pt; line-height: 1.35; }
h2 { font-size: 12pt; font-weight: 700; text-transform: uppercase; letter-spacing: 0.02em; text-align: left;
     margin: 15pt 0 6pt; break-after: avoid; }
h2.abstract { text-align: center; font-size: 12pt; margin: 0 0 6pt; }
p.abstract { margin: 0 0.5in 14pt; }
h3 { font-size: 10pt; font-weight: 700; margin: 10pt 0 4pt; break-after: avoid; }
p { margin: 0 0 6pt; }
ul { padding-left: 18pt; margin: 0 0 6pt; }
li { margin: 0 0 3pt; }
a { color: #0b3d91; text-decoration: none; }
code { font-family: "Courier New", Courier, monospace; font-size: 8.8pt; overflow-wrap: anywhere; }
pre { margin: 4pt 0 8pt; padding: 0; white-space: pre-wrap; overflow-wrap: anywhere; text-align: left; break-inside: avoid; }
pre code { font-size: 7.6pt; }
math { font-family: "Cambria Math", "STIX Two Math", serif; font-size: 1.04em; }
div.eq { display: flex; align-items: center; margin: 6pt 0 8pt; break-inside: avoid; }
div.eq math { flex: 1; }
div.eq span { width: 2.2em; text-align: right; }
table { border-collapse: collapse; margin: 0 auto 10pt; font-size: 9pt; break-inside: avoid; text-align: left;
        border-top: 1.2px solid #000; border-bottom: 1.2px solid #000; }
th { font-weight: 700; padding: 3pt 7pt; border-bottom: 0.7px solid #000; vertical-align: bottom; }
td { padding: 2.5pt 7pt; vertical-align: top; }
td[align="right"], th[align="right"] { text-align: right; white-space: nowrap; }
p.figure { margin: 8pt 0 4pt; text-align: center; break-inside: avoid; break-after: avoid; }
p.figure img { width: 100%; }
p.caption { margin: 0 0 10pt; }
p.caption em, p.tcaption em { font-style: normal; }
p.tcaption { margin: 8pt 0 4pt; text-align: center; break-after: avoid; }
h2.refs ~ p { text-align: left; padding-left: 1.5em; text-indent: -1.5em; margin-bottom: 4pt; font-size: 9.5pt; }
h2.appendix { break-before: page; }
h2.appendix ~ p { padding-left: 0; text-indent: 0; text-align: justify; font-size: 10pt; margin-bottom: 6pt; }
"""


def math_to_html(text):
    """Replace $$...$$ (numbered) and $...$ with MathML before Markdown can mangle them."""
    blocks, count = [], 0

    def keep(html):
        blocks.append(html)
        return f"MATHREF{len(blocks) - 1}ENDREF"

    def display(m):
        nonlocal count
        count += 1
        return "\n\n" + keep(f'<div class="eq">{to_mathml(m.group(1).strip(), display="block")}<span>({count})</span></div>') + "\n\n"

    text = re.sub(r"\$\$(.+?)\$\$", display, text, flags=re.S)
    text = re.sub(r"(?<![\\$])\$([^$\n]+?)\$", lambda m: keep(to_mathml(m.group(1))), text)
    return text, blocks


def build_html(path):
    text, blocks = math_to_html((ROOT / "SOLUTION.md").read_text(encoding="utf-8"))
    body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    body = re.sub(r"<p>MATHREF(\d+)ENDREF</p>|MATHREF(\d+)ENDREF", lambda m: blocks[int(m.group(1) or m.group(2))], body)
    body = body.replace("<h2>Abstract</h2>\n<p>", '<h2 class="abstract">Abstract</h2>\n<p class="abstract">', 1)
    body = body.replace("<h2>References</h2>", '<h2 class="refs">References</h2>')
    body = body.replace("<h2>Appendix A", '<h2 class="appendix">Appendix A')
    body = body.replace("<p><img", '<p class="figure"><img').replace("<p><em>Figure", '<p class="caption"><em>Figure')
    body = body.replace("<p><em>Table", '<p class="tcaption"><em>Table')
    body = body.replace('src="report/', 'src="' + HERE.as_uri() + "/")
    path.write_text(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{TEAM} technical report</title>'
                    f"<style>{CSS}</style></head><body>{body}</body></html>", encoding="utf-8")


def main():
    timeline_figure(HERE / "fig_timeline.svg")
    html = ROOT / "out" / "report" / f"{TEAM}.html"
    html.parent.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    build_html(html)
    pdf = OUT / f"{TEAM}.pdf"
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run([EDGE, "--headless=new", "--disable-gpu", f"--user-data-dir={profile}", "--no-pdf-header-footer",
                        f"--print-to-pdf={pdf}", html.as_uri()], check=True, capture_output=True)
    print("wrote", pdf.relative_to(ROOT))


if __name__ == "__main__":
    main()
