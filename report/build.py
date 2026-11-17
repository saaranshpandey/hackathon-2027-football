"""Build the technical report from SOLUTION.md: figures, then HTML, then an A4 PDF through Microsoft Edge.

    uv run --no-project --with markdown --with matplotlib --with pandas python report/build.py

Needs submissions/T_DBC.csv (python -m possession.run --approach T_DBC --device cuda --post 50 --name T_DBC).
Writes report/fig_*.svg and out/submission/NumpyWarriors.pdf. Only the public five-minute window is plotted.
"""

import subprocess
import tempfile
from pathlib import Path

import markdown
import matplotlib

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

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LABEL_COLOURS = {"H": "#2a78d6", "A": "#eb6834", "D": "#1baf7a"}
LABEL_NAMES = {"H": "Barcelona in possession", "A": "Athletic in possession", "D": "Ball dead"}

# Possession counts in the public window, from evaluation/run_evaluation.py.
LANES = [("Ground truth", "evaluation/gt/BAR-ATH_possession_gt_public.csv", 17),
         ("Our submission", "submissions/T_DBC.csv", 18)]

plt.rcParams.update({
    "svg.fonttype": "none", "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 8.5,
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
    fig, ax = plt.subplots(figsize=(7.0, 1.6), layout="constrained")
    for row, (name, csv, possessions) in enumerate(LANES):
        y = len(LANES) - 1 - row
        csv = ROOT / csv
        if not csv.exists() and csv.name == "T_DBC.csv":
            csv = OUT / f"{TEAM}.csv"  # the upload copy of the same file
        labels = pd.read_csv(csv).label.to_numpy()[window]
        for label, colour in LABEL_COLOURS.items():
            spans = [(a / FPS / 60, (b - a) / FPS / 60) for a, b, k in runs(labels) if k == label]
            ax.broken_barh(spans, (y - 0.33, 0.66), facecolors=colour, edgecolor="white", linewidth=0.7)
        ax.text(5.06, y, f"{possessions} possessions", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(LANES)), [n for n, _, _ in LANES][::-1])
    ax.set_xlim(0, 5)
    ax.set_xlabel("Minutes since kickoff")
    ax.tick_params(length=0)
    ax.spines["bottom"].set_visible(False)
    ax.legend(handles=[Patch(color=LABEL_COLOURS[k], label=LABEL_NAMES[k]) for k in LABEL_COLOURS],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=8, handlelength=1.2)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


CSS = """
@page { size: A4; margin: 14mm 16mm 15mm 16mm; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: "Segoe UI", system-ui, sans-serif; font-size: 9.4pt; line-height: 1.45; color: #1b1b1a;
       background: #fff; margin: 0; }
h1 { font-size: 19pt; line-height: 1.15; margin: 0 0 5pt; letter-spacing: -0.01em; }
h1 + p { color: #52514e; margin: 0 0 10pt; }
h2 { font-size: 12.5pt; margin: 14pt 0 5pt; padding-top: 5pt; border-top: 1.5px solid #1b1b1a; break-after: avoid; }
h3 { font-size: 10.2pt; margin: 9pt 0 3pt; break-after: avoid; }
p, li { margin: 0 0 5pt; }
ul, ol { padding-left: 15pt; margin: 0 0 6pt; }
a { color: #2a78d6; text-decoration: none; }
code { font-family: Consolas, monospace; font-size: 8.4pt; background: #f3f3f1; padding: 0 2px; border-radius: 2px; }
pre { background: #f3f3f1; padding: 7pt 9pt; border-radius: 4px; white-space: pre-wrap; overflow-wrap: anywhere;
      break-inside: avoid; margin: 0 0 7pt; }
pre code { background: none; padding: 0; font-size: 7.5pt; }
table { width: 100%; border-collapse: collapse; margin: 3pt 0 9pt; font-size: 8.4pt; break-inside: avoid; }
th, td { text-align: left; padding: 3.5pt 6pt; border-bottom: 1px solid #e4e3df; vertical-align: top; }
th { font-weight: 600; border-bottom: 1.5px solid #52514e; }
td[align="right"], th[align="right"] { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
table.keynums { table-layout: fixed; margin: 2pt 0 12pt; }
table.keynums th { font-size: 18pt; font-weight: 700; color: #2a78d6; background: #f3f6fb; border: none;
                   padding: 8pt 9pt 1pt; line-height: 1.1; }
table.keynums td { font-size: 8.2pt; color: #52514e; background: #f3f6fb; border: none; padding: 2pt 9pt 8pt; }
table.keynums th + th, table.keynums td + td { border-left: 5px solid #fff; }
p.figure { margin: 8pt 0 3pt; break-inside: avoid; break-after: avoid; }
p.figure img { width: 100%; display: block; }
p.caption { font-size: 8.2pt; color: #52514e; margin: 0 0 9pt; }
"""


def build_html(path):
    body = markdown.markdown((ROOT / "SOLUTION.md").read_text(encoding="utf-8"), extensions=["tables", "fenced_code"])
    body = body.replace("<table>", '<table class="keynums">', 1)
    body = body.replace("<p><img", '<p class="figure"><img').replace("<p><em>Figure", '<p class="caption"><em>Figure')
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
