"""Build the publication figure and revised Ego Gate paper PDF."""

from __future__ import annotations

import csv
import html
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper" / "ego_gate_paper_source.md"
RESULTS = ROOT / "results" / "digits" / "summary.csv"
FIGURE_DIR = ROOT / "output" / "figures"
PDF_DIR = ROOT / "output" / "pdf"
FIGURE_PNG = FIGURE_DIR / "ego_gate_experimental_output.png"
FIGURE_SVG = FIGURE_DIR / "ego_gate_experimental_output.svg"
PDF_PATH = PDF_DIR / "ego_gate_paper.pdf"
ROOT_PDF_PATH = ROOT / "ego_gate_paper.pdf"


NAVY = colors.HexColor("#13233F")
TEAL = colors.HexColor("#0A7B78")
PALE = colors.HexColor("#EAF3F2")
INK = colors.HexColor("#202833")
MUTED = colors.HexColor("#5E6A78")
RULE = colors.HexColor("#CBD5DF")


def read_results() -> list[dict[str, float | str]]:
    with RESULTS.open(newline="", encoding="utf-8") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            row: dict[str, float | str] = {"condition": raw["condition"]}
            for key, value in raw.items():
                if key not in {"condition", "n"}:
                    row[key] = float(value)
            rows.append(row)
    return rows


def display_name(condition: str) -> str:
    return {
        "none": "No replay",
        "random": "Random",
        "stratified_random": "Balanced random",
        "doubt": "Doubt",
        "curiosity": "Curiosity",
        "egogate": "EgoGate",
        "embedding_kcenter": "Embedding k-center",
        "egogate_diverse": "EgoGate + diversity",
        "full": "Full memory",
    }[condition]


def build_figure(rows: list[dict[str, float | str]]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    by_name = {str(row["condition"]): row for row in rows}
    order = [
        "random",
        "stratified_random",
        "embedding_kcenter",
        "curiosity",
        "egogate_diverse",
        "egogate",
        "doubt",
        "full",
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
        }
    )
    fig = plt.figure(figsize=(12, 7.2), facecolor="#F7F8FA")
    grid = fig.add_gridspec(2, 2, height_ratios=(0.78, 2.15), hspace=0.32, wspace=0.34)

    flow = fig.add_subplot(grid[0, :])
    flow.set_xlim(0, 1)
    flow.set_ylim(0, 1)
    flow.axis("off")
    boxes = [
        (0.02, "Train Task A", "60 epochs"),
        (0.22, "Score memories", "Doubt + Curiosity"),
        (0.44, "Select K = 40", "fixed memory"),
        (0.65, "Learn Task B", "2 replay / batch"),
        (0.84, "Evaluate", "retention + plasticity"),
    ]
    for index, (x, title, subtitle) in enumerate(boxes):
        width = 0.14 if index != 1 else 0.16
        patch = FancyBboxPatch(
            (x, 0.24), width, 0.50,
            boxstyle="round,pad=0.014,rounding_size=0.025",
            facecolor="#FFFFFF" if index != 2 else "#E6F3F1",
            edgecolor="#243B5A" if index != 2 else "#0A7B78",
            linewidth=1.2,
        )
        flow.add_patch(patch)
        flow.text(x + width / 2, 0.55, title, ha="center", va="center", weight="bold", color="#13233F")
        flow.text(x + width / 2, 0.38, subtitle, ha="center", va="center", color="#5E6A78", fontsize=8)
        if index < len(boxes) - 1:
            next_x = boxes[index + 1][0]
            flow.add_patch(
                FancyArrowPatch(
                    (x + width + 0.008, 0.49), (next_x - 0.008, 0.49),
                    arrowstyle="-|>", mutation_scale=12, linewidth=1.1, color="#6B7785"
                )
            )
    flow.text(0.02, 0.91, "A  Fixed-compute experimental protocol", weight="bold", color="#13233F", fontsize=11)

    ax = fig.add_subplot(grid[1, 0])
    forgetting = np.array([float(by_name[key]["forgetting_mean"]) for key in order]) * 100
    errors = np.array([float(by_name[key]["forgetting_std"]) for key in order]) * 100
    palette = ["#AAB4C0", "#8FA5B8", "#7898A9", "#4F9F9A", "#29928D", "#0A7B78", "#164E63", "#13233F"]
    ypos = np.arange(len(order))
    ax.barh(ypos, forgetting, xerr=errors, color=palette, edgecolor="white", capsize=3, height=0.68)
    ax.set_yticks(ypos, [display_name(key) for key in order])
    ax.invert_yaxis()
    ax.set_xlabel("Forgetting (percentage points; lower is better)")
    ax.set_title("B  Old-task forgetting after Task B", loc="left")
    ax.grid(axis="x", alpha=0.22)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for y, value in zip(ypos, forgetting):
        ax.text(value + 0.8, y, f"{value:.1f}", va="center", fontsize=8, color="#263341")
    ax.text(
        0.98, 0.04, "EgoGate: 39.4% less forgetting\nthan random (Holm p = 1.41e-5)",
        transform=ax.transAxes, ha="right", va="bottom", color="#0A625F", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#E6F3F1", edgecolor="#83BDB9"),
    )

    scatter = fig.add_subplot(grid[1, 1])
    scatter_order = ["random", "stratified_random", "doubt", "curiosity", "egogate", "egogate_diverse", "full"]
    for key in scatter_order:
        row = by_name[key]
        x = float(row["task_a_post_mean"]) * 100
        y = float(row["task_b_post_mean"]) * 100
        color = "#0A7B78" if key == "egogate" else ("#13233F" if key in {"doubt", "full"} else "#8496A8")
        size = 78 if key == "egogate" else 52
        scatter.scatter(x, y, s=size, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        offsets = {
            "random": (5, -12), "stratified_random": (5, 6), "doubt": (5, -13),
            "curiosity": (5, 6), "egogate": (5, 7), "egogate_diverse": (-85, -13), "full": (-65, -13)
        }
        scatter.annotate(display_name(key), (x, y), xytext=offsets[key], textcoords="offset points", fontsize=7.8)
    scatter.set_xlim(78, 95.5)
    scatter.set_ylim(97.2, 99.5)
    scatter.set_xlabel("Task-A retention (%)")
    scatter.set_ylabel("Task-B accuracy (%)")
    scatter.set_title("C  Stability-plasticity outcome", loc="left")
    scatter.grid(alpha=0.22)
    scatter.spines[["top", "right"]].set_visible(False)
    scatter.text(0.03, 0.04, "Upper-right is preferred", transform=scatter.transAxes, fontsize=8, color="#5E6A78")

    fig.suptitle(
        "Ego Gate proof of concept: storage policy changes retention at a fixed replay budget",
        x=0.055, y=0.985, ha="left", fontsize=15, weight="bold", color="#13233F"
    )
    fig.text(0.055, 0.945, "Split-Digits, K = 40, mean +/- sample standard deviation over 20 paired seeds", color="#5E6A78", fontsize=9.5)
    fig.subplots_adjust(left=0.16, right=0.97, top=0.88, bottom=0.10)
    fig.savefig(FIGURE_PNG, dpi=300, facecolor=fig.get_facecolor())
    fig.savefig(FIGURE_SVG, facecolor=fig.get_facecolor())
    plt.close(fig)


class PaperDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="body")
        self.addPageTemplates(PageTemplate(id="paper", frames=frame, onPage=self.draw_page))

    def draw_page(self, canvas, doc):
        canvas.saveState()
        if doc.page > 1:
            canvas.setStrokeColor(RULE)
            canvas.setLineWidth(0.5)
            canvas.line(2.0 * cm, A4[1] - 1.35 * cm, A4[0] - 2.0 * cm, A4[1] - 1.35 * cm)
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(MUTED)
            canvas.drawCentredString(A4[0] / 2, 1.05 * cm, str(doc.page))
        canvas.restoreState()


def styles():
    base = getSampleStyleSheet()
    return {
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName="Times-Roman", fontSize=10.1,
            leading=14.2, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=7,
        ),
        "abstract": ParagraphStyle(
            "Abstract", parent=base["BodyText"], fontName="Times-Roman", fontSize=10.2,
            leading=14.4, textColor=INK, alignment=TA_JUSTIFY, leftIndent=0.45*cm,
            rightIndent=0.45*cm, borderColor=TEAL, borderWidth=0, borderPadding=9,
            backColor=PALE, spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=16,
            leading=19, textColor=NAVY, spaceBefore=14, spaceAfter=8, keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "H3", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=11.5,
            leading=14, textColor=TEAL, spaceBefore=10, spaceAfter=5, keepWithNext=True,
        ),
        "equation": ParagraphStyle(
            "Equation", parent=base["Code"], fontName="Courier", fontSize=8.8,
            leading=12, textColor=NAVY, leftIndent=0.45*cm, rightIndent=0.45*cm,
            borderColor=RULE, borderWidth=0.6, borderPadding=7, backColor=colors.HexColor("#F4F7FA"),
            spaceBefore=4, spaceAfter=9,
        ),
        "caption": ParagraphStyle(
            "Caption", parent=base["BodyText"], fontName="Helvetica", fontSize=8.3,
            leading=11, textColor=MUTED, alignment=TA_LEFT, spaceBefore=4, spaceAfter=10,
        ),
        "reference": ParagraphStyle(
            "Reference", parent=base["BodyText"], fontName="Times-Roman", fontSize=8.6,
            leading=11.2, textColor=INK, leftIndent=0.45*cm, firstLineIndent=-0.45*cm, spaceAfter=4,
        ),
    }


def escape_markup(text: str) -> str:
    return html.escape(text, quote=False).replace("+/-", "&plusmn;")


def results_table(rows, body_style):
    wanted = ["none", "random", "stratified_random", "doubt", "curiosity", "egogate", "embedding_kcenter", "egogate_diverse", "full"]
    by_name = {str(row["condition"]): row for row in rows}
    data = [["Condition", "Task A pre", "Task A post", "Forgetting", "Task B post"]]
    for key in wanted:
        row = by_name[key]
        data.append([
            Paragraph(display_name(key), body_style),
            f"{float(row['task_a_pre_mean']):.3f}",
            f"{float(row['task_a_post_mean']):.3f} +/- {float(row['task_a_post_std']):.3f}",
            f"{float(row['forgetting_mean']):.3f} +/- {float(row['forgetting_std']):.3f}",
            f"{float(row['task_b_post_mean']):.3f} +/- {float(row['task_b_post_std']):.3f}",
        ])
    table = Table(data, colWidths=[4.15*cm, 2.35*cm, 3.1*cm, 3.1*cm, 3.1*cm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), NAVY), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 7.6), ("LEADING", (0,0), (-1,-1), 9.3),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F3F6F8")]),
        ("GRID", (0,0), (-1,-1), 0.35, RULE), ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN", (1,1), (-1,-1), "CENTER"), ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("RIGHTPADDING", (0,0), (-1,-1), 5), ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("BACKGROUND", (0,6), (-1,6), colors.HexColor("#DDEFEA")),
    ]))
    return table


def build_story(rows):
    style = styles()
    story = []
    story.extend([
        Spacer(1, 1.6*cm),
        Paragraph("THE EGO GATE", ParagraphStyle("Kicker", fontName="Helvetica-Bold", fontSize=10, leading=12, textColor=TEAL, alignment=TA_CENTER, tracking=2)),
        Spacer(1, 0.35*cm),
        Paragraph("Stimulus Valuation and Memory Consolidation for Continuous Learning", ParagraphStyle("Title", fontName="Helvetica-Bold", fontSize=26, leading=31, textColor=NAVY, alignment=TA_CENTER, spaceAfter=16)),
        Paragraph("Aditya Singh", ParagraphStyle("Author", fontName="Helvetica", fontSize=12, leading=15, textColor=INK, alignment=TA_CENTER)),
        Spacer(1, 0.25*cm),
        Paragraph("Revised proof-of-concept study - fixed memory, fixed replay compute, 20 paired seeds", ParagraphStyle("Sub", fontName="Helvetica", fontSize=9.5, leading=13, textColor=MUTED, alignment=TA_CENTER)),
        Spacer(1, 1.15*cm),
        Image(str(FIGURE_PNG), width=16.0*cm, height=9.6*cm),
        Spacer(1, 0.25*cm),
        Paragraph("The long-term architecture routes continuous real-world stimulus into discard, active-memory, quarantine, or consolidation pathways. The current experiment validates only replay-memory selection.", style["caption"]),
        PageBreak(),
    ])

    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    in_abstract = False
    reference_mode = False
    bullets = []

    def flush_bullets():
        nonlocal bullets
        if bullets:
            items = [ListItem(Paragraph(escape_markup(item), style["body"]), leftIndent=12) for item in bullets]
            story.append(ListFlowable(items, bulletType="bullet", start="circle", leftIndent=18, bulletFontName="Helvetica", bulletFontSize=7, spaceAfter=7))
            bullets = []

    for raw in lines:
        line = raw.strip()
        if not line:
            flush_bullets()
            continue
        if line.startswith("## "):
            flush_bullets()
            title = line[3:]
            in_abstract = title == "Abstract"
            reference_mode = title == "References"
            story.append(Paragraph(escape_markup(title), style["h2"]))
        elif line.startswith("### "):
            flush_bullets()
            story.append(Paragraph(escape_markup(line[4:]), style["h3"]))
        elif line.startswith("- "):
            bullets.append(line[2:])
        elif line.startswith("EQUATION: "):
            flush_bullets()
            story.append(Paragraph(escape_markup(line[len("EQUATION: "):]), style["equation"]))
        elif line == "[[EXPERIMENT_FIGURE]]":
            flush_bullets()
            story.append(KeepTogether([
                Image(str(FIGURE_PNG), width=16.2*cm, height=9.72*cm),
                Paragraph("Figure 1. Fixed-compute protocol and experimental outcomes. Error bars show sample standard deviation across 20 paired seeds.", style["caption"]),
            ]))
        elif line == "[[RESULTS_TABLE]]":
            flush_bullets()
            story.extend([
                results_table(rows, style["body"]),
                Paragraph("Table 1. Mean +/- sample standard deviation over 20 seeds. All bounded selectors use K = 40 and identical replay exposure.", style["caption"]),
            ])
        elif reference_mode and re.match(r"^\d+\.", line):
            story.append(Paragraph(escape_markup(line), style["reference"]))
        else:
            flush_bullets()
            story.append(Paragraph(escape_markup(line), style["abstract"] if in_abstract else style["body"]))
            in_abstract = False if in_abstract else in_abstract
    flush_bullets()
    return story


def build_pdf(rows) -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    doc = PaperDocTemplate(
        str(PDF_PATH), pagesize=A4, rightMargin=2.0*cm, leftMargin=2.0*cm,
        topMargin=1.75*cm, bottomMargin=1.55*cm,
        title="The Ego Gate: Stimulus Valuation and Memory Consolidation for Continuous Learning",
        author="Aditya Singh",
        subject="Continual learning, memory selection, and offline consolidation",
    )
    doc.build(build_story(rows))
    shutil.copy2(PDF_PATH, ROOT_PDF_PATH)


def main() -> None:
    rows = read_results()
    build_figure(rows)
    build_pdf(rows)
    print(f"figure_png={FIGURE_PNG}")
    print(f"figure_svg={FIGURE_SVG}")
    print(f"pdf={PDF_PATH}")
    print(f"root_pdf={ROOT_PDF_PATH}")


if __name__ == "__main__":
    main()
