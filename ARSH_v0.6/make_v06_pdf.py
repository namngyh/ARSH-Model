"""Render the Vietnamese v0.6 explanation to a shareable PDF (requires reportlab)."""

from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "ARSH_V06_LOGIC_V1.md"
TARGET = HERE / "ARSH_V06_LOGIC_V1.pdf"


def register_fonts() -> None:
    font_dir = Path("C:/Windows/Fonts")
    regular = font_dir / "arial.ttf"
    bold = font_dir / "arialbd.ttf"
    if not regular.is_file() or not bold.is_file():
        raise FileNotFoundError("Arial Unicode TTF fonts are needed to render Vietnamese text")
    pdfmetrics.registerFont(TTFont("ARSHArial", str(regular)))
    pdfmetrics.registerFont(TTFont("ARSHArial-Bold", str(bold)))
    pdfmetrics.registerFontFamily("ARSHArial", normal="ARSHArial", bold="ARSHArial-Bold")


def markup(value: str) -> str:
    escaped = html.escape(value)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"`([^`]+)`", r'<font color="#175a83">\1</font>', escaped)
    return escaped


def pipeline_diagram() -> Drawing:
    drawing = Drawing(490, 72)
    items = [
        ("Nến", "1 phút"), ("Lợi suất", "log"), ("Bộ", "chuẩn hóa"),
        ("HMM", "K = 7"), ("7 xác suất", "trạng thái"), ("Báo cáo", "cuối ngày"),
    ]
    box_width, gap = 73, 10
    for index, (first, second) in enumerate(items):
        x = index * (box_width + gap)
        drawing.add(Rect(x, 15, box_width, 46, 7, 7,
                              fillColor=colors.HexColor("#e8f1f7"),
                              strokeColor=colors.HexColor("#6a9dbb"), strokeWidth=0.8))
        drawing.add(String(x + box_width / 2, 42, first, textAnchor="middle",
                           fontName="ARSHArial-Bold", fontSize=7.7,
                           fillColor=colors.HexColor("#12384d")))
        drawing.add(String(x + box_width / 2, 29, second, textAnchor="middle",
                           fontName="ARSHArial", fontSize=7.4,
                           fillColor=colors.HexColor("#12384d")))
        if index < len(items) - 1:
            line_start = x + box_width + 1
            line_end = line_start + gap - 2
            drawing.add(Line(line_start, 38, line_end, 38,
                             strokeColor=colors.HexColor("#2d708f"), strokeWidth=1.4))
            drawing.add(Line(line_end - 3, 41, line_end, 38,
                             strokeColor=colors.HexColor("#2d708f"), strokeWidth=1.4))
            drawing.add(Line(line_end - 3, 35, line_end, 38,
                             strokeColor=colors.HexColor("#2d708f"), strokeWidth=1.4))
    return drawing


def footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#d0dbe2"))
    canvas.line(50, 43, A4[0] - 50, 43)
    canvas.setFont("ARSHArial", 8)
    canvas.setFillColor(colors.HexColor("#597080"))
    canvas.drawString(50, 30, "ARSH v0.6  |  Logic, mục tiêu và lộ trình lên v1")
    canvas.drawRightString(A4[0] - 50, 30, f"Trang {document.page}")
    canvas.restoreState()


def build() -> None:
    register_fonts()
    navy = colors.HexColor("#143a50")
    styles = {
        "title": ParagraphStyle("title", fontName="ARSHArial-Bold", fontSize=19,
                                leading=26, textColor=navy, spaceAfter=15, alignment=TA_LEFT),
        "h2": ParagraphStyle("h2", fontName="ARSHArial-Bold", fontSize=12.5,
                             leading=18, textColor=navy, spaceBefore=12, spaceAfter=6,
                             keepWithNext=True),
        "h3": ParagraphStyle("h3", fontName="ARSHArial-Bold", fontSize=10.6,
                             leading=15, textColor=colors.HexColor("#205b78"),
                             spaceBefore=9, spaceAfter=4, keepWithNext=True),
        "body": ParagraphStyle("body", fontName="ARSHArial", fontSize=9.5,
                               leading=13.8, textColor=colors.HexColor("#1f2d35"),
                               spaceAfter=6),
        "bullet": ParagraphStyle("bullet", fontName="ARSHArial", fontSize=9.5,
                                 leading=13.8, textColor=colors.HexColor("#1f2d35"),
                                 leftIndent=16, firstLineIndent=-11, spaceAfter=4),
    }
    story = []
    for raw_line in SOURCE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            story.append(Spacer(1, 3))
        elif line == "[[PIPELINE_DIAGRAM]]":
            story.append(Spacer(1, 5))
            story.append(pipeline_diagram())
            story.append(Spacer(1, 6))
        elif line.startswith("# "):
            story.append(Paragraph(markup(line[2:]), styles["title"]))
        elif line.startswith("## "):
            story.append(Paragraph(markup(line[3:]), styles["h2"]))
        elif line.startswith("### "):
            story.append(Paragraph(markup(line[4:]), styles["h3"]))
        elif line.startswith("- "):
            story.append(Paragraph("• " + markup(line[2:]), styles["bullet"]))
        else:
            story.append(Paragraph(markup(line), styles["body"]))
    document = SimpleDocTemplate(
        str(TARGET), pagesize=A4, leftMargin=50, rightMargin=50,
        topMargin=48, bottomMargin=57, title="ARSH v0.6 - Logic, mục tiêu và lộ trình lên v1",
        author="ARSH Project", subject="Giải thích chi tiết ARSH v0.6 và kế hoạch v1",
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Wrote {TARGET} ({TARGET.stat().st_size:,} bytes)")


if __name__ == "__main__":
    build()
