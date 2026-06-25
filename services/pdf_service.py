import os
from datetime import datetime

from reportlab.platypus import (  # type: ignore[import]
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  # type: ignore[import]
from reportlab.lib.pagesizes import A4  # type: ignore[import]
from reportlab.lib import colors  # type: ignore[import]
from reportlab.lib.units import mm  # type: ignore[import]


def generate_report(filename, result, score, authenticity_score, risk_result, ingredient_warnings, alternatives):

    doc = SimpleDocTemplate(filename, pagesize=A4,
                            rightMargin=20*mm, leftMargin=20*mm,
                            topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("ESTitle", parent=styles["Title"],
        fontSize=22, textColor=colors.HexColor("#6366f1"),
        spaceAfter=4, fontName="Helvetica-Bold")
    subtitle_style = ParagraphStyle("ESSub", parent=styles["Normal"],
        fontSize=10, textColor=colors.HexColor("#64748b"), spaceAfter=16)
    section_style = ParagraphStyle("ESSection", parent=styles["Heading2"],
        fontSize=12, textColor=colors.HexColor("#1e293b"),
        fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6)
    normal_style = ParagraphStyle("ESNormal", parent=styles["Normal"],
        fontSize=10, textColor=colors.HexColor("#374151"), spaceAfter=4, leading=15)
    bullet_style = ParagraphStyle("ESBullet", parent=normal_style,
        leftIndent=12, spaceAfter=3)
    label_style = ParagraphStyle("ESLabel", parent=styles["Normal"],
        fontSize=9, textColor=colors.HexColor("#64748b"),
        fontName="Helvetica-Bold", spaceAfter=2)
    footer_style = ParagraphStyle("ESFooter", parent=styles["Normal"],
        fontSize=8, textColor=colors.HexColor("#94a3b8"), alignment=1)

    content = []

    # Header
    content.append(Paragraph("EatSafe", title_style))
    content.append(Paragraph("AI-Powered Food Product Analysis Report", subtitle_style))
    content.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
    content.append(Spacer(1, 10))

    # Product info table
    scan_date = datetime.now().strftime("%d %B %Y, %I:%M %p")
    info_data = [
        ["Product",      result.get("product_name", "Unknown")],
        ["Brand",        result.get("brand", "Unknown")],
        ["Health Score", f"{score} / 10"],
        ["Authenticity", f"{authenticity_score:.1f}%"],
        ["Report Date",  scan_date],
    ]
    info_table = Table(info_data, colWidths=[45*mm, 120*mm])
    info_table.setStyle(TableStyle([
        ("FONTNAME",        (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",        (0, 0), (-1, -1), 10),
        ("TEXTCOLOR",       (0, 0), (0, -1), colors.HexColor("#64748b")),
        ("TEXTCOLOR",       (1, 0), (1, -1), colors.HexColor("#1e293b")),
        ("ROWBACKGROUNDS",  (0, 0), (-1, -1), [colors.HexColor("#f8fafc"), colors.white]),
        ("TOPPADDING",      (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING",   (0, 0), (-1, -1), 7),
        ("LEFTPADDING",     (0, 0), (-1, -1), 10),
        ("GRID",            (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
    ]))
    content.append(info_table)
    content.append(Spacer(1, 14))

    # Nutrition facts
    content.append(Paragraph("Nutrition Facts (per 100 g)", section_style))
    content.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0")))
    content.append(Spacer(1, 6))

    nutrition = result.get("nutrition_per_100g", {})
    nut_items = [
        ("Energy",        f"{nutrition.get('energy', 0)} kcal"),
        ("Protein",       f"{nutrition.get('protein', 0)} g"),
        ("Fat",           f"{nutrition.get('fat', 0)} g"),
        ("Saturated Fat", f"{nutrition.get('saturated_fat', 0)} g"),
        ("Sugar",         f"{nutrition.get('sugar', 0)} g"),
        ("Sodium",        f"{nutrition.get('sodium', 0)} mg"),
        ("Dietary Fiber", f"{nutrition.get('fiber', 0)} g"),
    ]
    nut_rows = []
    for i in range(0, len(nut_items), 2):
        row = []
        for j in range(2):
            if i + j < len(nut_items):
                lbl, val = nut_items[i + j]
                row.append(Paragraph(f"<b>{lbl}</b><br/>{val}", normal_style))
            else:
                row.append(Paragraph("", normal_style))
        nut_rows.append(row)

    nut_table = Table(nut_rows, colWidths=[82*mm, 82*mm])
    nut_table.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#f8fafc"), colors.white]),
        ("TOPPADDING",     (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 7),
        ("LEFTPADDING",    (0, 0), (-1, -1), 10),
        ("GRID",           (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
    ]))
    content.append(nut_table)
    content.append(Spacer(1, 14))

    # Risk factors
    content.append(Paragraph("Risk Factors", section_style))
    content.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0")))
    content.append(Spacer(1, 4))
    if risk_result.get("risks"):
        for risk in risk_result["risks"]:
            content.append(Paragraph(f"* {risk}", bullet_style))
    else:
        content.append(Paragraph("No major risks detected.", normal_style))
    content.append(Spacer(1, 10))

    # Suitable / Not Suitable side by side
    suit_data = [
        [Paragraph("<b>Suitable For</b>", label_style),
         Paragraph("<b>Not Suitable For</b>", label_style)],
        [Paragraph("<br/>".join(risk_result.get("suitable_for") or ["No specific recommendations"]), normal_style),
         Paragraph("<br/>".join(risk_result.get("not_suitable_for") or ["No restrictions detected"]), normal_style)]
    ]
    suit_table = Table(suit_data, colWidths=[82*mm, 82*mm])
    suit_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, -1), colors.HexColor("#f0fdf4")),
        ("BACKGROUND",    (1, 0), (1, -1), colors.HexColor("#fef2f2")),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
    ]))
    content.append(suit_table)
    content.append(Spacer(1, 14))

    # Ingredient warnings
    content.append(Paragraph("Ingredient Warnings", section_style))
    content.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0")))
    content.append(Spacer(1, 4))
    if ingredient_warnings:
        for warning in ingredient_warnings:
            content.append(Paragraph(f"* {warning}", bullet_style))
    else:
        content.append(Paragraph("No ingredient warnings found.", normal_style))
    content.append(Spacer(1, 10))

    # Healthier alternatives
    content.append(Paragraph("Healthier Alternatives", section_style))
    content.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0")))
    content.append(Spacer(1, 4))
    if alternatives:
        for item in alternatives:
            content.append(Paragraph(f"* {item}", bullet_style))
    else:
        content.append(Paragraph("No specific alternatives identified.", normal_style))
    content.append(Spacer(1, 16))

    # Footer
    content.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
    content.append(Spacer(1, 6))
    content.append(Paragraph(
        f"Generated by EatSafe AI  |  {scan_date}  |  For informational purposes only.",
        footer_style
    ))

    doc.build(content)
    return filename