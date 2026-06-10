"""
PDF Report Generator using ReportLab.
Exports prediction results, comparison tables, and stress-strain curves to PDF.
"""

import os
import io
import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch, cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, Image as RLImage, PageBreak)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate_report(output_path, prediction_data, curve_fig=None):
    """
    Generate a PDF report with prediction results.
    
    Args:
        output_path: path to save PDF
        prediction_data: dict with keys:
            - r0_inputs: dict of R0 input values
            - r1_inputs: dict of R1 input values (optional)
            - predicted_uts: float
            - predicted_ys: float
            - confidence_uts: float (±)
            - confidence_ys: float (±)
            - min_guaranteed_uts: float
            - min_guaranteed_ys: float
            - degradation_uts: float (%)
            - degradation_ys: float (%)
            - model_name: str
        curve_fig: matplotlib figure of stress-strain curves
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            topMargin=1.5*cm, bottomMargin=1.5*cm,
                            leftMargin=2*cm, rightMargin=2*cm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('CustomTitle', parent=styles['Title'],
                                  fontSize=18, spaceAfter=20, textColor=colors.HexColor('#1a237e'))
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Heading2'],
                                     fontSize=13, spaceAfter=10, textColor=colors.HexColor('#283593'))
    normal_style = ParagraphStyle('CustomNormal', parent=styles['Normal'],
                                   fontSize=10, spaceAfter=6)

    elements = []

    # Title
    elements.append(Paragraph("M250 Maraging Steel — R2 Weld Repair Property Prediction Report", title_style))
    elements.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    elements.append(Paragraph(f"Model Used: {prediction_data.get('model_name', 'Auto-Best')}", normal_style))
    elements.append(Spacer(1, 0.3*inch))

    # Prediction Results
    elements.append(Paragraph("Predicted R2 Properties", subtitle_style))
    pred_data = [
        ['Property', 'Predicted Value', 'Confidence (±2σ)', 'Min Guaranteed'],
        ['R2 UTS (MPa)',
         f"{prediction_data.get('pred_uts', 0):.2f}",
         f"± {prediction_data.get('conf_uts', 0):.2f}",
         f"{prediction_data.get('min_uts', 0):.2f}"],
        ['R2 Yield Strength (MPa)',
         f"{prediction_data.get('pred_ys', 0):.2f}",
         f"± {prediction_data.get('conf_ys', 0):.2f}",
         f"{prediction_data.get('min_ys', 0):.2f}"],
        ['R2 Elongation (%)',
         f"{prediction_data.get('pred_elong', 0):.2f}",
         f"± {prediction_data.get('conf_elong', 0):.2f}",
         f"{prediction_data.get('min_elong', 0):.2f}"],
        ['R2 HAZ Width (mm)',
         f"{prediction_data.get('pred_haz', 0):.2f}",
         f"± {prediction_data.get('conf_haz', 0):.2f}",
         f"{prediction_data.get('min_haz', 0):.2f}"],
        ['R2 Gauge Depth (mm)',
         f"{prediction_data.get('pred_gauge', 0):.2f}",
         f"± {prediction_data.get('conf_gauge', 0):.2f}",
         f"{prediction_data.get('min_gauge', 0):.2f}"],
        ['R2 Austenite Content (%)',
         f"{prediction_data.get('pred_austenite', 0):.2f}",
         f"± {prediction_data.get('conf_austenite', 0):.2f}",
         f"{prediction_data.get('min_austenite', 0):.2f}"],
    ]
    t = Table(pred_data, colWidths=[3.5*cm, 3.5*cm, 3.5*cm, 3.5*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a237e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#e8eaf6'), colors.white]),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.3*inch))

    # Degradation Summary
    elements.append(Paragraph("Degradation Analysis", subtitle_style))
    deg_data = [
        ['Metric', 'R0 → R2 Degradation'],
        ['UTS Degradation', f"{prediction_data.get('deg_uts', 0):.2f}%"],
        ['YS Degradation', f"{prediction_data.get('deg_ys', 0):.2f}%"],
        ['Elongation Degradation', f"{prediction_data.get('deg_elong', 0):.2f}%"],
        ['HAZ Width Degradation', f"{prediction_data.get('deg_haz', 0):.2f}%"],
        ['Gauge Depth Degradation', f"{prediction_data.get('deg_gauge', 0):.2f}%"],
        ['Austenite Content Change', f"{prediction_data.get('deg_austenite', 0):.2f}%"],
    ]
    t2 = Table(deg_data, colWidths=[5*cm, 5*cm])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e65100')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    elements.append(t2)
    elements.append(Spacer(1, 0.3*inch))

    # Input Properties
    elements.append(Paragraph("Input R0 Properties", subtitle_style))
    r0 = prediction_data.get('r0_inputs', {})
    input_data = [['Property', 'Value']]
    for key, val in r0.items():
        input_data.append([key, f"{val:.2f}" if isinstance(val, (int, float)) else str(val)])
    if input_data:
        t3 = Table(input_data, colWidths=[7*cm, 5*cm])
        t3.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e7d32')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#e8f5e9'), colors.white]),
        ]))
        elements.append(t3)

    # Stress-Strain Curve
    if curve_fig is not None:
        elements.append(PageBreak())
        elements.append(Paragraph("Stress-Strain Curves", subtitle_style))
        buf = io.BytesIO()
        curve_fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img = RLImage(buf, width=16*cm, height=10*cm)
        elements.append(img)

    # Footer
    elements.append(Spacer(1, 0.5*inch))
    footer_style = ParagraphStyle('Footer', parent=styles['Normal'],
                                   fontSize=8, textColor=colors.grey, alignment=TA_CENTER)
    elements.append(Paragraph(
        "M250 Maraging Steel Weld Repair Property Predictor — "
        "Vikram Sarabhai Space Centre (VSSC/ISRO)", footer_style))

    doc.build(elements)
    return output_path
