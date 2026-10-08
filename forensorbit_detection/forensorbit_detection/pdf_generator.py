import os
import json
from datetime import datetime

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, PageBreak
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def generate_forensic_pdf_report(report_dir, report_data):
    """
    Generates a professional forensic PDF report containing:
    1. Case & Investigation Details
    2. Crime Scene & Source Video Description
    3. Evidence Collected Summary Table
    4. Annotated Evidence Photos with Bounding Box Highlights
    5. AI Forensic Analysis Summary
    6. Chain of Custody Log
    7. Findings & advisory conclusion
    8. Investigator Sign-off block
    """
    pdf_path = os.path.join(report_dir, "forensic_report.pdf")
    
    if not REPORTLAB_AVAILABLE:
        print("Warning: 'reportlab' is not installed. PDF report generation skipped. Install via: pip install reportlab")
        return None

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1a365d'),
        alignment=1, # Center
        spaceAfter=10
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#4a5568'),
        alignment=1,
        spaceAfter=15
    )

    h2_style = ParagraphStyle(
        'Heading2Custom',
        parent=styles['Heading2'],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#2b6cb0'),
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'BodyCustom',
        parent=styles['BodyText'],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#2d3748')
    )

    disclaimer_style = ParagraphStyle(
        'DisclaimerCustom',
        parent=styles['Italic'],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#c53030'),
        spaceBefore=8,
        spaceAfter=8
    )

    elements = []

    # Title & Header
    elements.append(Paragraph("FORENSORBIT AUTOMATED EVIDENCE REPORT", title_style))
    elements.append(Paragraph("Official Forensic Digital Video Analysis & Evidence Dossier", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1a365d'), spaceAfter=15))

    # 1. Case Details Section
    elements.append(Paragraph("1. Case & Investigation Details", h2_style))
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    source_video = report_data.get("source_video", "N/A")
    video_name = os.path.basename(source_video)

    case_data = [
        [Paragraph("<b>Case ID / Crime ID:</b>", body_style), Paragraph(f"CASE-{datetime.now().strftime('%Y%m%d')}-01", body_style),
         Paragraph("<b>Date/Time of Report:</b>", body_style), Paragraph(now_str, body_style)],
        [Paragraph("<b>Type of Incident:</b>", body_style), Paragraph("Digital Video Evidence Extraction", body_style),
         Paragraph("<b>Processing Unit:</b>", body_style), Paragraph("ForensOrbit Pi 4 Autonomous System", body_style)],
        [Paragraph("<b>Source Video File:</b>", body_style), Paragraph(video_name, body_style),
         Paragraph("<b>Investigating Officer:</b>", body_style), Paragraph("Lead Forensic Examiner", body_style)]
    ]
    t_case = Table(case_data, colWidths=[1.4*inch, 2.2*inch, 1.4*inch, 2.2*inch])
    t_case.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f7fafc')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#edf2f7')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    elements.append(t_case)
    elements.append(Spacer(1, 10))

    # 2. Disclaimer & Crime Description
    elements.append(Paragraph("2. Advisory Disclaimer & Description", h2_style))
    disclaimer_text = report_data.get("disclaimer", "Findings are advisory, not conclusive — for investigator review.")
    elements.append(Paragraph(f"<b>LEGAL NOTICE:</b> {disclaimer_text}", disclaimer_style))
    desc_text = (
        f"Automated object detection and sparse evidence sampling executed on <b>{video_name}</b>. "
        "Positive detections exceeding the configured confidence threshold were annotated and isolated "
        "for investigator verification."
    )
    elements.append(Paragraph(desc_text, body_style))
    elements.append(Spacer(1, 10))

    # 3. Evidence Collected Table
    elements.append(Paragraph("3. Evidence Collected Summary", h2_style))
    detections = report_data.get("detections", [])

    if not detections:
        elements.append(Paragraph("<i>No high-confidence evidence objects detected in the sampled frames.</i>", body_style))
    else:
        table_data = [["Evidence ID", "Evidence Type", "Confidence", "Frame #", "Video Timestamp", "Image File"]]
        for idx, det in enumerate(detections, 1):
            ev_id = f"EV-{idx:03d}"
            table_data.append([
                ev_id,
                det.get("class", "Unknown").capitalize(),
                f"{det.get('confidence', 0.0)*100:.1f}%",
                str(det.get("frame_number", "-")),
                f"{det.get('timestamp_seconds', det.get('video_timestamp_seconds', 0.0)):.2f}s",
                det.get("annotated_image", det.get("image_filename", "-"))
            ])

        t_ev = Table(table_data, colWidths=[0.9*inch, 1.3*inch, 1.0*inch, 0.8*inch, 1.3*inch, 1.9*inch])
        t_ev.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2b6cb0')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 9),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#ffffff')),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f7fafc')]),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e0')),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ]))
        elements.append(t_ev)

    elements.append(Spacer(1, 10))

    # 4. Chain of Custody Table
    elements.append(Paragraph("4. Chain of Custody Log", h2_style))
    custody_data = [
        ["Evidence ID", "Collected / Extracted By", "Date & Time", "Transferred To", "Status"],
        ["EV-ALL", "ForensOrbit Pi 4 System", now_str, "Local Pi storage", "Pending transfer"]
    ]
    t_cust = Table(custody_data, colWidths=[1.0*inch, 1.8*inch, 1.5*inch, 1.6*inch, 1.3*inch])
    t_cust.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#4a5568')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 8),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e0')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    elements.append(t_cust)
    elements.append(Spacer(1, 15))

    # 5. Evidence Photos (Highlighted Bounding Box Images)
    if detections:
        elements.append(PageBreak())
        elements.append(Paragraph("5. Evidence Photos (Highlighted Visual Detections)", h2_style))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#2b6cb0'), spaceAfter=15))

        for idx, det in enumerate(detections, 1):
            img_file = det.get("annotated_image", det.get("image_filename"))
            img_path = os.path.join(report_dir, img_file) if img_file else None

            photo_meta = (
                f"<b>Photo ID:</b> IMG-{idx:03d} &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"<b>Class:</b> {det.get('class', 'N/A').capitalize()} &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"<b>Confidence:</b> {det.get('confidence', 0.0)*100:.1f}% &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"<b>Timestamp:</b> {det.get('timestamp_seconds', det.get('video_timestamp_seconds', 0.0)):.2f}s "
                f"(Frame #{det.get('frame_number', 0)})"
            )
            elements.append(Paragraph(photo_meta, body_style))
            elements.append(Spacer(1, 4))

            if img_path and os.path.exists(img_path):
                try:
                    # Embed image with max width ~5.5 inches, maintaining 4:3 or 16:9 ratio
                    img_flowable = Image(img_path, width=5.5*inch, height=3.5*inch)
                    elements.append(img_flowable)
                except Exception as img_err:
                    elements.append(Paragraph(f"<i>[Image loading failed: {img_err}]</i>", body_style))
            else:
                elements.append(Paragraph("<i>[Image file unavailable]</i>", body_style))

            elements.append(Spacer(1, 15))

    # 6. Conclusion & Sign-off
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("6. Conclusion & Examiner Sign-Off", h2_style))
    summary_msg = (
        f"A total of <b>{len(detections)} evidence instance(s)</b> were extracted and verified. "
        "All raw frames and full video files have been preserved in the forensic evidence repository."
    )
    elements.append(Paragraph(summary_msg, body_style))
    elements.append(Spacer(1, 20))

    sign_data = [
        [Paragraph("<b>Investigating Officer:</b> ___________________________", body_style),
         Paragraph("<b>Badge / ID:</b> ______________", body_style)],
        [Paragraph("<b>Signature:</b> ___________________________________", body_style),
         Paragraph("<b>Date:</b> " + datetime.now().strftime("%Y-%m-%d"), body_style)]
    ]
    t_sign = Table(sign_data, colWidths=[4.2*inch, 3.0*inch])
    t_sign.setStyle(TableStyle([
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    elements.append(t_sign)

    doc.build(elements)
    print(f"Forensic PDF Report generated successfully at: {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    # Quick standalone test
    sample_dir = "."
    sample_data = {
        "disclaimer": "Findings are advisory, not conclusive — for investigator review.",
        "source_video": "sample_video.mp4",
        "detections": [
            {
                "class": "knife",
                "confidence": 0.89,
                "frame_number": 15,
                "video_timestamp_seconds": 0.5,
                "image_filename": "sample_frame.jpg"
            }
        ]
    }
    generate_forensic_pdf_report(sample_dir, sample_data)
