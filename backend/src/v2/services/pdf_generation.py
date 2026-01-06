from fpdf import FPDF
from io import BytesIO
from config.settings import settings
from utils.text import clean_text

def generate_pdf_from_text(text: str) -> BytesIO:
    if not settings.FONT_PATH.exists():
        raise RuntimeError("Font file missing")

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)
    pdf.add_font(fname=str(settings.FONT_PATH), uni=True)
    pdf.set_font("DejaVuSans", size=11)

    pdf.add_page()
    pdf.multi_cell(0, 7, clean_text(text))

    output = BytesIO()
    output.write(pdf.output())
    output.seek(0)
    return output

def generate_merged_pdf(records: list) -> tuple[BytesIO, str]:
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)
    pdf.add_font(fname=str(settings.FONT_PATH), uni=True)
    pdf.set_font("DejaVuSans", size=11)

    for record in records:
        pdf.add_page()
        pdf.multi_cell(0, 7, clean_text(record.extracted_text))

    output = BytesIO(pdf.output())
    output.seek(0)

    filename = (
        f"clean_{records[0].filename}"
        if len(records) == 1
        else f"merged_clean_{len(records)}_docs.pdf"
    )
    return output, filename