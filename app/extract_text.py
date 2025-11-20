import pdfplumber

def extract_text(pdf_path):
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            text += (p.extract_text() or '') + '\n'
    return text
