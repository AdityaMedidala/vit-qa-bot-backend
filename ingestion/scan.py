import os
def extract_text_from_pdf(pdf_path:str)->str:
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    model_dict = create_model_dict()
    converter = PdfConverter(artifact_dict=model_dict,)
    rendered = converter(pdf_path)
    text, _, images = text_from_rendered(rendered)
    return text