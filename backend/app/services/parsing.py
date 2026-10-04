import io
import fitz  # PyMuPDF
from docx import Document
from typing import List, Tuple, Optional

class DocumentParsingError(Exception):
    pass

class ParsedPage:
    def __init__(self, text: str, page_num: Optional[int] = None):
        self.text = text
        self.page_num = page_num

def parse_pdf(file_content: bytes) -> List[ParsedPage]:
    pages = []
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        for i in range(len(doc)):
            page = doc[i]
            text = page.get_text("text")
            if text:
                pages.append(ParsedPage(text=text, page_num=i + 1))
        doc.close()
    except Exception as e:
        raise DocumentParsingError(f"Failed to parse PDF: {str(e)}")
    
    return pages

def parse_docx(file_content: bytes) -> List[ParsedPage]:
    try:
        doc = Document(io.BytesIO(file_content))
        text = "\n".join([para.text for para in doc.paragraphs])
        return [ParsedPage(text=text, page_num=None)] if text.strip() else []
    except Exception as e:
        raise DocumentParsingError(f"Failed to parse DOCX: {str(e)}")

def parse_txt(file_content: bytes) -> List[ParsedPage]:
    try:
        text = file_content.decode('utf-8')
        return [ParsedPage(text=text, page_num=None)] if text.strip() else []
    except UnicodeDecodeError:
        try:
            # Fallback to Latin-1 or ignore errors if utf-8 fails
            text = file_content.decode('utf-8', errors='ignore')
            return [ParsedPage(text=text, page_num=None)] if text.strip() else []
        except Exception as e:
            raise DocumentParsingError(f"Failed to parse TXT: {str(e)}")
    except Exception as e:
        raise DocumentParsingError(f"Failed to parse TXT: {str(e)}")

def parse_document(file_content: bytes, file_extension: str) -> List[ParsedPage]:
    ext = file_extension.lower()
    if ext == 'pdf':
        return parse_pdf(file_content)
    elif ext == 'docx':
        return parse_docx(file_content)
    elif ext in ['txt', 'md']:
        return parse_txt(file_content)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
