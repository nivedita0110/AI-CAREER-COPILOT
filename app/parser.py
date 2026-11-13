"""
PDF Extraction and Document Ingestion Service
Handles binary file streaming, page extraction, format validation,
and detects scanned / image-only PDFs.
"""
import pymupdf as fitz
from typing import Dict, Union
import io
import re
from app.config import MAX_FILE_SIZE_MB


def clean_extracted_text(text: str) -> str:
    """Normalizes whitespace and cleans unwanted control characters."""
    if not text:
        return ""
    # Replace non-breaking spaces and unusual whitespace
    text = text.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing whitespace on each line
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def parse_resume_pdf(file_source: Union[io.BytesIO, bytes, str], max_size_mb: int = MAX_FILE_SIZE_MB) -> Dict:
    """
    Parses a PDF file safely and returns structured extraction metadata.

    Returns:
        dict: {
            "success": bool,
            "text": str,
            "page_count": int,
            "is_scanned": bool,
            "char_count": int,
            "word_count": int,
            "error": str or None
        }
    """
    try:
        # 1. Source resolution and size check
        if isinstance(file_source, str):
            with open(file_source, "rb") as f:
                stream_bytes = f.read()
        elif hasattr(file_source, "read"):
            if hasattr(file_source, "seek"):
                file_source.seek(0)
            stream_bytes = file_source.read()
            if hasattr(file_source, "seek"):
                file_source.seek(0)
        elif isinstance(file_source, bytes):
            stream_bytes = file_source
        else:
            return {
                "success": False,
                "text": "",
                "page_count": 0,
                "is_scanned": False,
                "char_count": 0,
                "word_count": 0,
                "error": "Unsupported file source type."
            }

        # Size validation
        size_mb = len(stream_bytes) / (1024 * 1024)
        if size_mb > max_size_mb:
            return {
                "success": False,
                "text": "",
                "page_count": 0,
                "is_scanned": False,
                "char_count": 0,
                "word_count": 0,
                "error": f"File size ({size_mb:.1f} MB) exceeds maximum allowed limit ({max_size_mb} MB)."
            }

        # 2. PyMuPDF Ingestion
        page_texts = []
        with fitz.open(stream=stream_bytes, filetype="pdf") as doc:
            page_count = len(doc)
            if page_count == 0:
                return {
                    "success": False,
                    "text": "",
                    "page_count": 0,
                    "is_scanned": False,
                    "char_count": 0,
                    "word_count": 0,
                    "error": "The uploaded PDF has 0 pages or is corrupted."
                }

            for page_num in range(page_count):
                page = doc.load_page(page_num)
                page_text = page.get_text("text")
                if page_text:
                    page_texts.append(page_text)

        full_raw_text = "\n\n".join(page_texts)
        cleaned = clean_extracted_text(full_raw_text)

        # 3. Scanned PDF Detection (Check if pages exist but virtually no text was extracted)
        word_count = len(cleaned.split())
        is_scanned = page_count > 0 and word_count < 25

        return {
            "success": True,
            "text": cleaned,
            "page_count": page_count,
            "is_scanned": is_scanned,
            "char_count": len(cleaned),
            "word_count": word_count,
            "error": "Warning: Very little text detected. If this is a scanned document, OCR is recommended." if is_scanned else None
        }

    except Exception as e:
        return {
            "success": False,
            "text": "",
            "page_count": 0,
            "is_scanned": False,
            "char_count": 0,
            "word_count": 0,
            "error": f"Failed to parse PDF document: {str(e)}"
        }


def extract_text_from_pdf(file_source: Union[io.BytesIO, bytes, str]) -> str:
    """
    Backwards-compatible convenience function that returns pure text or an error message.
    """
    res = parse_resume_pdf(file_source)
    if res["success"]:
        return res["text"]
    return f"Error extracting PDF: {res['error']}"
