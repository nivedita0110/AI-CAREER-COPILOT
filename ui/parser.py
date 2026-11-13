"""
Backward-compatibility adapter for PDF parsing.
Delegates to app.parser for modern, safe extraction.
"""
from app.parser import extract_text_from_pdf, parse_resume_pdf

__all__ = ["extract_text_from_pdf", "parse_resume_pdf"]
