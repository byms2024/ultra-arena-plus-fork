"""
PDF metadata utilities.

Reads standard PDF document information and optionally parses custom fields
like '/DmsData' which may contain a JSON string with domain-specific data.
"""

from typing import Dict, Any
import json
import logging


def read_pdf_metadata_dict(pdf_path: str) -> Dict[str, Any]:
    """Read PDF metadata and parse DmsData JSON when present.

    Returns a dictionary with keys:
      - document_info: dict of raw document info entries (string keys)
      - dms_data: dict parsed from the '/DmsData' entry if present and valid JSON
    """
    document_info: Dict[str, Any] = {}
    dms_data: Dict[str, Any] = {}

    try:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except Exception as e:
            logging.warning(f"PyPDF2 not available for metadata extraction: {e}")
            return {"document_info": document_info, "dms_data": dms_data}

        reader = PdfReader(pdf_path)
        info = getattr(reader, "metadata", None)
        if info is None:
            # Older PyPDF2 versions
            info = getattr(reader, "getDocumentInfo", lambda: None)()

        if info:
            # info can be a dict-like object; iterate and stringify
            try:
                iterable = info.items()
            except Exception:
                # Fallback for objects that expose attributes
                iterable = [(k, getattr(info, k)) for k in dir(info) if k.startswith("/")]

            for k, v in iterable:
                key_str = str(k)
                try:
                    val_str = str(v) if v is not None else ""
                except Exception:
                    val_str = ""
                document_info[key_str] = val_str

            # Parse DmsData if present
            dms_raw = document_info.get("/DmsData")
            if dms_raw:
                try:
                    dms_data = json.loads(dms_raw)
                except Exception as e:
                    logging.warning(f"Failed to parse /DmsData JSON in {pdf_path}: {e}")

    except Exception as e:
        logging.error(f"Error reading PDF metadata from {pdf_path}: {e}")

    return {"document_info": document_info, "dms_data": dms_data}


