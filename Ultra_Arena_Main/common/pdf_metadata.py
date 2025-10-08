"""
PDF metadata utilities.

Reads standard PDF document information and optionally parses custom fields
like '/DmsData' which may contain a JSON string with domain-specific data.
"""

from typing import Dict, Any
import json
import logging
import re
from pathlib import Path


def read_pdf_metadata_dict(pdf_path: str) -> Dict[str, Any]:
    """Read PDF metadata and parse DmsData JSON when present.

    Returns a dictionary with keys:
      - document_info: dict of raw document info entries (string keys)
      - dms_data: dict parsed from the '/DmsData' entry if present and valid JSON
    """
    document_info: Dict[str, Any] = {}
    dms_data: Dict[str, Any] = {}

    logging.info(f"Reading PDF metadata from {pdf_path}")

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
            logging.info(f"PDF metadata: {info}")
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

    # Derive invoice number from file name pattern like '...NF1234...' if present
    try:
        # Use remote_file_name from DmsData if present, else fallback to file name
        remote_file_name = dms_data.get("remote_file_name")
        if not remote_file_name:
            remote_file_name = Path(pdf_path).name
        document_info["remote_file_name"] = remote_file_name
        # More flexible regex: allow optional spaces and underscores between NF, the number, and nota
        m = re.search(r"N[\s_]*F[\s_]*(\d{1,10})[\s_]*nota", remote_file_name, re.IGNORECASE)
        if m:
            try:
                val = int(m.group(1))
                if "invoice_no" not in dms_data or not dms_data.get("invoice_no"):
                    dms_data["invoice_no"] = str(val)
            except Exception:
                pass
    except Exception:
        pass

    return {"document_info": document_info, "dms_data": dms_data}


