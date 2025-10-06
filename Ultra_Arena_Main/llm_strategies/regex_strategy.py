"""
Regex processing strategy - placeholder for regex-based document processing.
"""

import logging
import time
from typing import Iterable, Dict, List, Any, Optional, Tuple
import io
import os
import re
from dataclasses import dataclass
from pathlib import Path
from unittest import result
import pandas as pd

from Ultra_Arena_Main.llm_strategies.strategy_factory import LinkStrategy



# Optional OCR / PDF tooling
try:
    from PIL import Image  # type: ignore
    import pytesseract  # type: ignore
    from pdf2image import convert_from_path  # type: ignore
    _ocr_modules_available = False
except Exception:
    _ocr_modules_available = False

try:
    import fitz  # type: ignore  # PyMuPDF
    _pymupdf_available = True
except Exception:
    _pymupdf_available = False

try:
    import PyPDF2  # type: ignore
except Exception:
    PyPDF2 = None  # type: ignore[assignment]

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text  # type: ignore
except Exception:
    pdfminer_extract_text = None  # type: ignore[assignment]
    
from .base_strategy import BaseProcessingStrategy
from llm_client.llm_client_factory import LLMClientFactory
from llm_metrics import TokenCounter

# =========================
# Patterns and normalizers
# =========================

NFS_E_REGEXES: list[re.Pattern[str]] = [
    re.compile(r"\bNF\s*[-]?\s*S\s*[-]?\s*E\b", re.IGNORECASE),
    re.compile(r"\bNFS\s*[-\s]?e\b", re.IGNORECASE),
    re.compile(r"\bNFSe\b", re.IGNORECASE),
    re.compile(r"\bNota\s+Fiscal\s+de\s+Servi[cç]o[s]?\b", re.IGNORECASE),
    re.compile(r"\bNota\s+Fiscal\s+de\s+Servi[cç]os?\s+Eletr[oô]nica\b", re.IGNORECASE),
    re.compile(r"\bNF\s*[-\s]?Servi[cç]o[s]?\b", re.IGNORECASE),
    re.compile(r"\bEletr[oô]nica\s+de\s+Servi[cç]o[s]?\b", re.IGNORECASE),
    re.compile(r"\bEletr[oô]nica\s+de\s+Servi[cç]os?\b", re.IGNORECASE),
    re.compile(r"\bTomador(a)?\s+de\s+Servi[cç]o[s]?\b", re.IGNORECASE),
    re.compile(r"\bServi[cç]o[s]?\s+Tomado[s]?\b", re.IGNORECASE),
]

NFE_REGEXES: list[re.Pattern[str]] = [
    re.compile(r"\bNF\s*[-]?\s*e\b", re.IGNORECASE),
    re.compile(r"\bNFe\b", re.IGNORECASE),
]

CLAIM_NO_REGEX: re.Pattern[str] = re.compile(
    r"(?P<prefix>BY)?DAMEBR(?P<body>[A-Z0-9]{8,30}_[0-9A-Z]{2})",
    re.IGNORECASE,
)

VIN_REGEX: re.Pattern[str] = re.compile(
    r"L[A-HJ-NPR-Z0-9][0X][A-HJ-NPR-Z0-9]{6,10}\d{7}",
    re.IGNORECASE,
)

CNPJ_BANNED: set[str] = {
    "17.140.820/0007-77",
    "17140820000777",
    "171408200007-77",
    "17140820/000777",
    "17140820/0007-77",
}

CNPJ_BANNED_DIGITS: set[str] = {re.sub(r"\D", "", v) for v in CNPJ_BANNED}

CNPJ_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?<!\d)\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}(?!\d)"),
    re.compile(r"(?<!\d)\d{2}\.\d{3}\.\d{3}/\d{6}(?!\d)"),
    re.compile(r"(?<!\d)\d{8}/\d{4}-\d{2}(?!\d)"),
    re.compile(r"(?<!\d)\d{8}/\d{6}(?!\d)"),
    re.compile(r"(?<!\d)\d{14}(?!\d)"),
]

MONEY_REGEX: re.Pattern[str] = re.compile(
    r"(?<!\d)"
    r"(?:"
    r"\d{1,3}(?:[.,\s]\d{3})+[.,]\s?\d{2}"
    r"|"
    r"\d+[.,]\s?\d{2}"
    r")"
    r"(?!\d)"
)


def _normalize_compact(value: str) -> str:
    return re.sub(r"[\s._\-/]", "", (value or "")).upper()


def _parse_amount_to_cents(amount) -> Optional[int]:
    if isinstance(amount, float):
        return int(round(amount * 100))
    else:
        s = (amount or "").strip()
        # Find the last decimal separator (either ',' or '.')
        last_comma = s.rfind(',')
        last_dot = s.rfind('.')
        last_sep_pos = max(last_comma, last_dot)
        if last_sep_pos == -1:
            # No decimal separator, treat as integer amount in cents
            digits = re.sub(r"[^0-9]", "", s)
            if digits.isdigit():
                return int(digits) * 100
            return None
        int_part = re.sub(r"[^0-9]", "", s[:last_sep_pos])
        dec_part = re.sub(r"[^0-9]", "", s[last_sep_pos + 1:])
        # Only take up to 2 decimal digits
        dec_part = (dec_part + "00")[:2]
        if not int_part.isdigit() or not dec_part.isdigit():
            return None
        try:
            return int(int_part) * 100 + int(dec_part)
        except Exception:
            return None


def _extract_money_candidates_cents(text: str) -> list[int]:
    candidates: set[int] = set()
    for m in MONEY_REGEX.findall(text or ""):
        cents = _parse_amount_to_cents(m)
        if cents is not None:
            candidates.add(cents)
    return sorted(candidates, reverse=True)


def _find_expected_value_in_candidates(text: str, expected_cents: int, tolerance_cents: int = 1) -> bool:
    candidates = _extract_money_candidates_cents(text or "")
    if not candidates:
        return False
    for candidate in candidates:
        if abs(candidate - expected_cents) <= tolerance_cents:
            return True
    return False


def _build_amount_regex_from_cents(expected_cents: int) -> re.Pattern[str]:
    digits = f"{expected_cents:02d}"
    if len(digits) < 3:
        digits = digits.rjust(3, "0")
    int_digits = digits[:-2]
    cent_digits = digits[-2:]
    parts: list[str] = ["(?<!\\d)"]
    for d in int_digits:
        parts.append(re.escape(d))
        parts.append(r"\D*")
    parts.append(r"[\D]?\D*")
    parts.append(re.escape(cent_digits[0]))
    parts.append(r"\D*")
    parts.append(re.escape(cent_digits[1]))
    parts.append(r"(?!\d)")
    return re.compile("".join(parts))


def _find_expected_value_in_text(text: str, expected_cents: int) -> bool:
    # Prefer candidate-based matching with cent-level tolerance first
    if _find_expected_value_in_candidates(text, expected_cents, tolerance_cents=1):
        return True
    # Fallback to permissive digit-order regex
    pattern = _build_amount_regex_from_cents(expected_cents)
    if pattern.search(text):
        return True
    condensed = re.sub(r"\s+", " ", text or "")
    return pattern.search(condensed) is not None


def _contains_with_0O_ambiguity(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    if needle in (haystack or ""):
        return True

    def build_pattern(s: str) -> str:
        parts: list[str] = []
        for ch in s:
            if ch in {"0", "O"}:
                parts.append("[0O]")
            elif ch == "8":
                parts.append("[08]")
            else:
                parts.append(re.escape(ch))
        return "".join(parts)

    pattern = build_pattern(needle)
    try:
        if re.search(pattern, haystack or ""):
            return True
        normalized_hay = (haystack or "").replace("O", "0")
        if re.search(pattern, normalized_hay):
            return True
    except Exception:
        return False
    return False


# =========================
# Text extraction utilities
# =========================

def _find_poppler_bin() -> Optional[str]:
    env_path = os.environ.get("POPPLER_PATH")
    if env_path and os.path.isdir(env_path):
        if os.path.exists(os.path.join(env_path, "pdftoppm.exe")) or os.path.exists(os.path.join(env_path, "pdftocairo.exe")):
            return env_path
    user_home = os.path.expanduser("~")
    dl_root = os.path.join(user_home, "Downloads", "poppler-25.09.0")
    if os.path.isdir(dl_root):
        for root, _, _ in os.walk(dl_root):
            if os.path.basename(root).lower() == "bin":
                if os.path.exists(os.path.join(root, "pdftoppm.exe")) or os.path.exists(os.path.join(root, "pdftocairo.exe")):
                    return root
    candidates = [
        r"C:\\Program Files\\poppler\\bin",
        r"C:\\Program Files (x86)\\poppler\\bin",
        os.path.join(user_home, "Desktop", "poppler-windows", "bin"),
        os.path.join(user_home, "Desktop", "poppler-windows-master", "bin"),
        os.path.join(user_home, "Desktop", "poppler-windows-master", "Library", "bin"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            if os.path.exists(os.path.join(candidate, "pdftoppm.exe")) or os.path.exists(os.path.join(candidate, "pdftocairo.exe")):
                return candidate
    return None


def _find_tesseract_cmd() -> Optional[str]:
    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd and os.path.exists(env_cmd):
        return env_cmd
    import shutil
    which_cmd = shutil.which("tesseract")
    if which_cmd:
        return which_cmd
    candidates = [
        r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
        r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe",
        r"C:\\ProgramData\\chocolatey\\bin\\tesseract.exe",
        r"C:\\Users\\alexandre.carrer\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe",
        "/mnt/c/Program Files/Tesseract-OCR/tesseract.exe",
        "/mnt/c/Program Files (x86)/Tesseract-OCR/tesseract.exe",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


_ocr_available = False
if _ocr_modules_available:
    try:
        _tesseract_cmd = _find_tesseract_cmd()
        if _tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd  # type: ignore[attr-defined]
            _ocr_available = True
    except Exception:
        _ocr_available = False

# PDFTEXTEXTRACTOR

class PdfTextExtractor:
    @staticmethod
    def extract_text_from_txt(file_path: Path) -> str:
        try:
            try:
                return file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return file_path.read_text(encoding="latin-1", errors="ignore")
        except Exception:
            return ""

    @staticmethod
    def extract_text_from_pdf(file_path: Path) -> str:
        extracted_text = ""
        if pdfminer_extract_text is not None:
            try:
                extracted_text = pdfminer_extract_text(str(file_path)) or ""
            except Exception:
                extracted_text = ""
        elif PyPDF2 is not None:
            try:
                fragments: list[str] = []
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)  # type: ignore[attr-defined]
                    for page in getattr(reader, "pages", []):
                        try:
                            extracted = page.extract_text() or ""
                        except Exception:
                            extracted = ""
                        if extracted:
                            fragments.append(extracted)
                extracted_text = "\n".join(fragments)
            except Exception:
                extracted_text = ""

        if len(extracted_text) < 1000 and _ocr_available:
            images = []
            poppler_bin = _find_poppler_bin()
            if poppler_bin is not None:
                try:
                    images = convert_from_path(str(file_path), poppler_path=poppler_bin, dpi=300)
                except Exception:
                    images = []
            if not images and _pymupdf_available:
                try:
                    doc = fitz.open(str(file_path))
                    zoom = 300.0 / 72.0
                    matrix = fitz.Matrix(zoom, zoom)
                    for page in doc:
                        pix = page.get_pixmap(matrix=matrix)
                        images.append(Image.open(io.BytesIO(pix.tobytes("png"))))
                except Exception:
                    images = []
            try:
                ocr_texts = []
                for img in images:
                    try:
                        ocr_text = pytesseract.image_to_string(img, lang="por")
                    except Exception:
                        ocr_text = pytesseract.image_to_string(img)
                    ocr_texts.append(ocr_text)
                ocr_result = "\n".join(ocr_texts)
                if len(ocr_result) > len(extracted_text):
                    extracted_text = ocr_result
            except Exception:
                pass
        return extracted_text

    @staticmethod
    def extract_text_best_effort(file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        if suffix == ".txt":
            return PdfTextExtractor.extract_text_from_txt(file_path)
        if suffix == ".pdf":
            return PdfTextExtractor.extract_text_from_pdf(file_path)
        try:
            return file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""


class PdfClassifier:
    @staticmethod
    def is_servico(text: str) -> bool:
        for p in NFS_E_REGEXES:
            if p.search(text or ""):
                return True
        return False

    @staticmethod
    def is_pecas(text: str) -> bool:
        for p in NFE_REGEXES:
            if p.search(text or ""):
                return True
        return False

    @staticmethod
    def classify_pdf(text: str) -> str:
        if PdfClassifier.is_servico(text):
            return "Serviço"
        if PdfClassifier.is_pecas(text):
            return "Peças"
        return "Outros"


class FieldExtractor:
    @staticmethod
    def extract_claim_no_blind(text: str) -> Optional[str]:
        m = CLAIM_NO_REGEX.search(text or "")
        if not m:
            return None
        body = m.group('body')
        return f"BYDAMEBR{body}"

    @staticmethod
    def match_expected_claim_no(text: str, expected_claim: Optional[str]) -> Optional[str]:
        if not expected_claim:
            return None
        text_compact = _normalize_compact(text or "")
        claim_compact = _normalize_compact(expected_claim)
        found = _contains_with_0O_ambiguity(text_compact, claim_compact)
        if not found and claim_compact.startswith("BY"):
            found = _contains_with_0O_ambiguity(text_compact, claim_compact[2:])
        if found:
            normalized = expected_claim
            if not normalized.upper().startswith("BY"):
                normalized = f"BY{normalized}"
            return normalized
        return None

    @staticmethod
    def extract_vin_blind(text: str) -> Optional[str]:
        m = VIN_REGEX.search(text or "")
        return m.group(0) if m else None

    @staticmethod
    def match_expected_vin(text: str, expected_vin: Optional[str]) -> Optional[str]:
        if not expected_vin:
            return None
        text_compact = _normalize_compact(text or "")
        vin_compact = _normalize_compact(expected_vin)
        return expected_vin if _contains_with_0O_ambiguity(text_compact, vin_compact) else None

    @staticmethod
    def extract_cnpj_blind(text: str) -> Optional[str]:
        for pattern in CNPJ_PATTERNS:
            candidates = pattern.findall(text or "")
            if not candidates:
                continue
            for c in candidates:
                digits = re.sub(r"\D", "", c)
                if digits in CNPJ_BANNED_DIGITS:
                    continue
                return c
        return None

    @staticmethod
    def match_expected_cnpj(text: str, expected_cnpj: Optional[str]) -> Optional[str]:
        if not expected_cnpj:
            return None
        text_digits = re.sub(r"\D", "", (text or "").replace('O', '0').replace('o', '0'))
        cnpj_digits = re.sub(r"\D", "", expected_cnpj.replace('O', '0').replace('o', '0'))
        if len(cnpj_digits) == 14 and cnpj_digits not in CNPJ_BANNED_DIGITS and cnpj_digits in text_digits:
            return expected_cnpj
        return None

    @staticmethod
    def extract_cnpj2_blind(text: str) -> str:
        text_digits_only = re.sub(r"\D", "", text or "")
        if "17140820000777" in text_digits_only or "171408201000777" in text_digits_only:
            return "17.140.820/0007-77"
        return ""

    @staticmethod
    def extract_price_candidates_cents(text: str) -> list[int]:
        return _extract_money_candidates_cents(text or "")

    @staticmethod
    def match_expected_amount(text: str, expected_amount_str: Optional[str]) -> Optional[int]:
        if not expected_amount_str:
            return None
        cents = _parse_amount_to_cents(expected_amount_str)
        if cents is None:
            return None
        return cents if _find_expected_value_in_text(text or "", cents) else None

    @staticmethod
    def extract_invoice_no_from_filename(filename: str) -> Optional[str]:
        try:
            m = re.search(r"NF\s*-?\s*(\d{1,7})", filename or "", re.IGNORECASE)
            if not m:
                return None
            num = (m.group(1) or "").strip()
            if not num.isdigit():
                return None
            val = int(num)
            if 1 <= val <= 1_000_000:
                return str(val)
        except Exception:
            return None
        return None

    @staticmethod
    def extract_invoice_no(text: str) -> Optional[str]:
        if not text:
            return None
        head = (text or "")[:2000]

        # Collect RPS numbers to exclude
        rps_numbers: set[str] = set()
        for m in re.finditer(r"(?:N[úu]mero\s*RPS|RPS\s*N[ºo]?|N[ºo]\s*RPS|No\s*RPS)\s*[:\-]?\s*(\d{1,10})", head, re.IGNORECASE):
            try:
                rps_numbers.add(m.group(1))
            except Exception:
                continue

        # Candidate patterns for invoice number (NF / NFS labels)
        patterns = [
            r"(?:N[ºo]\s*(?:da\s*)?NFS?)\s*[:\-]?\s*(\d{1,7})",
            r"(?:N[úu]mero\s*(?:da\s*)?NF(?:S)?|N[úu]mero\s*(?:da\s*)?NFS)\s*[:\-]?\s*(\d{1,7})",
            r"NF(?:S)?\s*N[ºo]?\s*[:\-]?\s*(\d{1,7})",
        ]

        for pat in patterns:
            try:
                for m in re.finditer(pat, head, re.IGNORECASE):
                    num = (m.group(1) or "").strip()
                    if not num or num in rps_numbers:
                        continue
                    if not num.isdigit():
                        continue
                    val = int(num)
                    if 1 <= val <= 1_000_000:
                        # Ensure local context does not mention RPS
                        start = max(m.start() - 30, 0)
                        ctx = head[start:m.start()]
                        if re.search(r"RPS", ctx, re.IGNORECASE):
                            continue
                        return num
            except Exception:
                continue
        return None

    @staticmethod
    def extract_invoice_issue_date(text: str) -> Optional[str]:
        if not text:
            return None
        head = (text or "")[:2000]
        date_pat = r"(\d{2}/\d{2}/\d{4})"
        triggers = [
            r"Emitida\s+em",
            r"Data\s+de\s+Emiss[aã]o",
            r"Emiss[aã]o",
        ]
        for trig in triggers:
            try:
                pat = re.compile(trig + r"\s*[:\-]?\s*" + date_pat, re.IGNORECASE)
                m = pat.search(head)
                if m:
                    return m.group(1)
            except Exception:
                continue
        # Fallback: first date near top of document
        m = re.search(date_pat, head)
        return m.group(1) if m else None


# =========================
# Public API helpers
# =========================

def _iter_pdf_files(root: Path) -> Iterable[Path]:
    skip_dirs = {".git", "node_modules", "venv", ".venv", "__pycache__"}
    if not root.exists() or not root.is_dir():
        return []
    for path in root.rglob("*.pdf"):
        try:
            if any(part in skip_dirs for part in path.parts):
                continue
            yield path
        except Exception:
            continue


def _format_brl_from_cents(cents: Optional[int]) -> str:
    if cents is None:
        return ""
    reais = cents // 100
    c = cents % 100
    return f"{reais:,}".replace(",", ".") + f",{c:02d}"


@dataclass
class Answers:
    claim_no: Optional[str] = None
    vin: Optional[str] = None
    service_price: Optional[str] = None
    parts_price: Optional[str] = None
    cnpj: Optional[str] = None


@dataclass
class PreprocessedData:
    files: list[Path]
    file_texts: dict[Path, str]
    file_classes: dict[Path, str]
    answers: Answers

def categorize_pdfs(root: str | Path) -> pd.DataFrame:
    root_path = Path(root).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for file_path in _iter_pdf_files(root_path):
        text = PdfTextExtractor.extract_text_best_effort(file_path)
        doc_class = PdfClassifier.classify_pdf(text)
        rows.append({"file": str(file_path), "class": doc_class})
    return pd.DataFrame(rows, columns=["file", "class"]) if rows else pd.DataFrame(columns=["file", "class"])

def _collect_from_files(
    files: list[Path],
    answers: Answers,
    want_service_price: bool,
    want_parts_price: bool,
    ) -> tuple[dict[str, Any], dict[str, bool]]:
        data: dict[str, Any] = {
            "claim_no": None,
            "vin": None,
            "cnpj": None,
            "service_price_cents": None,
            "parts_price_cents": None,
            "cnpj2": "",
        }
        used_answers: dict[str, bool] = {k: False for k in ["claim_no", "vin", "cnpj", "service_price_cents", "parts_price_cents"]}

        for pdf in files:
            text = PdfTextExtractor.extract_text_best_effort(pdf)

            # Try answers-guided for missing fields
            if data["claim_no"] is None:
                m = FieldExtractor.match_expected_claim_no(text, answers.claim_no)
                if m:
                    data["claim_no"] = m
                    used_answers["claim_no"] = True
            if data["vin"] is None:
                m = FieldExtractor.match_expected_vin(text, answers.vin)
                if m:
                    data["vin"] = m
                    used_answers["vin"] = True
            if data["cnpj"] is None:
                m = FieldExtractor.match_expected_cnpj(text, answers.cnpj)
                if m:
                    data["cnpj"] = m
                    used_answers["cnpj"] = True

            if want_service_price and data["service_price_cents"] is None:
                m = FieldExtractor.match_expected_amount(text, answers.service_price)
                if m is not None:
                    data["service_price_cents"] = m
                    used_answers["service_price_cents"] = True

            if want_parts_price and data["parts_price_cents"] is None:
                m = FieldExtractor.match_expected_amount(text, answers.parts_price if isinstance(answers, Answers) else answers.get("parts_price"))
                if m is not None:
                    data["parts_price_cents"] = m
                    used_answers["parts_price_cents"] = True

            # Blind fallback for remaining fields
            if data["claim_no"] is None:
                b = FieldExtractor.extract_claim_no_blind(text)
                if b:
                    data["claim_no"] = b
            if data["vin"] is None:
                b = FieldExtractor.extract_vin_blind(text)
                if b:
                    data["vin"] = b
            if data["cnpj"] is None:
                b = FieldExtractor.extract_cnpj_blind(text)
                if b:
                    data["cnpj"] = b

            if want_service_price and data["service_price_cents"] is None:
                candidates = FieldExtractor.extract_price_candidates_cents(text)
                if candidates:
                    data["service_price_cents"] = candidates[0]

            if want_parts_price and data["parts_price_cents"] is None:
                candidates = FieldExtractor.extract_price_candidates_cents(text)
                if candidates:
                    data["parts_price_cents"] = candidates[0]

            # CNPJ2 only makes sense for Serviço context; but harmless to collect here
            if not data["cnpj2"]:
                data["cnpj2"] = FieldExtractor.extract_cnpj2_blind(text)

            # Early stop if everything requested is found
            done = (
                data["claim_no"] is not None and
                data["vin"] is not None and
                data["cnpj"] is not None and
                (not want_service_price or data["service_price_cents"] is not None) and
                (not want_parts_price or data["parts_price_cents"] is not None)
            )
            if done:
                break

        return data, used_answers


class RegexPreProcessingStrategy(LinkStrategy):
    def __init__(self, config: Dict[str, Any] | None = None, streaming: bool = False):
        super().__init__(config, streaming)
        self.config = config or {}
        self.streaming = streaming

    def preprocess_filepaths(self, file_paths: list[str], manual_answers: Optional[Answers] = None) -> PreprocessedData:
        files: list[Path] = [Path(f).expanduser().resolve() for f in file_paths]

        auto_answers = self.get_target_from_pdfs_metadata(file_paths)
        # If there is no DMS metadata at all, skip any regex pre-processing work
        if not auto_answers:
            return PreprocessedData(
                files=files,
                file_texts={},
                file_classes={},
                answers=auto_answers,
            )
       
        if manual_answers is not None:
            auto_answers = Answers(
                claim_no=manual_answers.claim_no if manual_answers.claim_no is not None else auto_answers.claim_no,
                vin=manual_answers.vin if manual_answers.vin is not None else auto_answers.vin,
                service_price=manual_answers.service_price if manual_answers.service_price is not None else auto_answers.service_price,
                parts_price=manual_answers.parts_price if manual_answers.parts_price is not None else auto_answers.parts_price,
                cnpj=manual_answers.cnpj if manual_answers.cnpj is not None else auto_answers.cnpj,
            )

        file_texts: dict[Path, str] = {}
        file_classes: dict[Path, str] = {}
        for f in files:
            try:
                # FORCE BLACKLISTING FOR TESTING: Always fail text extraction
                t = PdfTextExtractor.extract_text_best_effort(f)
                # raise Exception("FORCED BLACKLISTING: Simulating text extraction failure")

                # Check if text extraction failed or returned empty/invalid text
                if not t or len(t.strip()) < 10:  # Consider text too short to be useful
                    self.blacklist_file(str(f), "Text extraction failed or returned insufficient content", "regex_preprocessing")
                    logging.warning(f"🚫 Blacklisting {f} due to poor text extraction quality")
                    continue
                file_texts[f] = t
                file_classes[f] = PdfClassifier.classify_pdf(t)
            except Exception as e:
                # Text extraction threw an exception - blacklist the file
                self.blacklist_file(str(f), f"Text extraction error: {str(e)}", "regex_preprocessing")
                logging.error(f"🚫 Blacklisting {f} due to text extraction exception: {e}")
                continue

        return PreprocessedData(
            files=files,
            file_texts=file_texts,
            file_classes=file_classes,
            answers=auto_answers,
        )
    
    def get_target_from_pdfs_metadata(self, file_group: list[str]) -> dict[str, dict]:
        from Ultra_Arena_Main.common.pdf_metadata import read_pdf_metadata_dict
        answers: dict[str, dict] = {}

        for file_path in file_group:
            # Check if file is blacklisted for regex processing
            if self.is_file_blacklisted(file_path, "regex_preprocessing"):
                logging.warning(f"🚫⚠️  SKIPPING BECAUSE BLACKLISTED: {file_path} - Previously failed regex text extraction")
                continue

            meta = read_pdf_metadata_dict(file_path)
            # Push parsed DMS data into passthrough extracted_data
            dms = meta.get("dms_data") or {}
            if dms:
                # Map keys we care about directly
                mapped = {
                    "claim_id": dms.get("claim_id"),
                    "claim_no": dms.get("claim_no"),
                    "vin": dms.get("vin"),
                    "dealer_code": dms.get("dealer_code"),
                    "dealer_name": dms.get("dealer_name"),
                    "cnpj1": dms.get("dealer_cnpj"),  # BYD CNPJ per example
                    "gross_credit_dms": dms.get("gross_credit"),
                    "labour_amount_dms": dms.get("labour_amount_dms"),
                    "part_amount_dms": dms.get("part_amount_dms"),
                    "dms_file_id": dms.get("file_id"),
                    "dms_embedded_at": dms.get("embedded_at"),
                    "invoice_no_dms": dms.get("invoice_no"),
                    "remote_file_name": (meta.get("document_info", {}) or {}).get("remote_file_name"),
                }
                # Store DMS data in passthrough for visibility in logs
                if any(v is not None for v in mapped.values()):
                    self.update_extracted_data(file_path, {k: v for k, v in mapped.items() if v is not None})
                # Store as a dict, not as an Answers object
                answers[Path(file_path).expanduser().resolve()] = {
                    "claim_no": mapped.get("claim_no"),
                    "vin": mapped.get("vin"),
                    "service_price": mapped.get("labour_amount_dms"),
                    "parts_price": mapped.get("part_amount_dms"),
                    "cnpj": mapped.get("cnpj1"),
                    "invoice_no": mapped.get("invoice_no_dms"),
                    "remote_file_name": mapped.get("remote_file_name"),
                }
            # Optionally keep raw document info
            if self.config.get("store_raw_pdf_info", False):
                # Only attach when metadata exists; avoid polluting passthrough when no metadata
                if meta.get("document_info"):
                    self.update_extracted_data(file_path, {"pdf_document_info": meta.get("document_info", {})})

        return answers  # dict mapping file_path to dict; override via manual answers if provided


    def process_file_group(
        self,
        *,
        config_manager=None,
        file_group: list[str],
        group_index: int,
        group_id: str = "",
        system_prompt: str = None,
        user_prompt: str = None,
        **kwargs
    ) -> tuple[list[PreprocessedData], dict, str]:
        """
        Process a group of files for pre-processing (text extraction, classification, answer inference).
        Returns a tuple: (list of PreprocessedData, stats dict, info string)
        """
        bundles = [self.preprocess_filepaths(file_group)]
        # Batch the entire file_group in a single preprocessing call
        stats = {"total_files": len(file_group)}
        info = ""
        return  bundles, stats, info



class RegexProcessingStrategy(LinkStrategy):
    def __init__(self, config: Dict[str, Any], streaming: bool = False):
        super().__init__(config)
        self.streaming = streaming
        self.llm_provider = config.get("llm_provider", "google")
        self.provider_config = config.get("provider_configs", {}).get(self.llm_provider, {})
        self.regex_patterns = config.get("regex_patterns", {})
        # Disable LLM fallback by default to avoid requiring provider credentials
        self.fallback_llm = bool(config.get("fallback_llm", False))
        # Initialize LLM client only if explicitly enabled and configured
        self.llm_client = None
        if self.fallback_llm and self.provider_config:
            try:
                self.llm_client = LLMClientFactory.create_client(
                    self.llm_provider, self.provider_config, streaming=self.streaming
                )
            except Exception as e:
                logging.warning(f"LLM fallback disabled due to client init error: {e}")
                self.llm_client = None

    def process_file_group(self, *, config_manager=None, file_group: List[str], group_index: int,
                          group_id: str = "", system_prompt: Optional[str] = None, user_prompt: str, pre_results: List[PreprocessedData] = None) -> Tuple[List[Tuple[str, Dict]], Dict, str]:
        """
        Process a group of files using regex-based extraction.

        This implementation uses pre-processing to extract text and targets,
        followed by processing to build per-file results.
        """
        start_time = time.time()
        results = []
        agg_stats = {
            "total_files": len(file_group),
            "successful_files": 0,
            "failed_files": 0,
            "total_tokens": 0,
            "estimated_tokens": 0,
            "prompt_tokens": 0,
            "candidate_tokens": 0,
            "processing_time": 0
        }


        # Use the answers from pre_results (a list of PreprocessedData) to set self.answers
        self.answers = None
        if pre_results and isinstance(pre_results, list) and len(pre_results) > 0:
            # Use the answers from the first PreprocessedData (assuming all files in group share answers)
            first_pre = pre_results[0]
            if hasattr(first_pre, "answers"):
                self.answers = first_pre.answers


        try:
            # Prepare manual overrides from self.answers if present
            manual_answers = None
            if self.answers is not None:
                manual_answers = Answers(
                    claim_no=getattr(self.answers, "claim_no", None),
                    vin=getattr(self.answers, "vin", None),
                    service_price=getattr(self.answers, "service_price", None),
                    parts_price=getattr(self.answers, "parts_price", None),
                    cnpj=getattr(self.answers, "cnpj", None),
                )

            # If there is no metadata for any file, return empty results to avoid overwriting prior subchain data
            has_any_metadata = False
            if pre_results and isinstance(pre_results, list):
                for p in pre_results:
                    try:
                        ans_map = getattr(p, "answers", None)
                        files = getattr(p, "files", [])
                        if isinstance(ans_map, dict):
                            for f in files:
                                ans = ans_map.get(f)
                                if isinstance(ans, dict) and len(ans) > 0:
                                    has_any_metadata = True
                                    break
                        if has_any_metadata:
                            break
                    except Exception:
                        continue

            if not has_any_metadata:
                for file_path in file_group:
                    try:
                        # Do not write to passthrough when skipping due to no metadata
                        results.append((file_path, {}))
                        agg_stats["successful_files"] += 1
                    except Exception as e:
                        logging.error(f"❌ Error preparing empty result for file {file_path}: {e}")
                        error_result = {"error": str(e), "file_path": file_path}
                        results.append((file_path, error_result))
                        agg_stats["failed_files"] += 1
                agg_stats["processing_time"] = time.time() - start_time
                return results, agg_stats, "skipped_no_metadata"

            # Blacklist any files that have empty text in pre_results
            try:
                if pre_results and isinstance(pre_results, list):
                    for p in pre_results:
                        files = getattr(p, "files", [])
                        file_texts = getattr(p, "file_texts", {}) or {}
                        for f in files:
                            try:
                                t = file_texts.get(f, "") if isinstance(file_texts, dict) else ""
                                if not t:
                                    self.blacklist_file(str(f), "Empty text in pre_results file_texts", "regex_preprocessing")
                            except Exception:
                                continue
            except Exception:
                # Non-fatal; continue processing
                pass

            # Process normally
            df = self.process_preprocessed_filepaths(pre_results)

            # For each file, collect the corresponding row as a dict
            for idx, row in df.iterrows():
                try:
                    file_path = row.get("file", None)
                    if file_path is None:
                        error_result = {"error": "No file path in row", "file_path": None}
                        results.append((None, error_result))
                        agg_stats["failed_files"] += 1
                        continue
                    file_result = row.to_dict()
                    results.append((file_path, file_result))
                    agg_stats["successful_files"] += 1
                except Exception as e:
                    logging.error(f"❌ Error extracting row for file {row.get('file', None)}: {e}")
                    error_result = {"error": str(e), "file_path": row.get("file", None)}
                    results.append((row.get("file", None), error_result))
                    agg_stats["failed_files"] += 1

        except Exception as e:
            logging.error(f"❌ Error processing file group: {e}")
            for file_path in file_group:
                error_result = {"error": str(e), "file_path": file_path}
                results.append((file_path, error_result))
                agg_stats["failed_files"] += 1

        agg_stats["processing_time"] = time.time() - start_time

        return results, agg_stats, "completed"


    def process_preprocessed_filepaths(self, pre: PreprocessedData) -> pd.DataFrame:

        # Normalize input to an iterable of PreprocessedData
        pre_list: list[PreprocessedData] = pre if isinstance(pre, list) else [pre]

        # If there is no metadata for any file, skip regex processing entirely
        has_any_metadata = False
        for p in pre_list:
            if isinstance(p.answers, dict):
                for f in p.files:
                    ans = p.answers.get(f, None)
                    if isinstance(ans, dict) and len(ans) > 0:
                        has_any_metadata = True
                        break
            if has_any_metadata:
                break

        columns = [
            "file",
            "class",
            "collected_service_price",
            "collected_parts_price",
            "collected_INVOICE_NO",
            "collected_INVOICE_ISSUE_DATE",
            "collected_CNPJ",
            "collected_CNPJ2",
            "collected_VIN",
            "collected_ClaimNO",
            "search_mode",
        ]

        if not has_any_metadata:
            rows_no_meta: list[dict[str, Any]] = []
            for p in pre_list:
                for f in p.files:
                    cls = p.file_classes.get(f, "")
                    rows_no_meta.append({
                        "file": str(f.name),
                        "class": cls,
                        "collected_service_price": "",
                        "collected_parts_price": "",
                        "collected_INVOICE_NO": "",
                        "collected_INVOICE_ISSUE_DATE": "",
                        "collected_CNPJ": "",
                        "collected_CNPJ2": "",
                        "collected_VIN": "",
                        "collected_ClaimNO": "",
                        "search_mode": "skipped_no_metadata",
                    })
            return pd.DataFrame(rows_no_meta, columns=columns) if rows_no_meta else pd.DataFrame(columns=columns)

        rows: list[dict[str, Any]] = []
        found_service_price_any = False
        found_parts_price_any = False
        temp_rows: dict[Path, dict[str, Any]] = {}

        for p in pre_list:

            for f in p.files:
                file_texts = p.file_texts.get(f, "")
                file_classes = p.file_classes.get(f, "")
                answers = p.answers.get(f) if isinstance(p.answers, dict) else None

                cls = file_classes
                text = file_texts

                row: dict[str, Any] = {
                    "file": str(f.name),
                    "class": cls,
                    "collected_service_price": "",
                    "collected_parts_price": "",
                    "collected_INVOICE_NO": "",
                    "collected_INVOICE_ISSUE_DATE": "",
                    "collected_CNPJ": "",
                    "collected_CNPJ2": "",
                    "collected_VIN": "",
                    "collected_ClaimNO": "",
                    "search_mode": "blind",
                }

                used_answers_any = False

                claim_no_ans = answers.get("claim_no") if isinstance(answers, dict) else None
                vin_ans = answers.get("vin") if isinstance(answers, dict) else None
                cnpj_ans = answers.get("cnpj") if isinstance(answers, dict) else None
                service_price_ans = answers.get("service_price") if isinstance(answers, dict) else None
                parts_price_ans = answers.get("parts_price") if isinstance(answers, dict) else None

                if cls == "Serviço":
                    m = FieldExtractor.match_expected_claim_no(text, claim_no_ans)
                    if m:
                        row["collected_ClaimNO"] = m
                        used_answers_any = True
                    if not row["collected_ClaimNO"]:
                        b = FieldExtractor.extract_claim_no_blind(text)
                        if b:
                            row["collected_ClaimNO"] = b

                    m = FieldExtractor.match_expected_vin(text, vin_ans)
                    if m:
                        row["collected_VIN"] = m
                        used_answers_any = True
                    if not row["collected_VIN"]:
                        b = FieldExtractor.extract_vin_blind(text)
                        if b:
                            row["collected_VIN"] = b

                    m = FieldExtractor.match_expected_cnpj(text, cnpj_ans)
                    if m:
                        row["collected_CNPJ"] = m
                        used_answers_any = True
                    if not row["collected_CNPJ"]:
                        b = FieldExtractor.extract_cnpj_blind(text)
                        if b:
                            row["collected_CNPJ"] = b

                    m_amt = FieldExtractor.match_expected_amount(text, service_price_ans)
                    if m_amt is not None:
                        row["collected_service_price"] = _format_brl_from_cents(m_amt)
                        used_answers_any = True
                    if not row["collected_service_price"]:
                        cands = FieldExtractor.extract_price_candidates_cents(text)
                        if cands and service_price_ans:
                            clean_target = str(service_price_ans).replace(".", "").replace(",", "")
                            for candidate in cands:
                                str_candidate = str(candidate)
                                formatted = _format_brl_from_cents(candidate)
                                if clean_target and str_candidate in clean_target:
                                    row["collected_service_price"] = formatted
                                    break

                    row["collected_CNPJ2"] = FieldExtractor.extract_cnpj2_blind(text)

                    # New: attempt invoice fields extraction
                    inv_no = FieldExtractor.extract_invoice_no(text)
                    if not inv_no and isinstance(answers, dict):
                        # fallback to filename-derived invoice number when present in metadata read
                        inv_no = FieldExtractor.extract_invoice_no_from_filename(
                            (answers.get("remote_file_name") or "")
                        ) or FieldExtractor.extract_invoice_no_from_filename(str(f.name))
                    if inv_no:
                        row["collected_INVOICE_NO"] = inv_no
                    inv_date = FieldExtractor.extract_invoice_issue_date(text)
                    if inv_date:
                        row["collected_INVOICE_ISSUE_DATE"] = inv_date

                    if row["collected_service_price"]:
                        found_service_price_any = True

                elif cls == "Peças":
                    m = FieldExtractor.match_expected_claim_no(text, claim_no_ans)
                    if m:
                        row["collected_ClaimNO"] = m
                        used_answers_any = True
                    if not row["collected_ClaimNO"]:
                        b = FieldExtractor.extract_claim_no_blind(text)
                        if b:
                            row["collected_ClaimNO"] = b

                    m = FieldExtractor.match_expected_vin(text, vin_ans)
                    if m:
                        row["collected_VIN"] = m
                        used_answers_any = True
                    if not row["collected_VIN"]:
                        b = FieldExtractor.extract_vin_blind(text)
                        if b:
                            row["collected_VIN"] = b

                    m = FieldExtractor.match_expected_cnpj(text, cnpj_ans)
                    if m:
                        row["collected_CNPJ"] = m
                        used_answers_any = True
                    if not row["collected_CNPJ"]:
                        b = FieldExtractor.extract_cnpj_blind(text)
                        if b:
                            row["collected_CNPJ"] = b

                    m_amt = FieldExtractor.match_expected_amount(text, parts_price_ans)
                    if m_amt is not None:
                        row["collected_parts_price"] = _format_brl_from_cents(m_amt)
                        used_answers_any = True
                    if not row["collected_parts_price"]:
                        cands = FieldExtractor.extract_price_candidates_cents(text)
                        if cands and parts_price_ans:
                            clean_target = str(parts_price_ans).replace(".", "").replace(",", "")
                            for candidate in cands:
                                str_candidate = str(candidate)
                                formatted = _format_brl_from_cents(candidate)
                                if clean_target and str_candidate in clean_target:
                                    row["collected_parts_price"] = formatted
                                    break

                    if row["collected_parts_price"]:
                        found_parts_price_any = True

                    # New: attempt invoice fields extraction
                    inv_no = FieldExtractor.extract_invoice_no(text)
                    if not inv_no and isinstance(answers, dict):
                        inv_no = FieldExtractor.extract_invoice_no_from_filename(
                            (answers.get("remote_file_name") or "")
                        ) or FieldExtractor.extract_invoice_no_from_filename(str(f.name))
                    if inv_no:
                        row["collected_INVOICE_NO"] = inv_no
                    inv_date = FieldExtractor.extract_invoice_issue_date(text)
                    if inv_date:
                        row["collected_INVOICE_ISSUE_DATE"] = inv_date

                row["search_mode"] = "answers" if used_answers_any else "blind"
                temp_rows[f] = row
            for f in p.files:
                file_texts = p.file_texts.get(f, "")
                file_classes = p.file_classes.get(f, "")
                answers = p.answers.get(f) if isinstance(p.answers, dict) else None

                if file_classes != "Outros":
                    continue
                text = file_texts
                row = temp_rows.get(f)
                if row is None:
                    row = {
                        "file": str(f),
                        "class": "Outros",
                        "collected_service_price": "",
                        "collected_parts_price": "",
                        "collected_INVOICE_NO": "",
                        "collected_INVOICE_ISSUE_DATE": "",
                        "collected_CNPJ": "",
                        "collected_CNPJ2": "",
                        "collected_VIN": "",
                        "collected_ClaimNO": "",
                        "search_mode": "blind",
                    }

                used_answers_any = False

                service_price_ans = answers.get("service_price") if isinstance(answers, dict) else None
                parts_price_ans = answers.get("parts_price") if isinstance(answers, dict) else None

                if not found_service_price_any:
                    m_amt = FieldExtractor.match_expected_amount(text, service_price_ans)
                    if m_amt is not None:
                        row["collected_service_price"] = _format_brl_from_cents(m_amt)
                        used_answers_any = True
                    if not row["collected_service_price"]:
                        cands = FieldExtractor.extract_price_candidates_cents(text)
                        if cands:
                            row["collected_service_price"] = "0,0"

                if not found_parts_price_any:
                    m_amt = FieldExtractor.match_expected_amount(text, parts_price_ans)
                    if m_amt is not None:
                        row["collected_parts_price"] = _format_brl_from_cents(m_amt)
                        used_answers_any = True
                    if not row["collected_parts_price"]:
                        cands = FieldExtractor.extract_price_candidates_cents(text)
                        if cands:
                            row["collected_parts_price"] = "0,0"

                # New: attempt invoice fields extraction from header for 'Outros' too
                inv_no = FieldExtractor.extract_invoice_no(text)
                if not inv_no and isinstance(answers, dict):
                    inv_no = FieldExtractor.extract_invoice_no_from_filename(
                        (answers.get("remote_file_name") or "")
                    ) or FieldExtractor.extract_invoice_no_from_filename(str(f.name))
                if inv_no:
                    row["collected_INVOICE_NO"] = inv_no
                inv_date = FieldExtractor.extract_invoice_issue_date(text)
                if inv_date:
                    row["collected_INVOICE_ISSUE_DATE"] = inv_date

                if used_answers_any:
                    row["search_mode"] = "answers"

                temp_rows[f] = row

            rows.extend(temp_rows[file] for file in p.files)

        return pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)

    def _fallback_to_llm(self, file_path: str, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback to LLM processing if regex extraction is insufficient.

        This is a placeholder implementation.
        """
        if not self.llm_client:
            return extracted_data

        # Placeholder: LLM fallback logic would go here
        logging.info(f"🤖 Fallback to LLM for file: {file_path}")
        return extracted_data


def process_filepaths(
    file_paths: list[str],
    claim_no_answer: Optional[str] = None,
    vin_answer: Optional[str] = None,
    service_price_answer: Optional[str] = None,
    parts_price_answer: Optional[str] = None,
    cnpj_answer: Optional[str] = None,
) -> pd.DataFrame:
    """Return one row per file with collected fields via pre-processing + processing."""
    manual_answers = Answers(
        claim_no=claim_no_answer,
        vin=vin_answer,
        service_price=service_price_answer,
        parts_price=parts_price_answer,
        cnpj=cnpj_answer,
    )

    preprocessor = RegexPreProcessingStrategy({})
    pre = preprocessor.preprocess_filepaths(file_paths, manual_answers=manual_answers)

    processor = RegexProcessingStrategy(config={}, streaming=False, answers=manual_answers)
    return processor.process_preprocessed_filepaths(pre)


def process_repository(
    root: str | Path,
    claim_no_answer: Optional[str] = None,
    vin_answer: Optional[str] = None,
    service_price_answer: Optional[str] = None,
    parts_price_answer: Optional[str] = None,
    cnpj_answer: Optional[str] = None,
) -> pd.DataFrame:
    answers = Answers(
        claim_no=claim_no_answer,
        vin=vin_answer,
        service_price=service_price_answer,
        parts_price=parts_price_answer,
        cnpj=cnpj_answer,
    )

    # Collect from Serviços and Peças first
    servico_data = collect_data_from_servicos(root, answers)
    pecas_data = collect_data_from_pecas(root, answers)

    # Merge results preferring explicit values; prices come from specific classes
    claim_no = servico_data.get("claim_no") or pecas_data.get("claim_no") or ""
    vin = servico_data.get("vin") or pecas_data.get("vin") or ""
    cnpj = servico_data.get("cnpj") or pecas_data.get("cnpj") or ""
    service_price = servico_data.get("service_price") or ""
    parts_price = pecas_data.get("parts_price") or ""

    # Fallback via Outros if prices are missing
    outros_data = collect_data_from_outros(root, answers, existing_service_price=service_price, existing_parts_price=parts_price)
    service_price = service_price or outros_data.get("service_price") or ""
    parts_price = parts_price or outros_data.get("parts_price") or ""

    # Overall search mode: if any used answers, mark as answers else blind
    mode_candidates = [servico_data.get("search_mode"), pecas_data.get("search_mode"), outros_data.get("search_mode")]
    search_mode = "answers" if any(m == "answers" for m in mode_candidates) else "blind"

    df = pd.DataFrame([
        {
            "collected_service_price": service_price,
            "collected_parts_price": parts_price,
            "collected_CNPJ": cnpj or "",
            "collected_VIN": vin or "",
            "collected_ClaimNO": claim_no or "",
            "collected_CNPJ2": servico_data.get("cnpj2", ""),
            "search_mode": search_mode,
        }
    ])
    return df


def collect_data_from_pecas(root: str | Path, answers: Answers) -> dict[str, Any]:
    root_path = Path(root).expanduser().resolve()
    pecas_df = categorize_pdfs(root_path)
    pecas_mask = pecas_df["class"] == "Peças"
    pecas_files = [Path(r["file"]) for _, r in pecas_df[pecas_mask].iterrows()]
    data, used = _collect_from_files(pecas_files, answers, want_service_price=False, want_parts_price=True)
    search_mode = "answers" if any(used.values()) else "blind"
    return {
        "claim_no": data["claim_no"],
        "vin": data["vin"],
        "cnpj": data["cnpj"],
        "parts_price": _format_brl_from_cents(data["parts_price_cents"]),
        "search_mode": search_mode,
    }


def collect_data_from_servicos(root: str | Path, answers: Answers) -> dict[str, Any]:
    root_path = Path(root).expanduser().resolve()
    # Avoid using column names with non-identifier characters in .query()
    servicos_df = categorize_pdfs(root_path)
    servicos_mask = servicos_df["class"] == "Serviço"
    servico_files = [Path(r["file"]) for _, r in servicos_df[servicos_mask].iterrows()]
    data, used = _collect_from_files(servico_files, answers, want_service_price=True, want_parts_price=False)
    search_mode = "answers" if any(used.values()) else "blind"
    return {
        "claim_no": data["claim_no"],
        "vin": data["vin"],
        "cnpj": data["cnpj"],
        "service_price": _format_brl_from_cents(data["service_price_cents"]),
        "cnpj2": data["cnpj2"],
        "search_mode": search_mode,
    }


def collect_data_from_outros(
    root: str | Path,
    answers: Answers,
    existing_service_price: Optional[str],
    existing_parts_price: Optional[str],
) -> dict[str, Any]:
    root_path = Path(root).expanduser().resolve()
    outros_df = categorize_pdfs(root_path)
    outros_files = [Path(r["file"]) for _, r in outros_df[outros_df["class"] == "Outros"].iterrows()]

    # We only search for missing prices here, as a fallback
    want_service = not existing_service_price
    want_parts = not existing_parts_price
    if not want_service and not want_parts:
        return {"service_price": existing_service_price or "", "parts_price": existing_parts_price or "", "search_mode": ""}

    data, used = _collect_from_files(outros_files, answers, want_service_price=want_service, want_parts_price=want_parts)
    search_mode = "answers" if any(used.values()) else "blind"
    return {
        "service_price": existing_service_price or _format_brl_from_cents(data["service_price_cents"]) or "",
        "parts_price": existing_parts_price or _format_brl_from_cents(data["parts_price_cents"]) or "",
        "search_mode": search_mode,
    }


__all__ = [
    "categorize_pdfs",
    "collect_data_from_pecas",
    "collect_data_from_servicos",
    "collect_data_from_outros",
    "process_repository",
    "process_filepaths",
    "Answers",
    "PreprocessedData",
    "RegexPreProcessingStrategy",
    "RegexProcessingStrategy",
    "mock_build_answers_from_pdfs",
]


