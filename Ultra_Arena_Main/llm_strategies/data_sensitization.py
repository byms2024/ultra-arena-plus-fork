from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
import secrets
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, Any, List

import pandas as pd

# Optional OCR / PDF tooling
try:
    from PIL import Image  # type: ignore
    import pytesseract  # type: ignore
    from pdf2image import convert_from_path  # type: ignore
    _ocr_modules_available = True
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

try:
    import spacy  # type: ignore
    _spacy_available = True
except Exception:
    spacy = None  # type: ignore[assignment]
    _spacy_available = False

# ==============================================================================

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

# Claim number patterns: prefer underscore form; fallback to no-underscore
# Require 'WCN' in the body (claim indicator) and exclude 'WRO' (repair order)
CLAIM_NO_REGEX_STRICT: re.Pattern[str] = re.compile(
    r"BYDAMEBR(?P<body>(?=[A-Za-z0-9]*WCN)(?![A-Za-z0-9]*WRO)[A-Za-z0-9]{8,30})_(?P<suffix>\d{2})",
    re.IGNORECASE,
)
CLAIM_NO_REGEX_FALLBACK: re.Pattern[str] = re.compile(
    r"BYDAMEBR(?P<body>(?=[A-Za-z0-9]*WCN)(?![A-Za-z0-9]*WRO)[A-Za-z0-9]{8,30})(?P<suffix>\d{2})(?!_)",
    re.IGNORECASE,
)

    # For claims, use the two-pass patterns
def _has_claim(v: str) -> bool:
    return bool(CLAIM_NO_REGEX_STRICT.search(v) or CLAIM_NO_REGEX_FALLBACK.search(v))

def _find_claim_matches_with_spans(text: str) -> list[tuple[int, int, str]]:
    matches: list[tuple[int, int, str]] = []
    if not text:
        return matches
    strict_spans: list[tuple[int, int]] = []
    # First pass: strict with underscore
    for m in CLAIM_NO_REGEX_STRICT.finditer(text):
        s, e = m.start(), m.end()
        strict_spans.append((s, e))
        matches.append((s, e, m.group(0)))
    # Second pass: fallback without underscore, only if not overlapping strict
    def _overlaps(s1: int, e1: int, s2: int, e2: int) -> bool:
        return s1 < e2 and s2 < e1
    for m in CLAIM_NO_REGEX_FALLBACK.finditer(text):
        s, e = m.start(), m.end()
        if any(_overlaps(s, e, ss, se) for ss, se in strict_spans):
            continue
        matches.append((s, e, m.group(0)))
    return matches

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
    r"(?<!\d)(?:\d{1,3}(?:[.,]\d{3})+[.,]\d{2}|\d+[.,]\d{2})(?!\d)"
)

CPF_REGEX: re.Pattern[str] = re.compile(r"(?<!\d)\d{3}\.\d{3}\.\d{3}-\d{2}(?!\d)|(?<!\d)\d{11}(?!\d)")
CEP_REGEX: re.Pattern[str] = re.compile(r"(?<!\d)\d{5}-?\d{3}(?!\d)")

# Phone number patterns (Brazil + generic): supports +55, area code, 8-9 digits, and 0800
PHONE_REGEXES: list[re.Pattern[str]] = [
    re.compile(r"(?<!\d)(?:\+?55[\s.-]?)?(?:\(?\d{2}\)?[\s.-]?)?(?:9?\d{4}[\s.-]?\d{4})(?!\d)"),
    re.compile(r"(?<!\d)0?800[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")
]

# Brazilian vehicle plates: old (LLL-NNNN) and Mercosur (LLLNLNN). Optional UF suffix like -DF
PLATE_REGEXES: list[re.Pattern[str]] = [
    re.compile(r"(?<![A-Z0-9])[A-Z]{3}[- ]?\d{4}(?:-[A-Z]{1,2})?(?![A-Z0-9])"),
    re.compile(r"(?<![A-Z0-9])[A-Z]{3}\d[A-Z]\d{2}(?:-[A-Z]{1,2})?(?![A-Z0-9])"),
]

# A loose address heuristic: sequences containing street keywords
ADDRESS_KEYWORDS = [
    "rua", "avenida", "av.", "rodovia", "estrada", "alameda", "travessa", "praça", "praca",
    "bairro", "bloco", "quadra", "lote", "apto", "apartamento", "casa", "nº", "numero", "n°", "cep"
]

# Typical Brazilian first names (normalized, no accents, lowercase)
TYPICAL_BR_FIRST_NAMES: set[str] = {
    "maria","ana","joao","jose","carlos","paulo","pedro","lucas","gabriel","marcos",
    "luiz","felipe","rafael","bruno","gustavo","rodrigo","thiago","diego","matheus",
    "eduardo","andre","fernando","daniel","marcelo","vinicius","leonardo","ricardo","hugo",
    "fabio","caio","vitor","victor","renato","sergio","rogerio","alessandro","antonio",
    "juliana","camila","fernanda","patricia","aline","carla","mariana","tatiana","bianca",
    "leticia","bruna","gabriela","amanda","beatriz","carolina","renata","debora","priscila",
    "luana","luciana","simone","paula","silvia","karina","claudia","natalia","sabrina"
}

_NAME_CONNECTORS: set[str] = {"de","da","do","dos","das","e","d'"}

def _strip_accents(s: str) -> str:
    try:
        import unicodedata
        return "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != 'Mn')
    except Exception:
        return s

def _is_typical_br_person_name(name: str) -> bool:
    tokens_raw = re.findall(r"[A-Za-zÁÂÃÀÉÊÍÓÔÕÚÇáâãàéêíóôõúç]+", name or "")
    if not tokens_raw:
        return False
    tokens_norm = []
    for t in tokens_raw:
        t_norm = _strip_accents(t).lower()
        if t_norm in _NAME_CONNECTORS:
            continue
        tokens_norm.append(t_norm)
    if not tokens_norm:
        return False
    # Require the first token to be a common given name
    return tokens_norm[0] in TYPICAL_BR_FIRST_NAMES

# ==============================================================================

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
# ======================================================================
# ReverseMapStore singleton: holds placeholder->original across the run
class ReverseMapStore:
    _instance: "ReverseMapStore" | None = None

    def __new__(cls) -> "ReverseMapStore":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._map: dict[str, str] = {}
            cls._instance._sensitizationconfig = bool
        return cls._instance

    def add_entries(self, entries: Dict[str, str]) -> None:
        if not entries:
            return
        self._map.update(entries)

    def get_map(self) -> Dict[str, str]:
        return dict(self._map)

    def clear(self) -> None:
        self._map.clear()




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

_spacy_nlp = None
if _spacy_available:
    for model_name in ("pt_core_news_lg", "pt_core_news_md", "pt_core_news_sm"):
        try:
            _spacy_nlp = spacy.load(model_name)  # type: ignore[attr-defined]
            break
        except Exception:
            _spacy_nlp = None

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

def _build_text_hash_maps(values_by_label: dict[str, set[str]]) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """Build reversible maps per label and combined reverse map."""
    per_label_maps: dict[str, dict[str, str]] = {}
    reverse_map: dict[str, str] = {}
    for label, vals in values_by_label.items():
        if not vals:
            per_label_maps[label] = {}
            continue
        # Do not hash organizations
        if label == "ORG":
            per_label_maps[label] = {}
            continue
        to_hash, to_plain = reversible_hash_values(sorted(vals), label)
        per_label_maps[label] = to_hash
        reverse_map.update(to_plain)
    # Populate global store with this batch's reverse map
    try:
        ReverseMapStore().add_entries(reverse_map)
    except Exception:
        pass
    return per_label_maps, reverse_map

def _hash_text_with_maps(text: str, per_label_maps: dict[str, dict[str, str]]) -> str:
    if not text:
        return text
    # Build a unified map: plain -> placeholder
    unified: dict[str, str] = {}
    for m in per_label_maps.values():
        unified.update(m)
    if not unified:
        return text
    # Replace longer strings first to avoid partial overlaps
    sorted_items = sorted(unified.items(), key=lambda kv: len(kv[0]), reverse=True)
    out = text
    for plain, placeholder in sorted_items:
        if not plain:
            continue
        try:
            out = out.replace(plain, placeholder)
        except Exception:
            continue
    return out


# ==============================================================================
def _collect_sensitive_values_from_text(text: str) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {
        "CNPJ": set(),
        "CPF": set(),
        "CEP": set(),
        "VIN": set(),
        "CLAIM": set(),
        "NAME": set(),
        "ADDRESS": set(),
        "PHONE": set(),
        "ORG": set(),
        "PLATE": set(),
    }
    t = text or ""
    for pat in CNPJ_PATTERNS:
        for m in pat.finditer(t):
            digits = re.sub(r"\D", "", m.group(0))
            # if digits in CNPJ_BANNED_DIGITS:
            #     continue
            values["CNPJ"].add(m.group(0))
    # for m in CPF_REGEX.finditer(t):
    #     values["CPF"].add(m.group(0))
    # for m in CEP_REGEX.finditer(t):
    #     values["CEP"].add(m.group(0))
    for m in VIN_REGEX.finditer(t):
        values["VIN"].add(m.group(0))
    # Claim numbers: two-pass detection (underscore form preferred)
    for s, e, val in _find_claim_matches_with_spans(t):
        values["CLAIM"].add(val)
    # phones
    # for pat in PHONE_REGEXES:
    #     for m in pat.finditer(t):
    #         values["PHONE"].add(m.group(0))
    # vehicle plates
    for pat in PLATE_REGEXES:
        for m in pat.finditer(t):
            values["PLATE"].add(m.group(0))
    # address-like lines
    # for line in t.splitlines():
    #     low = line.lower()
    #     if any(k in low for k in ADDRESS_KEYWORDS) and len(line) >= 10:
    #         values["ADDRESS"].add(line)
    # names via spaCy or heuristic, filtered to typical BR person names
    if _spacy_nlp is not None and t:
        try:
            doc = _spacy_nlp(t)
            # for ent in doc.ents:
                # if ent.label_ == "PER":
                #     if _is_typical_br_person_name(ent.text):
                #         values["NAME"].add(ent.text)
                # elif ent.label_ == "ORG":
                #     values["ORG"].add(ent.text)
        except Exception:
            pass
    else:
        for s, e, _ in _heuristic_name_spans(t):
            cand = t[s:e]
        #     if _is_typical_br_person_name(cand):
        #         values["NAME"].add(cand)
        # for s, e, _ in _heuristic_org_spans(t):
        #     values["ORG"].add(t[s:e])
    return values


def _heuristic_name_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    # Allow accented letters; connectors common in PT names
    name_token = r"[A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][a-záâãàéêíóôõúç]{2,}"
    connector = r"(?:\s+(?:de|da|do|dos|das|e|d'))?"
    pattern = re.compile(rf"\b{name_token}(?:{connector}\s+{name_token}){{1,4}}\b")
    org_suffix = re.compile(r"\b(SA|S\.A\.|LTDA|ME|EPP|EIRELI)\b", re.IGNORECASE)
    for m in pattern.finditer(text):
        cand = m.group(0)
        if any(ch.isdigit() for ch in cand):
            continue
        if org_suffix.search(cand):
            continue
        # avoid capturing entire long address-like lines
        if len(cand) > 80:
            continue
        spans.append((m.start(), m.end(), "NAME"))
    # also lines prefixed with common labels
    label_pattern = re.compile(r"\b(nome|solicitante|cliente|responsável)\s*:\s*(.+)", re.IGNORECASE)
    for m in label_pattern.finditer(text):
        val = m.group(2).strip()
        if 3 <= len(val) <= 80 and not any(ch.isdigit() for ch in val):
            start = m.start(2)
            spans.append((start, start + len(val), "NAME"))
    return spans

def _heuristic_org_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    # Organization-like names with common suffixes and keywords
    name_token = r"[A-ZÁÂÃÀÉÊÍÓÔÕÚÇ][\w&\.-ÁÂÃÀÉÊÍÓÔÕÚÇaáâãàéêíóôõúç]{2,}"
    connector = r"(?:\s+(?:de|da|do|dos|das|e|d'))?"
    suffix = r"(?:S\.?A\.?|S/\s*A|LTDA|ME|EPP|EIRELI|INC|LLC|GMBH|S\.?R\.?L\.?)"
    pattern_suffix = re.compile(rf"\b{name_token}(?:{connector}\s+{name_token}){{0,6}}\s+{suffix}\b", re.IGNORECASE)
    for m in pattern_suffix.finditer(text):
        spans.append((m.start(), m.end(), "ORG"))
    # Keywords for public bodies/organizations without suffixes
    keywords = r"(?:Prefeitura|Governo|Minist[eé]rio|Secretaria|Universidade|Banco|Fundação|Fundacao|Instituto|Companhia|Empresa|C[âa]mara|Assembleia|Tribunal)"
    pattern_kw = re.compile(rf"\b{keywords}\s+{name_token}(?:{connector}\s+{name_token}){{0,6}}\b", re.IGNORECASE)
    for m in pattern_kw.finditer(text):
        spans.append((m.start(), m.end(), "ORG"))
    return spans

def _is_word_boundary(text: str, start: int, end: int) -> bool:
    try:
        before_ok = start <= 0 or not (text[start - 1].isalnum())
        after_ok = end >= len(text) or not (text[end].isalnum())
        return bool(before_ok and after_ok)
    except Exception:
        return True


def _find_sensitive_custom_patterns(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    if not text:
        return spans
    # CNPJ patterns (multiple formats)
    for pat in CNPJ_PATTERNS:
        for m in pat.finditer(text):
            try:
                digits = re.sub(r"\D", "", m.group(0))
                # if digits in CNPJ_BANNED_DIGITS:
                #     continue
            except Exception:
                pass
            spans.append((m.start(), m.end(), "CNPJ"))
    # CPF
    # for m in CPF_REGEX.finditer(text):
    #     spans.append((m.start(), m.end(), "CPF"))
    # CEP
    # for m in CEP_REGEX.finditer(text):
    #     spans.append((m.start(), m.end(), "CEP"))
    # VIN
    for m in VIN_REGEX.finditer(text):
        spans.append((m.start(), m.end(), "VIN"))
    # CLAIM: two-pass detection (underscore form preferred)
    for s, e, _ in _find_claim_matches_with_spans(text):
        spans.append((s, e, "CLAIM"))
    # PHONE
    # for pat in PHONE_REGEXES:
    #     for m in pat.finditer(text):
    #         spans.append((m.start(), m.end(), "PHONE"))
    # PLATE
    for pat in PLATE_REGEXES:
        for m in pat.finditer(text):
            spans.append((m.start(), m.end(), "PLATE"))
    return spans

def _filter_spans_for_redaction(
    text: str,
    spans: list[tuple[int, int, str]],
    min_name_tokens: int = 2,
    min_entity_len: int = 5,
) -> list[tuple[int, int, str]]:
    filtered: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int, str]] = set()
    for start, end, label in spans:
        if end <= start:
            continue
        snippet = text[start:end]
        if not snippet or snippet.strip() == "":
            continue
        # Require word boundaries to avoid partial in-word matches
        if not _is_word_boundary(text, start, end):
            continue
        s_len = len(snippet)
        # Length and token requirements by label
        if label == "NAME":
            tokens = [t for t in snippet.split() if t.isalpha()]
            if len(tokens) < max(1, min_name_tokens):
                continue
            if s_len < max(4, min_entity_len):
                continue
        elif label in {"ORG", "LOC", "MISC"}:
            tokens = [t for t in snippet.split() if any(ch.isalpha() for ch in t)]
            if len(tokens) < 2 and s_len < 8:
                continue
        else:
            # For other labels keep as-is, but avoid very short spans
            if s_len < 4:
                continue
        key = (start, end, label)
        if key in seen:
            continue
        seen.add(key)
        filtered.append((start, end, label))
    # Merge overlapping spans preferring longer ones
    filtered.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    result: list[tuple[int, int, str]] = []
    last_end = -1
    for s, e, lab in filtered:
        if not result:
            result.append((s, e, lab))
            last_end = e
            continue
        rs, re, rlab = result[-1]
        if s <= re:  # overlap
            # keep the longer span
            if (e - s) > (re - rs):
                result[-1] = (s, e, lab)
                last_end = e
            continue
        result.append((s, e, lab))
        last_end = e
    return result


def redact_pdf_text(
    input_pdf: str | Path,
    output_pdf: str | Path,
    use_spacy_only_for_names: bool = True,
    enable_heuristic_names: bool = False,
    min_name_tokens: int = 2,
    min_entity_len: int = 5,
) -> None:
    """Best-effort redaction by overlaying rectangles on detected sensitive text.

    Uses regex-based spans and spacy NER (if available) to detect entities.
    """
    input_path = Path(input_pdf).expanduser().resolve()
    output_path = Path(output_pdf).expanduser().resolve()
    if not _pymupdf_available:
        # PyMuPDF is required for in-PDF redaction overlay
        return
    doc = fitz.open(str(input_path))  # type: ignore
    for page in doc:
        try:
            text_page = page.get_text("text")
        except Exception:
            text_page = ""
        # Collect spans from custom patterns: IDs, money, codes, etc.
        spans = []
        if enable_heuristic_names:
            spans.extend(_heuristic_name_spans(text_page or ""))
        # Extend with spaCy NER for names/organizations/locations
        if _spacy_nlp is not None and text_page:
            try:
                nlp_doc = _spacy_nlp(text_page)
                # for ent in nlp_doc.ents:
                    # Portuguese models commonly use PER, LOC, ORG, MISC
                    # if ent.label_ in {"PER", "LOC", "ORG", "MISC"}:
                    #     label = "NAME" if ent.label_ == "PER" else ent.label_
                    #     spans.append((ent.start_char, ent.end_char, label))
            except Exception:
                pass
        elif use_spacy_only_for_names:
            # If requested to use spaCy-only for names but not available, remove heuristic NAME spans
            spans = [s for s in spans if s[2] != "NAME"]
        # Filter spans conservatively to avoid mid-word hits and too-short matches
        spans = _filter_spans_for_redaction(text_page, spans, min_name_tokens=min_name_tokens, min_entity_len=min_entity_len)
        # Draw redaction rectangles for each span (approximate positioning using search_for)
        for start, end, label in spans:
            snippet = text_page[start:end]
            if not snippet.strip():
                continue
            try:
                rects = page.search_for(snippet, quads=False)  # type: ignore[attr-defined]
            except Exception:
                rects = []
            for r in rects:
                try:
                    page.add_redact_annot(r, fill=(0, 0, 0))
                except Exception:
                    try:
                        page.draw_rect(r, color=(0, 0, 0), fill=(0, 0, 0))
                    except Exception:
                        pass
        try:
            page.apply_redactions()
        except Exception:
            pass
    doc.save(str(output_path))
    doc.close()


def redact_folder_pdfs(
    root: str | Path,
    output_root: str | Path,
    use_spacy_only_for_names: bool = True,
    enable_heuristic_names: bool = False,
    min_name_tokens: int = 2,
    min_entity_len: int = 5,
) -> None:
    root_path = Path(root).expanduser().resolve()
    output_path = Path(output_root).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    for pdf in _iter_pdf_files(root_path):
        out_file = output_path / pdf.name
        try:
            redact_pdf_text(
                pdf,
                out_file,
                use_spacy_only_for_names=use_spacy_only_for_names,
                enable_heuristic_names=enable_heuristic_names,
                min_name_tokens=min_name_tokens,
                min_entity_len=min_entity_len,
            )
        except Exception:
            continue


def reversible_hash_values(values: list[str], id_prefix: str) -> tuple[dict[str, str], dict[str, str]]:
    """Create reversible mappings using opaque placeholders with a prefix.

    Returns (to_hash_map, to_plain_map). to_hash_map maps plain->placeholder,
    to_plain_map maps placeholder->plain. Placeholders like PREFIX_<RANDOMHEX>.
    """
    to_hash: dict[str, str] = {}
    to_plain: dict[str, str] = {}
    seen: set[str] = set()
    for v in values:
        key = (v or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        # Generate secure random token (hex) and build placeholder
        token = secrets.token_hex(8).upper()  # 16 hex chars
        placeholder = f"{id_prefix}_{token}"
        # Ensure uniqueness
        while placeholder in to_plain:
            token = secrets.token_hex(8).upper()
            placeholder = f"{id_prefix}_{token}"
        to_hash[key] = placeholder
        to_plain[placeholder] = key
    return to_hash, to_plain


def hash_dataframe_identifiers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return a copy of df with sensitive identifiers replaced by reversible placeholders.

    Returns (hashed_df, reverse_map) where reverse_map maps placeholder->plain.
    Columns considered: collected_CNPJ, collected_CNPJ2, collected_VIN, collected_ClaimNO
    """
    cols = ["collected_CNPJ", "collected_CNPJ2", "collected_VIN", "collected_ClaimNO"]
    present = [c for c in cols if c in df.columns]
    values: list[str] = []
    for c in present:
        values.extend([str(v) for v in df[c].fillna("").tolist() if str(v)])

    # Build maps per type to preserve identifiable prefixes
    cnpjs = [v for v in values if v and v.replace(".", "").replace("/", "").replace("-", "").isdigit() and len(re.sub(r"\D", "", v)) in (14,)]
    vins = [v for v in values if re.fullmatch(VIN_REGEX, v) is not None or v.upper().startswith("L")]
    claims = [v for v in values if v.upper().startswith("BYDAMEBR") or _has_claim(v)]

    cnpj_to_hash, cnpj_reverse = reversible_hash_values(cnpjs, "CNPJ")
    vin_to_hash, vin_reverse = reversible_hash_values(vins, "VIN")
    claim_to_hash, claim_reverse = reversible_hash_values(claims, "CLAIM")

    reverse_map: dict[str, str] = {}
    reverse_map.update(cnpj_reverse)
    reverse_map.update(vin_reverse)
    reverse_map.update(claim_reverse)

    def replace_val(val: Any) -> Any:
        s = str(val) if val is not None else ""
        if s in cnpj_to_hash:
            return cnpj_to_hash[s]
        if s in vin_to_hash:
            return vin_to_hash[s]
        if s in claim_to_hash:
            return claim_to_hash[s]
        return s

    hashed_df = df.copy()
    for c in present:
        hashed_df[c] = hashed_df[c].map(replace_val)
    return hashed_df, reverse_map


def _collect_sensitive_values_from_text(text: str) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {
        "CNPJ": set(),
        "CPF": set(),
        "CEP": set(),
        "VIN": set(),
        "CLAIM": set(),
        "NAME": set(),
        "ADDRESS": set(),
        "PHONE": set(),
        "ORG": set(),
        "PLATE": set(),
    }
    t = text or ""
    for pat in CNPJ_PATTERNS:
        for m in pat.finditer(t):
            digits = re.sub(r"\D", "", m.group(0))
            # if digits in CNPJ_BANNED_DIGITS:
            #     continue
            values["CNPJ"].add(m.group(0))
    # for m in CPF_REGEX.finditer(t):
    #     values["CPF"].add(m.group(0))
    # for m in CEP_REGEX.finditer(t):
    #     values["CEP"].add(m.group(0))
    for m in VIN_REGEX.finditer(t):
        values["VIN"].add(m.group(0))
    # Claim numbers: two-pass detection (underscore form preferred)
    for _, _, val in _find_claim_matches_with_spans(t):
        values["CLAIM"].add(val)
    # phones
    # for pat in PHONE_REGEXES:
    #     for m in pat.finditer(t):
    #         values["PHONE"].add(m.group(0))
    # vehicle plates
    for pat in PLATE_REGEXES:
        for m in pat.finditer(t):
            values["PLATE"].add(m.group(0))
    # address-like lines
    # for line in t.splitlines():
    #     low = line.lower()
    #     if any(k in low for k in ADDRESS_KEYWORDS) and len(line) >= 10:
    #         values["ADDRESS"].add(line)
    # names via spaCy or heuristic
    if _spacy_nlp is not None and t:
        try:
            doc = _spacy_nlp(t)
            # for ent in doc.ents:
            #     if ent.label_ == "PER":
            #         values["NAME"].add(ent.text)
            #     elif ent.label_ == "ORG":
            #         values["ORG"].add(ent.text)
        except Exception:
            pass
    else:
        # for s, e, _ in _heuristic_name_spans(t):
        #     values["NAME"].add(t[s:e])
        for s, e, _ in _heuristic_org_spans(t):
            values["ORG"].add(t[s:e])
    return values


def _build_text_hash_maps(values_by_label: dict[str, set[str]]) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """Build reversible maps per label and combined reverse map."""
    per_label_maps: dict[str, dict[str, str]] = {}
    reverse_map: dict[str, str] = {}
    for label, vals in values_by_label.items():
        if not vals:
            per_label_maps[label] = {}
            continue
        # Do not hash organizations
        if label == "ORG":
            per_label_maps[label] = {}
            continue
        to_hash, to_plain = reversible_hash_values(sorted(vals), label)
        per_label_maps[label] = to_hash
        reverse_map.update(to_plain)
    # Populate global store with this batch's reverse map
    try:
        ReverseMapStore().add_entries(reverse_map)
    except Exception:
        pass
    return per_label_maps, reverse_map


def _hash_text_with_maps(text: str, per_label_maps: dict[str, dict[str, str]]) -> str:
    if not text:
        return text
    # Build a unified map: plain -> placeholder
    unified: dict[str, str] = {}
    for m in per_label_maps.values():
        unified.update(m)
    if not unified:
        return text
    # Replace longer strings first to avoid partial overlaps
    sorted_items = sorted(unified.items(), key=lambda kv: len(kv[0]), reverse=True)
    out = text

    for plain, placeholder in sorted_items:
        if not plain:
            continue
        try:
            out = out.replace(plain, placeholder)
        except Exception:
            continue

    return out

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

        if len(extracted_text) < 10000 and _ocr_available:
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

def redact_and_export_texts(
    root: str | Path,
    output_pdf_root: str | Path,
    output_text_root: str | Path,
    use_spacy_only_for_names: bool = True,
    enable_heuristic_names: bool = False,
    min_name_tokens: int = 2,
    min_entity_len: int = 5,
) -> dict[str, str]:
    """Redact PDFs to output_pdf_root and write hashed text files to output_text_root.

    Returns a dict with paths to a reverse_map CSV saved in the text output folder.
    """
    root_path = Path(root).expanduser().resolve()
    pdf_out = Path(output_pdf_root).expanduser().resolve()
    txt_out = Path(output_text_root).expanduser().resolve()
    pdf_out.mkdir(parents=True, exist_ok=True)
    txt_out.mkdir(parents=True, exist_ok=True)

    files = list(_iter_pdf_files(root_path))

    # First pass: collect sensitive values across all files
    aggregate_values: dict[str, set[str]] = {
        "CNPJ": set(),
        "CPF": set(),
        "CEP": set(),
        "VIN": set(),
        "CLAIM": set(),
        "NAME": set(),
        "ADDRESS": set(),
        "PHONE": set(),
        "ORG": set(),
        "PLATE": set(),
    }

    file_text_cache: dict[Path, str] = {}

    for pdf in files:
        try:
            # Redact PDF
            redact_pdf_text(
                pdf,
                pdf_out / pdf.name,
                use_spacy_only_for_names=use_spacy_only_for_names,
                enable_heuristic_names=enable_heuristic_names,
                min_name_tokens=min_name_tokens,
                min_entity_len=min_entity_len,
            )
        except Exception:
            # proceed with text extraction even if redaction fails
            pass
        try:
            text = extract_text_from_pdf(pdf)
        except Exception:
            text = ""
        file_text_cache[pdf] = text
        vals = _collect_sensitive_values_from_text(text)
        for k, s in vals.items():
            aggregate_values[k].update(s)

    # Build reversible maps
    per_label_maps, reverse_map = _build_text_hash_maps(aggregate_values)

    # Second pass: write hashed text outputs
    for pdf in files:
        text = file_text_cache.get(pdf, "")
        hashed_text = _hash_text_with_maps(text, per_label_maps)
        out_txt = txt_out / (pdf.stem + ".txt")
        try:
            out_txt.write_text(hashed_text, encoding="utf-8")
        except Exception:
            try:
                out_txt.write_text(hashed_text, encoding="latin-1", errors="ignore")
            except Exception:
                continue

    # Save reverse map for recovery (placeholder -> original)
    try:
        import csv
        rev_path = txt_out / "reverse_map.csv"
        with rev_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["placeholder", "original"])
            for placeholder, original in reverse_map.items():
                writer.writerow([placeholder, original])
        return {"reverse_map_csv": str(rev_path)}
    except Exception:
        return {"reverse_map_csv": ""}


def _load_reverse_map_for_files(files: list[Path]) -> dict[str, str]:
    """Search for reverse_map CSVs near the given files and load them into a dict."""
    candidates: list[Path] = []
    seen: set[Path] = set()

    def _add_candidate(p: Path) -> None:
        if p.exists() and p.is_file() and p not in seen:
            seen.add(p)
            candidates.append(p)

    # Scan same and parent directories for reverse_map files
    for f in files:
        try:
            start = f.parent
            # Check up to 5 levels up
            steps = [start] + list(start.parents)[:5]
            for d in steps:
                _add_candidate(d / "reverse_map.csv")
                # Also accept variations like reverse_map_2025-*.csv
                try:
                    for g in d.glob("reverse_map*.csv"):
                        _add_candidate(g)
                except Exception:
                    pass
        except Exception:
            continue

    # Sort by most recent first
    candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)

    reverse_map: dict[str, str] = {}
    import csv
    for csv_path in candidates:
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                for idx, row in enumerate(reader):
                    if not row:
                        continue
                    if idx == 0 and len(row) >= 2 and row[0].lower() == "placeholder":
                        # header row
                        continue
                    if len(row) >= 2:
                        placeholder = (row[0] or "").strip()
                        original = (row[1] or "").strip()
                        if placeholder and original:
                            reverse_map[placeholder] = original
        except Exception:
            continue
    return reverse_map


def resensitize_output(
    hashed_text_file_a: str | Path,
    hashed_text_file_b: str | Path,
) -> None:
    """Resensitize two hashed text files by replacing placeholders using nearby reverse_map CSVs.

    The function looks for reverse_map CSV files in the same directories as the inputs
    (and parent directories), merges them, and applies replacements back to both files.
    """
    files = [Path(hashed_text_file_a).expanduser().resolve(), Path(hashed_text_file_b).expanduser().resolve()]
    files = [f for f in files if f.exists() and f.is_file()]
    if not files:
        return

    # Prefer in-memory store; fallback to nearby CSV discovery
    mem_map = {}
    try:
        mem_map = ReverseMapStore().get_map()
    except Exception:
        mem_map = {}
    file_map = _load_reverse_map_for_files(files)
    reverse_map = {**file_map, **mem_map}
    if not reverse_map:
        return

    # Replace longer placeholders first to avoid partial overlaps
    sorted_items = sorted(reverse_map.items(), key=lambda kv: len(kv[0]), reverse=True)

    for file_path in files:
        # Read file content
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception:
            try:
                content = file_path.read_text(encoding="latin-1", errors="ignore")
            except Exception:
                continue
        
        for placeholder, original in sorted_items:
            if not placeholder:
                continue
            try:
                content = content.replace(placeholder, original)
            except Exception:
                continue

        # Write back
        try:
            file_path.write_text(content, encoding="utf-8")
        except Exception:
            try:
                file_path.write_text(content, encoding="latin-1", errors="ignore")
            except Exception:
                continue

    
    # Clear the in-memory map after using it
    try:
        ReverseMapStore().clear()
    except Exception:
        pass

def soft_resensitize_output(
    hashed_text_file_a: str | Path,
    hashed_text_file_b: str | Path,
) -> None:
    """Resensitize two hashed text files without clearing the in-memory reverse map.

    Uses the in-memory ReverseMapStore if available, falling back to discovering
    reverse_map*.csv files near the provided files. Applies placeholder->original
    replacements to both files, but keeps the ReverseMapStore intact for further use.
    """
    files = [Path(hashed_text_file_a).expanduser().resolve(), Path(hashed_text_file_b).expanduser().resolve()]
    files = [f for f in files if f.exists() and f.is_file()]
    if not files:
        return

    # Prefer in-memory store; fallback to nearby CSV discovery
    mem_map = {}
    try:
        mem_map = ReverseMapStore().get_map()
    except Exception:
        mem_map = {}
    file_map = _load_reverse_map_for_files(files)
    reverse_map = {**file_map, **mem_map}
    if not reverse_map:
        return

    # Replace longer placeholders first to avoid partial overlaps
    sorted_items = sorted(reverse_map.items(), key=lambda kv: len(kv[0]), reverse=True)

    for file_path in files:
        # Read file content
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception:
            try:
                content = file_path.read_text(encoding="latin-1", errors="ignore")
            except Exception:
                continue

        for placeholder, original in sorted_items:
            if not placeholder:
                continue
            try:
                content = content.replace(placeholder, original)
            except Exception:
                continue

        # Write back
        try:
            file_path.write_text(content, encoding="utf-8")
        except Exception:
            try:
                file_path.write_text(content, encoding="latin-1", errors="ignore")
            except Exception:
                continue

def resensitize_data(result):
    """
    Resensitize a dict, list of dicts, or list of (str, dict) tuples by replacing placeholders using ReverseMapStore's map only.
    Handles:
      - dict
      - list of dicts
      - list of (str, dict) tuples
      - list of str
      - single str
    """
    try:
        reverse_map: dict[str, str] = ReverseMapStore().get_map()
    except Exception:
        reverse_map = {}

    if not reverse_map:
        return result

    # Replace longer placeholders first to avoid partial overlaps
    sorted_items: list[tuple[str, str]] = sorted(reverse_map.items(), key=lambda kv: len(kv[0]), reverse=True)

    def _replace_in_obj(obj):
        if isinstance(obj, str):
            out = obj
            for placeholder, original in sorted_items:
                try:
                    out = out.replace(placeholder, original)
                except Exception:
                    continue
            return out
        elif isinstance(obj, dict):
            return {k: _replace_in_obj(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # Check if this is a list of (str, dict) tuples
            if all(isinstance(item, tuple) and len(item) == 2 for item in obj):
                return [(_replace_in_obj(t[0]), _replace_in_obj(t[1])) for t in obj]
            else:
                return [_replace_in_obj(it) for it in obj]
        elif isinstance(obj, tuple) and len(obj) == 2:
            return (_replace_in_obj(obj[0]), _replace_in_obj(obj[1]))
        else:
            return obj

    try:
        return _replace_in_obj(result)
    except Exception:
        return result