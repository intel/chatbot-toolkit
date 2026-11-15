# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import re
import unicodedata

import cairosvg
import pytesseract
from langchain_community.document_loaders import UnstructuredImageLoader
from PIL import Image, ImageOps

from comps.cores.mega.logger import change_erag_logger_level, get_erag_logger
from comps.text_extractor.utils.file_loaders.abstract_loader import AbstractLoader
from comps.text_extractor.utils.nltk_data import ensure_nltk_data

logger = get_erag_logger(f"{__file__.split('comps/')[1].split('/', 1)[0]}_microservice")
change_erag_logger_level(logger, log_level=os.getenv("ERAG_LOGGER_LEVEL", "INFO"))

_VOWELS = set("aeiouyAEIOUYąęóĄĘÓ")
_MEANINGFUL_STANDALONE = set("€$£¥₹¢%‰+±=≠<>≤≥°~×÷&§©®")
_VALID_SINGLE_LETTERS = set("aAiIwWzZoOuUmMsgVAWNLCFKxyzXYZ")
_COMMON_UNITS_AND_ABBR = {
    "km", "kg", "mg", "mm", "cm", "dm", "ms", "ns", "ps", "db", "hz",
    "khz", "mhz", "ghz", "thz", "kw", "mw", "gw", "ph", "px", "pt",
    "dpi", "rpm", "fps", "mph", "kph", "kbps", "mbps", "gbps",
    "dr", "nr", "mgr", "vs", "st", "nd", "rd", "th", "src", "dst", "pkg",
}


class LoadImage(AbstractLoader):
    def __init__(self, file_path):
        super().__init__(file_path)
        ensure_nltk_data()

    def extract_text(self):
        """Run OCR on the image. Try Unstructured first, fall back to pytesseract."""
        if self.file_type == "svg":
            self.convert_to_png(self.file_path)

        # Try Unstructured OCR
        try:
            docs = UnstructuredImageLoader(self.file_path, strategy="ocr_only", language=["eng", "pol"]).load()
            if docs and (text := docs[0].page_content.strip()):
                return self._clean_ocr_text(text)
        except Exception as e:
            logger.debug(f"OCR (unstructured) failed, falling back to pytesseract: {e}")

        # Fallback to pytesseract
        try:
            with Image.open(self.file_path) as img:
                text = pytesseract.image_to_string(ImageOps.autocontrast(img.convert("RGB")), 
                                                    lang="eng+pol", config="--psm 6")
                return self._clean_ocr_text(text)
        except Exception as e:
            logger.error(f"OCR failed for {self.file_path}: {e}")
            return ""

    def convert_to_png(self, image_path):
        """Convert image file to png file."""
        png_path = image_path.split(".")[0] + ".png"
        cairosvg.svg2png(url=image_path, write_to=png_path)
        self.file_path = png_path
        self.file_type = "png"

    def _is_noise_token(self, token: str) -> bool:
        """Detect if an individual token is OCR noise."""
        if not token:
            return True

        # Standalone single character
        if len(token) == 1:
            if token.isalnum():
                return not (token in _VALID_SINGLE_LETTERS or token.isdigit())
            return not (unicodedata.category(token) == "Sc" or token in _MEANINGFUL_STANDALONE)

        # Strip standard leading/trailing punctuation and symbols
        stripped = token.strip(".,;:!?()[]{}\"'`~-_—–#$€£¥%°§©®")
        if not stripped:
            return token != "..."

        # Invalid internal punctuation/symbols that never occur inside real words or numbers
        if re.search(r"[;!|^~@*{}\\=<>`]", stripped):
            return True

        # Multiple repeated punctuation inside a token (e.g. ';;', '--', '//')
        if re.search(r"[.,/:_-]{2,}", stripped):
            return True

        # Meaningful standalone symbol after stripping
        if len(stripped) == 1 and (unicodedata.category(stripped) == "Sc" or stripped in _MEANINGFUL_STANDALONE):
            return False

        # Numbers and dates (e.g. '12345', '67890', '100', '2024-01-15', '3.14', '100€', '20°C', '95%')
        clean_num = re.sub(r"[°%]+$|°[CFcf]$", "", stripped)
        if re.fullmatch(r"[-+]?\d+([.,/-]\d+)*", clean_num):
            return False

        # Alphanumeric codes / models (e.g. 'XYZ-100', 'Model-X', 'A4', 'E-01', 'B2B', 'MP3')
        if (
            any(c.isdigit() for c in stripped)
            and any(c.isalpha() for c in stripped)
            and re.fullmatch(r"[A-Za-z0-9]+([-_./][A-Za-z0-9]+)*", stripped)
        ):
            return False

        # Alphabetic words, units, and abbreviations
        letters_only = re.sub(r"[-_.'’]", "", stripped)
        if letters_only.isalpha():
            if len(letters_only) == 1 and letters_only in _VALID_SINGLE_LETTERS:
                return False
            # Word with vowels -> valid
            if any(c in _VOWELS for c in letters_only):
                return False
            # Uppercase acronym without vowels (e.g. 'TV', 'CD', 'DVD', 'GPS', 'USB') -> valid
            if letters_only.isupper() and 2 <= len(letters_only) <= 6:
                return False
            # Known units and abbreviations (e.g. 'km', 'kg', 'mg', 'ms', 'dr') -> valid
            return letters_only.lower() not in _COMMON_UNITS_AND_ABBR

        # High ratio of non-alphanumeric characters
        non_ws = [c for c in token if not c.isspace()]
        alnum = sum(c.isalnum() for c in non_ws)
        return alnum / len(non_ws) < 0.5

    def _clean_ocr_text(self, text: str) -> str:
        """Filter OCR noise from extracted image text.

        Filters out diagram and graphic noise at the token level while
        preserving valid words, numeric codes, abbreviations, and meaningful
        symbols (e.g. currency, math operators, units).
        """
        if not text:
            return ""
        text = text.strip().replace("—", "-").replace("_", " ")
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ""

        tokens = text.split(" ")
        n = len(tokens)
        is_noise = [self._is_noise_token(t) for t in tokens]

        valid_tokens = []
        for i, token in enumerate(tokens):
            if is_noise[i]:
                continue

            # If a short number (1-2 digits) is flanked by noise or at a boundary next to noise,
            # it is an isolated OCR noise artifact (e.g. '^~| ;: @# 17 ;;--')
            is_short_num = token.isdigit() and len(token) <= 2
            prev_is_noise = i > 0 and is_noise[i - 1]
            next_is_noise = i < n - 1 and is_noise[i + 1]
            if is_short_num:
                if prev_is_noise and next_is_noise:
                    continue
                if (prev_is_noise and i == n - 1) or (next_is_noise and i == 0):
                    continue

            valid_tokens.append(token)

        if not valid_tokens:
            return ""

        # Must contain at least one substantial token:
        # a word with vowels, an acronym, a recognized unit/abbreviation, or a number
        has_substantial = any(
            (any(c in _VOWELS for c in t) and any(c.isalpha() for c in t))
            or (t.isupper() and len(t) >= 2 and t.isalpha())
            or (t.lower() in _COMMON_UNITS_AND_ABBR)
            or any(c.isdigit() for c in t)
            for t in valid_tokens
        )
        if not has_substantial:
            return ""

        return " ".join(valid_tokens).strip()

