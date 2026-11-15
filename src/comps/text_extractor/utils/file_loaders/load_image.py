# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import cairosvg
import pytesseract
import regex
from langchain_community.document_loaders import UnstructuredImageLoader
from PIL import Image, ImageOps

from comps.cores.mega.logger import change_erag_logger_level, get_erag_logger
from comps.text_extractor.utils.file_loaders.abstract_loader import AbstractLoader
from comps.text_extractor.utils.nltk_data import ensure_nltk_data


logger = get_erag_logger(f"{__file__.split('comps/')[1].split('/', 1)[0]}_microservice")
change_erag_logger_level(logger, log_level=os.getenv("ERAG_LOGGER_LEVEL", "INFO"))


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

    def _clean_ocr_text(self, text: str) -> str:
        """Normalize OCR noise: whitespace, punctuation, leading/trailing junk.

        Discards unreadable OCR noise (e.g. artifacts from diagrams or photos)
        by validating alphanumeric character density and requiring at least one
        coherent word.
        """
        if not text:
            return ""
        text = text.strip().replace("—", "-").replace("_", " ")
        text = regex.sub(r"\s+", " ", text).strip()
        if not text:
            return ""

        # Filter standalone single-symbol tokens commonly produced by diagram OCR
        tokens = [t for t in text.split(" ") if t and (t.isalnum() or len(t) > 1)]
        if not tokens:
            return ""

        cleaned = " ".join(tokens).strip()
        non_ws = [c for c in cleaned if not c.isspace()]
        if not non_ws:
            return ""

        # Require that at least 50% of non-whitespace characters are alphanumeric
        alnum_count = sum(c.isalnum() for c in non_ws)
        if alnum_count / len(non_ws) < 0.5:
            return ""

        # Require at least one coherent word with letters and vowels (eng + pol)
        vowels = set("aeiouyAEIOUYąęóĄĘÓ")
        has_coherent_word = any(
            len(t) >= 2 and any(c in vowels for c in t) and any(c.isalpha() for c in t)
            for t in tokens
        )
        if not has_coherent_word:
            return ""

        return cleaned

