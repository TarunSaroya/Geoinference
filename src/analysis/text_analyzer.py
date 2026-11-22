import re
from difflib import SequenceMatcher

from src.core.evidence import Evidence


class TextAnalyzer:
    """
    Classifies OCR detections into UI, noise, or geographic candidates.
    """

    UI_TERMS = {
        "geoguessr",
        "round",
        "score",
        "world",
        "total",
        "google",
    }

    UI_PATTERNS = (
        r"^round\s\*\d+$",
        r"^\d+/\d+$",
        r"^\d[\d,.]\*$",
        r"^map\s+data\b",
        r"^terms?\b.\*\buse$",
        r"^termi\b.\*\buse$",
        r"^place\s+your\s+pin\b",
        r"^0l\s+use$",
    )

    def analyze(self, evidence: Evidence) -> Evidence:
        if evidence.source != "ocr":
            return evidence

        text = str(evidence.value).strip()
        normalized = re.sub(r"\s+", " ", text.lower())

        if not normalized:
            return self._classify(
                evidence,
                "noise",
                "none",
                "OCR returned empty text.",
            )

        if self._looks_like_ui(normalized):
            return self._classify(
                evidence,
                "ui_text",
                "none",
                "Text matches GeoGuessr or map interface text.",
            )

        if evidence.confidence < 0.10:
            return self._classify(
                evidence,
                "low_confidence",
                "low",
                "OCR confidence is too low for reliable geographic use.",
            )

        if self._looks_like_noise(normalized, evidence.confidence):
            return self._classify(
                evidence,
                "noise",
                "none",
                "OCR output appears too noisy to provide reliable geographic evidence.",
            )

        return self._classify(
            evidence,
            "possible_location_text",
            "potential",
            "Text is not recognized as interface text and may contain "
            "location, language, or regional information.",
        )

    @classmethod
    def _looks_like_ui(cls, text: str) -> bool:
        if text in cls.UI_TERMS:
            return True

        if any(re.match(pattern, text) for pattern in cls.UI_PATTERNS):
            return True

        # OCR frequently corrupts short UI words such as TOTAL.
        if len(text) <= 8:
            for term in cls.UI_TERMS:
                similarity = SequenceMatcher(
                    None,
                    text,
                    term,
                ).ratio()

                if similarity >= 0.80:
                    return True

        return False

    @staticmethod
    def _looks_like_noise(text: str, confidence: float) -> bool:
        letters = [character for character in text if character.isalpha()]
        digits = [character for character in text if character.isdigit()]

        if not letters:
            return True

        if len(letters) <= 3 and confidence < 0.65:
            return True

        if letters and digits and confidence < 0.40:
            return True

        punctuation_count = sum(
            not character.isalnum() and not character.isspace() for character in text
        )

        if punctuation_count >= 2 and confidence < 0.50:
            return True

        return False

    @staticmethod
    def _classify(
        evidence: Evidence,
        classification: str,
        strength: str,
        reason: str,
    ) -> Evidence:
        evidence.geographic_relevance = strength
        evidence.metadata["classification"] = classification
        evidence.metadata["reason"] = reason

        return evidence
