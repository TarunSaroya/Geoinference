from src.core.evidence import Evidence
from src.core.result import ExtractorResult


class EvidenceBuilder:
    """
    Converts raw extractor results into structured geographic evidence.
    """

    def build(self, result: ExtractorResult) -> list[Evidence]:
        if not result.success:
            return []

        builders = {
            "ocr": self._build_ocr_evidence,
        }

        builder = builders.get(result.extractor)

        if builder is None:
            return []

        return builder(result)

    def _build_ocr_evidence(
        self,
        result: ExtractorResult,
    ) -> list[Evidence]:
        evidence: list[Evidence] = []

        for detection in result.data.get("detections", []):
            confidence = float(detection["confidence"])

            evidence.append(
                Evidence(
                    evidence_type="text",
                    value=detection["text"],
                    confidence=confidence,
                    source="ocr",
                    location={
                        "bounding_box": detection["bounding_box"],
                    },
                )
            )

        return evidence
