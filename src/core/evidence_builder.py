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
            "scene": self._build_scene_evidence,
            "road": self._build_road_evidence,
            "semantic_scene": self._build_semantic_scene_evidence,
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
                        "image_width": detection["image_width"],
                        "image_height": detection["image_height"],
                    },
                )
            )

        return evidence

    def _build_scene_evidence(
        self,
        result: ExtractorResult,
    ) -> list[Evidence]:
        evidence: list[Evidence] = []

        for key, value in result.data.items():
            evidence.append(
                Evidence(
                    evidence_type="scene",
                    value=value,
                    confidence=result.confidence,
                    source="scene",
                    metadata={
                        "feature": key,
                    },
                )
            )

        return evidence

    def _build_road_evidence(
        self,
        result: ExtractorResult,
    ) -> list[Evidence]:
        return [
            Evidence(
                evidence_type="road",
                value=result.data,
                confidence=result.confidence,
                source="road",
                metadata={
                    "feature": "road_geometry",
                },
            )
        ]

    def _build_semantic_scene_evidence(
        self,
        result: ExtractorResult,
    ) -> list[Evidence]:
        evidence: list[Evidence] = []

        classes = result.data.get("classes", {})

        for class_name, statistics in classes.items():
            coverage = float(statistics["coverage"])
            confidence = float(statistics["confidence"])

            evidence.append(
                Evidence(
                    evidence_type="semantic_scene",
                    value=class_name,
                    confidence=confidence,
                    source="semantic_scene",
                    geographic_relevance="contextual",
                    metadata={
                        "coverage": coverage,
                        "pixel_count": statistics["pixel_count"],
                        "class_id": statistics["class_id"],
                        "bounding_box": statistics["bounding_box"],
                        "centroid": statistics["centroid"],
                        "reason": (
                            "Semantic segmentation identified this scene category "
                            "and measured its spatial distribution."
                        ),
                    },
                )
            )

        return evidence
