from src.core.evidence import Evidence
from src.reasoning.hypothesis import LocationHypothesis


class GeoScorer:
    def analyze(
        self,
        evidence: list[Evidence],
    ) -> list[LocationHypothesis]:
        scores = {}

        brazil_score = 0.0
        brazil_reasons = []

        for item in evidence:
            value = str(item.value).lower()

            # vegetation clue
            if item.evidence_type == "semantic_scene":
                if value == "vegetation":
                    brazil_score += 0.2

                    brazil_reasons.append("Dense vegetation detected")

                if value == "terrain":
                    brazil_score += 0.1

                    brazil_reasons.append("Terrain characteristics detected")

            # OCR clue
            if item.evidence_type == "text":
                text = value

                portuguese_words = [
                    "rua",
                    "avenida",
                    "capim",
                    "rio",
                ]

                if any(word in text for word in portuguese_words):
                    brazil_score += 0.3

                    brazil_reasons.append("Portuguese-like text detected")

        scores["Brazil"] = LocationHypothesis(
            region="Brazil",
            score=brazil_score,
            reasons=brazil_reasons,
        )

        return sorted(
            scores.values(),
            key=lambda x: x.score,
            reverse=True,
        )
