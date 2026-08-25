from src.core.evidence import Evidence


class EvidenceStore:
    """
    Stores all evidence generated during an analysis run.
    """

    def __init__(self) -> None:
        self._evidence: list[Evidence] = []

    def add(self, evidence: Evidence) -> None:
        self._evidence.append(evidence)

    def add_many(self, evidence: list[Evidence]) -> None:
        self._evidence.extend(evidence)

    def get_all(self) -> list[Evidence]:
        return list(self._evidence)

    def get_by_source(self, source: str) -> list[Evidence]:
        return [evidence for evidence in self._evidence if evidence.source == source]

    def get_by_type(self, evidence_type: str) -> list[Evidence]:
        return [
            evidence
            for evidence in self._evidence
            if evidence.evidence_type == evidence_type
        ]

    def clear(self) -> None:
        self._evidence.clear()

    def __len__(self) -> int:
        return len(self._evidence)
