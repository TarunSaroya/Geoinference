from src.core.evidence import Evidence
from src.core.evidence_store import EvidenceStore


def test_evidence_store_add_and_retrieve() -> None:
    store = EvidenceStore()

    evidence = Evidence(
        evidence_type="text",
        value="GROSSO",
        confidence=0.87,
        source="ocr",
    )

    store.add(evidence)

    assert len(store) == 1
    assert store.get_all()[0] == evidence


def test_evidence_store_filters_by_source() -> None:
    store = EvidenceStore()

    store.add(
        Evidence(
            evidence_type="text",
            value="GROSSO",
            confidence=0.87,
            source="ocr",
        )
    )

    store.add(
        Evidence(
            evidence_type="road",
            value="left",
            confidence=0.91,
            source="road",
        )
    )

    ocr_evidence = store.get_by_source("ocr")

    assert len(ocr_evidence) == 1
    assert ocr_evidence[0].value == "GROSSO"


def test_evidence_store_filters_by_type() -> None:
    store = EvidenceStore()

    store.add(
        Evidence(
            evidence_type="text",
            value="Brazil",
            confidence=0.95,
            source="ocr",
        )
    )

    store.add(
        Evidence(
            evidence_type="text",
            value="GROSSO",
            confidence=0.87,
            source="ocr",
        )
    )

    assert len(store.get_by_type("text")) == 2
