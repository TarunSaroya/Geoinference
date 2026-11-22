from src.analysis.text_analyzer import TextAnalyzer
from src.core.evidence import Evidence


def test_ui_text_is_rejected() -> None:
    evidence = Evidence(
        evidence_type="text",
        value="GEOGUESSR",
        confidence=0.99,
        source="ocr",
    )

    result = TextAnalyzer().analyze(evidence)

    assert result.geographic_relevance == "none"
    assert result.metadata["classification"] == "ui_text"


def test_low_confidence_text_is_rejected() -> None:
    evidence = Evidence(
        evidence_type="text",
        value="AftA",
        confidence=0.01,
        source="ocr",
    )

    result = TextAnalyzer().analyze(evidence)

    assert result.geographic_relevance == "low"
    assert result.metadata["classification"] == "low_confidence"


def test_potential_location_text_is_kept() -> None:
    evidence = Evidence(
        evidence_type="text",
        value="GROSSO",
        confidence=0.87,
        source="ocr",
    )

    result = TextAnalyzer().analyze(evidence)

    assert result.geographic_relevance == "potential"
    assert result.metadata["classification"] == "possible_location_text"
