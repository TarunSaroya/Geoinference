from src.reasoning.hypothesis import LocationHypothesis


class ExplanationGenerator:
    def generate(
        self,
        hypothesis: LocationHypothesis,
    ) -> str:
        output = []

        output.append(f"Region: {hypothesis.region}")

        output.append(f"Confidence score: {hypothesis.score:.2f}")

        output.append("\nEvidence:")

        for reason in hypothesis.reasons:
            output.append(f"- {reason}")

        return "\n".join(output)
