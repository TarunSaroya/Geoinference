from src.core.evidence_builder import EvidenceBuilder
from src.core.evidence_store import EvidenceStore
from src.core.pipeline import Pipeline
from src.core.services import ServiceContainer
from src.extractors.image_info import ImageInfoExtractor
from src.extractors.ocr import OCRExtractor


class Application:
    def __init__(self) -> None:
        self.services = ServiceContainer()
        self.evidence_store = EvidenceStore()
        self.evidence_builder = EvidenceBuilder()

        self.pipeline = Pipeline()

        self._register_extractors()

    def _register_extractors(self) -> None:
        self.pipeline.register(ImageInfoExtractor())

        self.pipeline.register(OCRExtractor(self.services.ocr))

    def run(self, image):
        results = self.pipeline.run(image)

        self.evidence_store.clear()

        for result in results:
            evidence = self.evidence_builder.build(result)
            self.evidence_store.add_many(evidence)

        return results
