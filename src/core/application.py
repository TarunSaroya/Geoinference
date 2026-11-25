from pathlib import Path
from time import perf_counter

from src.analysis.text_analyzer import TextAnalyzer
from src.core.application_result import ApplicationResult
from src.core.evidence_builder import EvidenceBuilder
from src.core.evidence_store import EvidenceStore
from src.core.geographic_meta_builder import GeographicMetaBuilder
from src.core.image import ImageData
from src.core.meta_store import GeographicMetaStore
from src.core.pipeline import Pipeline
from src.core.services import ServiceContainer
from src.extractors.image_info import ImageInfoExtractor
from src.extractors.ocr import OCRExtractor
from src.extractors.road import RoadExtractor
from src.extractors.scene import SceneExtractor
from src.extractors.semantic_scene import SemanticSceneExtractor
from src.extractors.wiring import WiringExtractor


class Application:
    def __init__(self) -> None:
        self.services = ServiceContainer()
        self.pipeline = Pipeline()

        self.evidence_store = EvidenceStore()
        self.evidence_builder = EvidenceBuilder()

        self.geographic_meta_store = GeographicMetaStore()
        self.geographic_meta_builder = GeographicMetaBuilder()

        self.text_analyzer = TextAnalyzer()

        self._register_extractors()

    def _register_extractors(self) -> None:
        self.pipeline.register(ImageInfoExtractor())
        self.pipeline.register(SceneExtractor())
        self.pipeline.register(SemanticSceneExtractor())
        self.pipeline.register(RoadExtractor())
        self.pipeline.register(OCRExtractor(self.services.ocr))
        self.pipeline.register(WiringExtractor())

    def run(
        self,
        image: ImageData,
        top_k: int = 5,
    ) -> ApplicationResult:
        if image.path is None:
            raise ValueError("Image path is required for geographic prediction")

        image_path = Path(image.path)

        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        start = perf_counter()

        self.evidence_store.clear()
        self.geographic_meta_store.clear()

        results = self.pipeline.run(image)

        for result in results:
            evidence_items = self.evidence_builder.build(result)

            for evidence in evidence_items:
                analyzed_evidence = self.text_analyzer.analyze(evidence)

                self.evidence_store.add(analyzed_evidence)

            geographic_metas = self.geographic_meta_builder.build(result)

            self.geographic_meta_store.add_many(geographic_metas)

        print("Running geographic prediction")

        geographic_prediction = self.services.geoclip.predict(
            image_path,
            top_k=top_k,
        )

        execution_time = (perf_counter() - start) * 1000

        return ApplicationResult(
            extractor_results=tuple(results),
            evidence=tuple(self.evidence_store.get_all()),
            geographic_metas=tuple(self.geographic_meta_store.get_all()),
            geographic_prediction=geographic_prediction,
            execution_time_ms=execution_time,
        )
