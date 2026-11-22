from src.analysis.text_analyzer import TextAnalyzer
from src.core.evidence_builder import EvidenceBuilder
from src.core.evidence_store import EvidenceStore
from src.core.geographic_meta_builder import GeographicMetaBuilder
from src.core.meta_store import GeographicMetaStore
from src.core.pipeline import Pipeline
from src.core.services import ServiceContainer
from src.extractors.image_info import ImageInfoExtractor
from src.extractors.ocr import OCRExtractor
from src.extractors.road import RoadExtractor
from src.extractors.scene import SceneExtractor
from src.extractors.semantic_scene import SemanticSceneExtractor
from src.extractors.wiring import WiringExtractor
from src.reasoning.explanation import ExplanationGenerator
from src.reasoning.scorer import GeoScorer


class Application:
    def __init__(self) -> None:
        self.services = ServiceContainer()

        self.pipeline = Pipeline()

        self.evidence_store = EvidenceStore()
        self.evidence_builder = EvidenceBuilder()
        self.text_analyzer = TextAnalyzer()

        self._register_extractors()

    def _register_extractors(self) -> None:
        self.pipeline.register(ImageInfoExtractor())
        self.pipeline.register(SceneExtractor())

        self.pipeline.register(SemanticSceneExtractor())

        self.pipeline.register(RoadExtractor())

        self.pipeline.register(OCRExtractor(self.services.ocr))
        self.geo_scorer = GeoScorer()
        self.explanation = ExplanationGenerator()
        self.geographic_meta_store = GeographicMetaStore()
        self.geographic_meta_builder = GeographicMetaBuilder()
        self.pipeline.register(WiringExtractor())

    def run(self, image):
        results = self.pipeline.run(image)

        for result in results:
            evidence_items = self.evidence_builder.build(result)

            for evidence in evidence_items:
                self.evidence_store.add(evidence)

            geographic_metas = self.geographic_meta_builder.build(result)

            for meta in geographic_metas:
                self.geographic_meta_store.add(meta)

        hypotheses = self.geo_scorer.analyze(self.evidence_store.get_all())

        for hypothesis in hypotheses:
            print("\n=== LOCATION HYPOTHESIS ===")
            print(self.explanation.generate(hypothesis))

        return results
