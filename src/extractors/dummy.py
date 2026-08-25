from src.core.extractor import BaseExtractor
from src.core.image import ImageData
from src.core.result import ExtractorResult


class DummyExtractor(BaseExtractor):
    @property
    def name(self) -> str:
        return "dummy"

    def extract(self, image: ImageData) -> ExtractorResult:
        return ExtractorResult(
            extractor=self.name,
            success=True,
            confidence=1.0,
            metadata={"width": image.width, "height": image.height},
        )
