from src.core.extractor import BaseExtractor
from src.core.image import ImageData
from src.core.result import ExtractorResult


class ImageInfoExtractor(BaseExtractor):
    @property
    def name(self) -> str:
        return "image_info"

    def extract(self, image: ImageData) -> ExtractorResult:
        aspect_ratio = round(image.width / image.height, 3)
        megapixels = round((image.width * image.height) / 1_000_000, 2)

        return ExtractorResult(
            extractor=self.name,
            success=True,
            confidence=1.0,
            data={
                "width": image.width,
                "height": image.height,
                "channels": image.channels,
                "aspect_ratio": aspect_ratio,
                "megapixels": megapixels,
            },
        )
