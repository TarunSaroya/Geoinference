from src.core.extractor import BaseExtractor
from src.core.image import ImageData
from src.core.result import ExtractorResult
from src.services.ocr_service import OCRService


class OCRExtractor(BaseExtractor):
    def __init__(self, ocr_service: OCRService) -> None:
        self.ocr_service = ocr_service

    @property
    def name(self) -> str:
        return "ocr"

    def extract(self, image: ImageData) -> ExtractorResult:
        detections = self.ocr_service.read(image.image)

        parsed_detections = []

        for detection in detections:
            bounding_box, text, confidence = detection

            parsed_detections.append(
                {
                    "text": text,
                    "confidence": float(confidence),
                    "bounding_box": bounding_box,
                    "image_width": image.width,
                    "image_height": image.height,
                }
            )

        return ExtractorResult(
            extractor=self.name,
            success=True,
            confidence=1.0,
            data={
                "detections": parsed_detections,
                "count": len(parsed_detections),
            },
        )
