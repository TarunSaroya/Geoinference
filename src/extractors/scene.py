import cv2
import numpy as np

from src.core.extractor import BaseExtractor
from src.core.image import ImageData
from src.core.result import ExtractorResult


class SceneExtractor(BaseExtractor):
    """
    Extracts coarse visual scene properties from an image.
    """

    @property
    def name(self) -> str:
        return "scene"

    def extract(self, image: ImageData) -> ExtractorResult:
        frame = image.image

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))

        edges = cv2.Canny(gray, 100, 200)
        edge_ratio = float(np.count_nonzero(edges) / edges.size)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        saturation = float(np.mean(hsv[:, :, 1]))

        return ExtractorResult(
            extractor=self.name,
            success=True,
            confidence=1.0,
            data={
                "brightness": brightness,
                "contrast": contrast,
                "edge_ratio": edge_ratio,
                "saturation": saturation,
            },
        )
