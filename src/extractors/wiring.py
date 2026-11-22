import cv2
import numpy as np

from src.core.extractor import BaseExtractor
from src.core.image import ImageData
from src.core.result import ExtractorResult


class WiringExtractor(BaseExtractor):
    """Detects potential overhead-wire segments using image geometry."""

    @property
    def name(self) -> str:
        return "wiring"

    def extract(self, image: ImageData) -> ExtractorResult:
        frame = image.image
        height, width = frame.shape[:2]

        roi_height = int(height * 0.75)
        roi = frame[:roi_height, :]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=40,
            minLineLength=max(30, int(width * 0.04)),
            maxLineGap=15,
        )

        candidates = []

        if lines is not None:
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                dx = x2 - x1
                dy = y2 - y1

                length = float(np.hypot(dx, dy))

                if length == 0:
                    continue

                angle = float(np.degrees(np.arctan2(dy, dx)))

                # Treat line direction as undirected.
                angle = (angle + 90) % 180 - 90

                # Exclude very steep segments.
                if abs(angle) > 60:
                    continue

                candidates.append(
                    {
                        "start": {
                            "x": int(x1),
                            "y": int(y1),
                        },
                        "end": {
                            "x": int(x2),
                            "y": int(y2),
                        },
                        "normalized_center": {
                            "x": float((x1 + x2) / (2 * width)),
                            "y": float((y1 + y2) / (2 * height)),
                        },
                        "length_pixels": length,
                        "length_normalized": float(length / width),
                        "angle_degrees": angle,
                    }
                )

        candidates.sort(
            key=lambda candidate: candidate["length_pixels"],
            reverse=True,
        )

        total_candidate_count = len(candidates)
        stored_candidates = candidates[:100]

        return ExtractorResult(
            extractor=self.name,
            success=True,
            confidence=0.0,
            data={
                "analysis_region": {
                    "x": 0,
                    "y": 0,
                    "width": width,
                    "height": roi_height,
                },
                "method": "canny_hough_line_candidates",
                "total_candidate_count": total_candidate_count,
                "stored_candidate_count": len(stored_candidates),
                "candidates": stored_candidates,
                "confidence_status": "not_calibrated",
                "interpretation": (
                    "Potential overhead-line segments; not verified electrical wires."
                ),
            },
            warnings=[
                "Line segments may represent branches, roof edges, "
                "or other structures rather than electrical wires."
            ],
        )
