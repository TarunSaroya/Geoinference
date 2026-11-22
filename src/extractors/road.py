import cv2
import numpy as np

from src.core.extractor import BaseExtractor
from src.core.image import ImageData
from src.core.result import ExtractorResult


class RoadExtractor(BaseExtractor):
    @property
    def name(self) -> str:
        return "road"

    def extract(self, image: ImageData) -> ExtractorResult:
        frame = image.image
        height, width = frame.shape[:2]

        roi_start = int(height * 0.45)
        roi = frame[roi_start:, :]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # --------------------------------------------------
        # 1. ROAD GEOMETRY: EDGE AND LINE DETECTION
        # --------------------------------------------------

        edges = cv2.Canny(gray, 50, 150)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=40,
            maxLineGap=20,
        )

        detected_lines = []

        if lines is not None:
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                dx = x2 - x1
                dy = y2 - y1

                length = float(np.hypot(dx, dy))

                if length == 0:
                    continue

                angle = float(np.degrees(np.arctan2(dy, dx)))

                detected_lines.append(
                    {
                        "x1": int(x1),
                        "y1": int(y1 + roi_start),
                        "x2": int(x2),
                        "y2": int(y2 + roi_start),
                        "angle": angle,
                        "length": length,
                    }
                )

        line_count = len(detected_lines)

        if detected_lines:
            lengths = np.array(
                [line["length"] for line in detected_lines],
                dtype=np.float32,
            )

            angles = np.array(
                [line["angle"] for line in detected_lines],
                dtype=np.float32,
            )

            weighted_angle = float(np.average(angles, weights=lengths))

            mean_length = float(np.mean(lengths))
            total_line_length = float(np.sum(lengths))

        else:
            weighted_angle = 0.0
            mean_length = 0.0
            total_line_length = 0.0

        roi_area = float(roi.shape[0] * roi.shape[1])

        line_density = total_line_length / roi_area if roi_area > 0 else 0.0

        # --------------------------------------------------
        # 2. SURFACE CHARACTERISTICS
        # --------------------------------------------------

        mean_brightness = float(np.mean(gray))
        brightness_std = float(np.std(gray))

        mean_saturation = float(np.mean(hsv[:, :, 1]))
        mean_value = float(np.mean(hsv[:, :, 2]))

        surface_features = {
            "mean_brightness": mean_brightness,
            "brightness_std": brightness_std,
            "mean_saturation": mean_saturation,
            "mean_value": mean_value,
        }

        # These describe the lower image region, not a
        # verified road-only segmentation mask.

        # --------------------------------------------------
        # 3. BRIGHT LANE-MARKING CANDIDATES
        # --------------------------------------------------

        value_channel = hsv[:, :, 2]
        saturation_channel = hsv[:, :, 1]

        bright_mask = cv2.inRange(
            cv2.merge([hsv[:, :, 0], saturation_channel, value_channel]),
            np.array([0, 0, 180], dtype=np.uint8),
            np.array([180, 100, 255], dtype=np.uint8),
        )

        kernel = np.ones((3, 3), dtype=np.uint8)

        cleaned_mask = cv2.morphologyEx(
            bright_mask,
            cv2.MORPH_OPEN,
            kernel,
        )

        component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cleaned_mask,
            connectivity=8,
        )

        marking_candidates = []

        minimum_area = max(15, int(roi_area * 0.00001))

        for component_id in range(1, component_count):
            x = int(stats[component_id, cv2.CC_STAT_LEFT])
            y = int(stats[component_id, cv2.CC_STAT_TOP])
            box_width = int(stats[component_id, cv2.CC_STAT_WIDTH])
            box_height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
            area = int(stats[component_id, cv2.CC_STAT_AREA])

            if area < minimum_area:
                continue

            aspect_ratio = max(box_width, box_height) / max(
                1, min(box_width, box_height)
            )

            # Retain elongated bright regions as candidates.
            if aspect_ratio < 1.5:
                continue

            centroid_x, centroid_y = centroids[component_id]

            marking_candidates.append(
                {
                    "bounding_box": {
                        "x_min": x,
                        "y_min": y + roi_start,
                        "x_max": x + box_width,
                        "y_max": y + roi_start + box_height,
                    },
                    "centroid": {
                        "x": float(centroid_x / width),
                        "y": float((centroid_y + roi_start) / height),
                    },
                    "pixel_area": area,
                    "aspect_ratio": float(aspect_ratio),
                    "orientation": (
                        "vertical" if box_height > box_width else "horizontal"
                    ),
                }
            )

        marking_candidates.sort(
            key=lambda candidate: candidate["pixel_area"],
            reverse=True,
        )

        # Avoid unnecessarily large extractor results.
        marking_candidates = marking_candidates[:30]

        # --------------------------------------------------
        # 4. SPATIAL DISTRIBUTION OF CANDIDATES
        # --------------------------------------------------

        candidate_positions = {
            "left": 0,
            "center": 0,
            "right": 0,
        }

        for candidate in marking_candidates:
            x = candidate["centroid"]["x"]

            if x < 0.33:
                candidate_positions["left"] += 1
            elif x < 0.66:
                candidate_positions["center"] += 1
            else:
                candidate_positions["right"] += 1

        # --------------------------------------------------
        # 5. EXTRACTOR RESULT
        # --------------------------------------------------

        return ExtractorResult(
            extractor=self.name,
            success=True,
            confidence=0.5 if line_count > 0 else 0.0,
            data={
                "road_region": {
                    "x": 0,
                    "y": roi_start,
                    "width": width,
                    "height": height - roi_start,
                },
                "line_count": line_count,
                "mean_line_length": mean_length,
                "weighted_line_angle": weighted_angle,
                "line_density": line_density,
                "lines": detected_lines,
                "surface_features": surface_features,
                "lane_marking_analysis": {
                    "method": "bright_low_saturation_regions",
                    "candidate_count": len(marking_candidates),
                    "candidates": marking_candidates,
                    "spatial_distribution": candidate_positions,
                    "interpretation": ("Potential lane markings; not verified."),
                },
            },
        )
