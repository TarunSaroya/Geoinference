from time import perf_counter
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    SegformerForSemanticSegmentation,
)

from src.core.extractor import BaseExtractor
from src.core.image import ImageData
from src.core.result import ExtractorResult


class SemanticSceneExtractor(BaseExtractor):
    """
    Extract semantic classes and spatial object candidates.

    Uses SegFormer trained on Cityscapes.
    Connected components are candidate regions, not verified
    individual object detections.
    """

    MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"

    MIN_COMPONENT_AREA = {
        "pole": 12,
        "traffic sign": 20,
        "traffic light": 12,
        "building": 100,
    }

    MAX_OBJECTS_PER_CLASS = 100

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = AutoImageProcessor.from_pretrained(self.MODEL_NAME)

        self.model = SegformerForSemanticSegmentation.from_pretrained(self.MODEL_NAME)

        self.model.to(self.device)
        self.model.eval()

    @property
    def name(self) -> str:
        return "semantic_scene"

    def extract(self, image: ImageData) -> ExtractorResult:
        start = perf_counter()

        rgb = cv2.cvtColor(image.image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        inputs = self.processor(
            images=pil_image,
            return_tensors="pt",
        )

        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)

            logits = torch.nn.functional.interpolate(
                outputs.logits,
                size=(image.height, image.width),
                mode="bilinear",
                align_corners=False,
            )

            probabilities = torch.softmax(logits, dim=1)
            confidence, mask = torch.max(probabilities, dim=1)

        mask = mask[0].cpu().numpy()
        confidence = confidence[0].cpu().numpy()

        class_statistics: dict[str, Any] = {}
        object_instances: dict[str, Any] = {}

        for class_id, class_name in self.model.config.id2label.items():
            class_id = int(class_id)

            class_mask = mask == class_id
            pixel_count = int(np.count_nonzero(class_mask))

            if pixel_count == 0:
                continue

            coverage = pixel_count / mask.size
            mean_confidence = float(confidence[class_mask].mean())

            ys, xs = np.where(class_mask)

            class_statistics[class_name] = {
                "class_id": class_id,
                "pixel_count": pixel_count,
                "coverage": float(coverage),
                "confidence": mean_confidence,
                "bounding_box": {
                    "x_min": int(xs.min()),
                    "y_min": int(ys.min()),
                    "x_max": int(xs.max()) + 1,
                    "y_max": int(ys.max()) + 1,
                },
                "centroid": {
                    "x": float(xs.mean() / image.width),
                    "y": float(ys.mean() / image.height),
                },
            }

            minimum_area = self.MIN_COMPONENT_AREA.get(class_name)

            if minimum_area is not None:
                object_instances[class_name] = self._extract_components(
                    class_mask=class_mask,
                    confidence_map=confidence,
                    image_width=image.width,
                    image_height=image.height,
                    minimum_area=minimum_area,
                )

        execution_time_ms = (perf_counter() - start) * 1000

        return ExtractorResult(
            extractor=self.name,
            success=True,
            confidence=1.0,
            data={
                "classes": class_statistics,
                "object_instances": object_instances,
                "image_width": image.width,
                "image_height": image.height,
                "device": str(self.device),
                "model_name": self.MODEL_NAME,
            },
            execution_time_ms=execution_time_ms,
        )

    def _extract_components(
        self,
        class_mask: np.ndarray,
        confidence_map: np.ndarray,
        image_width: int,
        image_height: int,
        minimum_area: int,
    ) -> dict[str, Any]:
        binary_mask = class_mask.astype(np.uint8)

        component_count, component_labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                binary_mask,
                connectivity=8,
            )
        )

        components: list[dict[str, Any]] = []

        for component_id in range(1, component_count):
            x = int(stats[component_id, cv2.CC_STAT_LEFT])
            y = int(stats[component_id, cv2.CC_STAT_TOP])
            width = int(stats[component_id, cv2.CC_STAT_WIDTH])
            height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
            area = int(stats[component_id, cv2.CC_STAT_AREA])

            if area < minimum_area:
                continue

            centroid_x, centroid_y = centroids[component_id]

            region_labels = component_labels[
                y : y + height,
                x : x + width,
            ]

            region_confidence = confidence_map[
                y : y + height,
                x : x + width,
            ]

            component_confidence = region_confidence[region_labels == component_id]

            aspect_ratio = max(width, height) / max(
                1,
                min(width, height),
            )

            if height > width:
                orientation = "vertical"
            elif width > height:
                orientation = "horizontal"
            else:
                orientation = "approximately_square"

            components.append(
                {
                    "component_id": component_id,
                    "bounding_box": {
                        "x_min": x,
                        "y_min": y,
                        "x_max": x + width,
                        "y_max": y + height,
                    },
                    "centroid": {
                        "x": float(centroid_x / image_width),
                        "y": float(centroid_y / image_height),
                    },
                    "width": width,
                    "height": height,
                    "pixel_area": area,
                    "aspect_ratio": float(aspect_ratio),
                    "orientation": orientation,
                    "coverage": float(area / (image_width * image_height)),
                    "mean_confidence": float(component_confidence.mean()),
                }
            )

        components.sort(
            key=lambda component: component["pixel_area"],
            reverse=True,
        )

        total_candidate_count = len(components)
        stored_candidates = components[: self.MAX_OBJECTS_PER_CLASS]

        return {
            "method": "semantic_mask_connected_components",
            "total_candidate_count": total_candidate_count,
            "stored_candidate_count": len(stored_candidates),
            "minimum_component_area": minimum_area,
            "candidates": stored_candidates,
            "interpretation": (
                "Connected semantic regions; individual objects "
                "are not independently verified."
            ),
        }
