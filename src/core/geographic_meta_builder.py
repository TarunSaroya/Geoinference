from typing import Any

from src.core.meta import GeographicMeta
from src.core.result import ExtractorResult


class GeographicMetaBuilder:
    def build(self, result: ExtractorResult) -> list[GeographicMeta]:
        if not result.success:
            return []

        builders = {
            "semantic_scene": self._build_semantic_scene,
            "road": self._build_road,
            "wiring": self._build_wiring,
        }

        builder = builders.get(result.extractor)

        if builder is None:
            return []

        return builder(result)

    def _build_semantic_scene(
        self,
        result: ExtractorResult,
    ) -> list[GeographicMeta]:
        metas = []

        classes = result.data.get("classes", {})

        for class_name, statistics in classes.items():
            confidence = float(statistics["confidence"])
            coverage = float(statistics["coverage"])

            spatial = {
                "bounding_box": statistics["bounding_box"],
                "centroid": statistics["centroid"],
            }

            presence_definitions = {
                "road": (
                    "road_presence",
                    "road surface detected",
                    "medium",
                ),
                "vegetation": (
                    "vegetation_presence",
                    "vegetation detected",
                    "medium",
                ),
                "terrain": (
                    "terrain_presence",
                    "terrain detected",
                    "high",
                ),
                "building": (
                    "building_presence",
                    "built structures detected",
                    "medium",
                ),
                "pole": (
                    "pole_presence",
                    "roadside or utility poles detected",
                    "high",
                ),
                "traffic sign": (
                    "traffic_sign_presence",
                    "traffic signage detected",
                    "high",
                ),
                "traffic light": (
                    "traffic_light_presence",
                    "traffic light detected",
                    "medium",
                ),
            }

            definition = presence_definitions.get(class_name)

            if definition is None:
                continue

            meta_type, observation, relevance = definition

            metas.append(
                self._create_presence_meta(
                    meta_type=meta_type,
                    observation=observation,
                    statistics=statistics,
                    confidence=confidence,
                    coverage=coverage,
                    relevance=relevance,
                    spatial=spatial,
                )
            )

        metas.extend(self._build_utility_poles(result))

        return metas

    def _create_presence_meta(
        self,
        meta_type: str,
        observation: str,
        statistics: dict[str, Any],
        confidence: float,
        coverage: float,
        relevance: str,
        spatial: dict[str, Any],
    ) -> GeographicMeta:
        centroid = spatial["centroid"]

        return GeographicMeta(
            meta_type=meta_type,
            observation=observation,
            confidence=confidence,
            geographic_relevance=relevance,
            source="semantic_scene",
            spatial=spatial,
            metadata={
                "coverage": coverage,
                "vertical_position": self._vertical_position(centroid["y"]),
                "horizontal_position": self._horizontal_position(centroid["x"]),
                "pixel_count": statistics["pixel_count"],
                "class_id": statistics["class_id"],
            },
        )

    def _vertical_position(self, normalized_y: float) -> str:
        if normalized_y < 0.33:
            return "upper"

        if normalized_y < 0.66:
            return "middle"

        return "lower"

    def _horizontal_position(self, normalized_x: float) -> str:
        if normalized_x < 0.33:
            return "left"

        if normalized_x < 0.66:
            return "center"

        return "right"

    def _build_utility_poles(
        self,
        result: ExtractorResult,
    ) -> list[GeographicMeta]:
        pole_data = result.data.get(
            "object_instances",
            {},
        ).get("pole")

        if not pole_data:
            return []

        candidates = pole_data.get("candidates", [])

        return [
            GeographicMeta(
                meta_type="utility_pole_candidates",
                observation=("Individual pole-like semantic regions detected"),
                confidence=0.0,
                geographic_relevance="high",
                source="semantic_scene",
                spatial=candidate.get("bounding_box"),
                metadata={
                    **candidate,
                    "detection_method": pole_data.get("method"),
                    "confidence_status": "not_calibrated",
                    "interpretation": (
                        "Pole-class segmentation candidates; "
                        "utility-pole identity is not verified."
                    ),
                },
            )
            for candidate in candidates
        ]

    def _build_wiring(
        self,
        result: ExtractorResult,
    ) -> list[GeographicMeta]:
        candidates = result.data.get("candidates", [])

        metas = []

        for candidate in candidates:
            start = candidate["start"]
            end = candidate["end"]

            metas.append(
                GeographicMeta(
                    meta_type="overhead_wire_candidates",
                    observation=("Potential overhead-line segment detected"),
                    confidence=0.0,
                    geographic_relevance="high",
                    source="wiring",
                    spatial={
                        "start": start,
                        "end": end,
                    },
                    metadata={
                        "normalized_center": candidate["normalized_center"],
                        "length_pixels": candidate["length_pixels"],
                        "length_normalized": candidate["length_normalized"],
                        "angle_degrees": candidate["angle_degrees"],
                        "detection_method": result.data.get("method"),
                        "confidence_status": "not_calibrated",
                        "interpretation": (
                            "Geometric line candidate; not verified wiring."
                        ),
                    },
                )
            )

        return metas

    def _build_road(
        self,
        result: ExtractorResult,
    ) -> list[GeographicMeta]:
        data = result.data

        metas = [
            GeographicMeta(
                meta_type="road_geometry",
                observation="road geometry detected",
                confidence=result.confidence,
                geographic_relevance="high",
                source="road",
                spatial=data.get("road_region"),
                metadata={
                    "line_count": data.get("line_count", 0),
                    "mean_line_length": data.get(
                        "mean_line_length",
                        0.0,
                    ),
                    "weighted_line_angle": data.get(
                        "weighted_line_angle",
                        0.0,
                    ),
                    "line_density": data.get(
                        "line_density",
                        0.0,
                    ),
                },
            )
        ]

        # ----------------------------------------------
        # ROAD SURFACE CHARACTERISTICS
        # ----------------------------------------------

        surface = data.get("surface_features")

        if surface:
            metas.append(
                GeographicMeta(
                    meta_type="road_surface_characteristics",
                    observation=("Lower image region surface characteristics measured"),
                    confidence=0.0,
                    geographic_relevance="medium",
                    source="road",
                    spatial=data.get("road_region"),
                    metadata={
                        **surface,
                        "measurement_scope": ("lower_image_region_not_verified_road"),
                        "confidence_status": "not_calibrated",
                    },
                )
            )

        # ----------------------------------------------
        # LANE MARKING CANDIDATES
        # ----------------------------------------------

        marking_analysis = data.get("lane_marking_analysis")

        if marking_analysis:
            candidates = marking_analysis.get("candidates", [])

            metas.append(
                GeographicMeta(
                    meta_type="lane_marking_candidates",
                    observation=(
                        "Potential bright elongated road-region markings detected"
                    ),
                    confidence=0.0,
                    geographic_relevance="high",
                    source="road",
                    spatial=data.get("road_region"),
                    metadata={
                        "candidate_count": len(candidates),
                        "candidates": candidates,
                        "detection_method": marking_analysis.get("method"),
                        "interpretation": (
                            "Candidates only; not verified lane markings"
                        ),
                        "confidence_status": "not_calibrated",
                    },
                )
            )

            # ------------------------------------------
            # LANE MARKING SPATIAL DISTRIBUTION
            # ------------------------------------------

            distribution = marking_analysis.get(
                "spatial_distribution",
                {},
            )

            metas.append(
                GeographicMeta(
                    meta_type="lane_marking_distribution",
                    observation=(
                        "Spatial distribution of bright marking candidates measured"
                    ),
                    confidence=0.0,
                    geographic_relevance="medium",
                    source="road",
                    spatial=data.get("road_region"),
                    metadata={
                        "distribution": distribution,
                        "candidate_count": len(candidates),
                        "confidence_status": "not_calibrated",
                    },
                )
            )

        return metas
