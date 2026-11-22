from dataclasses import dataclass

from src.knowledge.meta_catalog import META_CATALOG


@dataclass(frozen=True, slots=True)
class MetaStatus:
    name: str
    category: str
    priority: str
    status: str
    implementation: str


#
IMPLEMENTED_METAS = {
    "road_geometry",
    "road_surface_characteristics",
    "lane_marking_candidates",
    "lane_marking_distribution",
    "traffic_sign_presence",
    "utility_pole_candidates",
    "overhead_wire_candidates",
}


def build_meta_status() -> list[MetaStatus]:
    statuses = []

    for meta in META_CATALOG:
        if meta.name in IMPLEMENTED_METAS:
            implementation = "IMPLEMENTED"
        elif meta.status == "AVAILABLE":
            implementation = "AVAILABLE"
        else:
            implementation = "PLANNED"

        statuses.append(
            MetaStatus(
                name=meta.name,
                category=meta.category,
                priority=meta.priority,
                status=meta.status,
                implementation=implementation,
            )
        )

    return statuses
