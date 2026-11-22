from src.knowledge.meta_status import build_meta_status


def test_every_catalog_meta_has_status():
    statuses = build_meta_status()

    assert statuses
    assert len(statuses) > 30


def test_implemented_metas_are_explicit():
    statuses = build_meta_status()

    implemented = {
        status.name for status in statuses if status.implementation == "IMPLEMENTED"
    }

    assert "road_surface_characteristics" in implemented
    assert "road_geometry" in implemented
    assert "traffic_sign_presence" in implemented
