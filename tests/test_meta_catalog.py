from src.knowledge.meta_catalog import META_CATALOG


def test_meta_catalog_names_are_unique():
    names = [meta.name for meta in META_CATALOG]

    assert len(names) == len(set(names))


def test_meta_catalog_has_required_fields():
    for meta in META_CATALOG:
        assert meta.name
        assert meta.category
        assert meta.description
        assert meta.status
        assert meta.priority


def test_meta_catalog_has_expected_categories():
    categories = {meta.category for meta in META_CATALOG}

    assert "road" in categories
    assert "infrastructure" in categories
    assert "signs" in categories
    assert "text" in categories
    assert "vehicles" in categories
    assert "architecture" in categories
    assert "environment" in categories
    assert "street_furniture" in categories
    assert "imagery" in categories
    assert "geometry" in categories
    assert "learned_prior" in categories
