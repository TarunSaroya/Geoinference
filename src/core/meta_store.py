from src.core.meta import GeographicMeta


class GeographicMetaStore:
    def __init__(self) -> None:
        self._metas: list[GeographicMeta] = []

    def add(self, meta: GeographicMeta) -> None:
        self._metas.append(meta)

    def add_many(self, metas: list[GeographicMeta]) -> None:
        self._metas.extend(metas)

    def get_all(self) -> list[GeographicMeta]:
        return list(self._metas)

    def get_by_type(self, meta_type: str) -> list[GeographicMeta]:
        return [meta for meta in self._metas if meta.meta_type == meta_type]

    def clear(self) -> None:
        self._metas.clear()

    def __len__(self) -> int:
        return len(self._metas)
