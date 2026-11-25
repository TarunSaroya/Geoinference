from time import perf_counter

from src.core.extractor import BaseExtractor
from src.core.image import ImageData
from src.core.result import ExtractorResult


class Pipeline:
    def __init__(self) -> None:
        self.extractors: list[BaseExtractor] = []

    def register(
        self,
        extractor: BaseExtractor,
    ) -> None:
        self.extractors.append(extractor)

    def run(
        self,
        image: ImageData,
    ) -> list[ExtractorResult]:
        results = []

        for extractor in self.extractors:
            print(f"Running {extractor.name}")

            start = perf_counter()

            result = extractor.extract(image)

            result.execution_time_ms = (perf_counter() - start) * 1000

            results.append(result)

        return results
