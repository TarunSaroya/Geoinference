from src.core.extractor import BaseExtractor
from src.core.image import ImageData


class Pipeline:
    def __init__(self):
        self.extractors: list[BaseExtractor] = []

    def register(self, extractor: BaseExtractor):
        self.extractors.append(extractor)

    def run(self, image: ImageData):
        results = []

        for extractor in self.extractors:
            print(f"Running {extractor.name}")

            results.append(extractor.extract(image))

        return results
