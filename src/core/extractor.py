from abc import ABC, abstractmethod

from src.core.image import ImageData
from src.core.result import ExtractorResult


class BaseExtractor(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def extract(self, image: ImageData) -> ExtractorResult: ...
