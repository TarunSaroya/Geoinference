from src.services.geoclip_service import GeoCLIPService
from src.services.ocr_service import OCRService


class ServiceContainer:
    """
    Holds long-lived services used by the application.
    """

    def __init__(self) -> None:
        self.ocr = OCRService()
        self.geoclip = GeoCLIPService()
