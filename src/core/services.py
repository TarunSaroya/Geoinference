from src.services.ocr_service import OCRService


class ServiceContainer:
    """
    Holds long-lived services used by the application.
    """

    def __init__(self) -> None:
        self.ocr = OCRService()
