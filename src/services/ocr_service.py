import easyocr


class OCRService:
    """
    Service responsible for loading and running EasyOCR.
    """

    def __init__(self) -> None:
        self.reader = easyocr.Reader(
            ["en"],
            gpu=True,
        )

    def read(self, image):
        return self.reader.readtext(image)
