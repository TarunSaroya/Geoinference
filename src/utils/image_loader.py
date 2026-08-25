from pathlib import Path

import cv2

from src.core.image import ImageData


class ImageLoader:
    @staticmethod
    def load(path: str) -> ImageData:
        image = cv2.imread(path)

        if image is None:
            raise FileNotFoundError(f"Unable to load image: {path}")

        height, width, channels = image.shape

        return ImageData(
            image=image,
            width=width,
            height=height,
            channels=channels,
            path=str(Path(path)),
        )
