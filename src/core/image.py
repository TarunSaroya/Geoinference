from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ImageData:
    """
    Standard image object passed through the pipeline.
    """

    image: np.ndarray
    width: int
    height: int
    channels: int
    path: str | None = None
