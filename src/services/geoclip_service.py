from importlib.metadata import version
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

import torch

from src.core.geographic_candidate import GeographicCandidate
from src.core.prediction import GeographicPrediction


class GeoCLIPService:
    SOURCE = "geoclip"
    DEFAULT_TOP_K = 5

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model: Any = None
        self._lock = Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _load_model(self) -> float:
        if self._model is not None:
            return 0.0

        from geoclip import GeoCLIP

        start = perf_counter()

        model = GeoCLIP()
        model.to(self.device)
        model.eval()

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        self._model = model

        return (perf_counter() - start) * 1000

    def predict(
        self,
        image_path: str | Path,
        top_k: int = DEFAULT_TOP_K,
    ) -> GeographicPrediction:
        path = Path(image_path)

        if not path.is_file():
            raise FileNotFoundError(path)

        if not 1 <= top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")

        with self._lock:
            loading_time = self._load_model()

            start = perf_counter()

            with torch.inference_mode():
                coordinates, scores = self._model.predict(
                    str(path),
                    top_k=top_k,
                )

            coordinates = coordinates.detach().cpu()
            scores = scores.detach().cpu()

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            candidates = tuple(
                GeographicCandidate(
                    latitude=float(coordinate[0]),
                    longitude=float(coordinate[1]),
                    score=float(score),
                    rank=rank,
                    source=self.SOURCE,
                )
                for rank, (coordinate, score) in enumerate(
                    zip(coordinates, scores, strict=True),
                    start=1,
                )
            )

            execution_time = (perf_counter() - start) * 1000

        return GeographicPrediction(
            source=self.SOURCE,
            candidates=candidates,
            execution_time_ms=execution_time,
            model_loading_time_ms=loading_time,
            device=str(self.device),
            model_version=version("geoclip"),
            score_type="gallery_softmax_uncalibrated",
        )
