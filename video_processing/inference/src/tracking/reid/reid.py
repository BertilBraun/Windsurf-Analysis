import numpy as np
from typing import List, Sequence
from typing import Protocol

from server.inference.src.util.similarity_helpers import Embedding


class ReID(Protocol):
    def get_features_for_crops(self, crops: List[np.ndarray]) -> Sequence[Embedding]: ...
