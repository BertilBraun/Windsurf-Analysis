import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from typing import Sequence, Union
from pathlib import Path

from inference.src.util.similarity_helpers import VectorEmbedding


class ReIDOSNet:
    def __init__(self, model_path: Union[str, Path], device: str | None = None):
        from torchreid.reid.utils import FeatureExtractor
        from torchvision import transforms

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        self.device = torch.device(device)
        self.half = device == 'cuda'

        self.extractor = FeatureExtractor(
            model_name='osnet_ain_x1_0',
            model_path=str(model_path),
            device=str(self.device),
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def preprocess_crop(self, crop: np.ndarray) -> torch.Tensor:
        img = Image.fromarray(crop[..., ::-1])  # BGR to RGB
        tensor = self.transform(img).unsqueeze(0).to(self.device)  # type: ignore
        return tensor.half() if self.half else tensor

    def get_features(self, bboxes: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """
        Args:
            bboxes: Nx4 numpy array of [x1, y1, x2, y2]
            frame: full frame (H×W×3, BGR uint8)

        Returns:
            NxD normalized feature array
        """
        crops = [frame[int(y1) : int(y2), int(x1) : int(x2)] for x1, y1, x2, y2 in bboxes]
        batch = torch.cat([self.preprocess_crop(crop) for crop in crops], dim=0)
        with torch.no_grad():
            feats = self.extractor(batch)
        feats = F.normalize(feats, dim=1)
        return feats.cpu().numpy()

    def get_features_for_crops(self, crops: list[np.ndarray]) -> Sequence[VectorEmbedding]:
        """
        Args:
            crops: list of cropped person images (H×W×3, BGR uint8)

        Returns:
            NxD normalized feature array in the same order as input crops
        """
        if len(crops) == 0:
            return []

        batch = torch.cat([self.preprocess_crop(c) for c in crops], dim=0)
        with torch.no_grad():
            feats = self.extractor(batch)
        feats = F.normalize(feats, dim=1)
        return [VectorEmbedding(feat) for feat in feats.cpu().numpy()]
