import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from typing import List, Union
from typing import Protocol
from pathlib import Path


class ReID(Protocol):
    def get_features_for_crops(self, crops: List[np.ndarray]) -> np.ndarray: ...


class ReIDViT:
    def __init__(self, device: str | None = None):
        import open_clip

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.device = torch.device(device)
        self.half = device == 'cuda'

        # Models
        model_name, pretrained = 'ViT-B-32', 'laion2b_s34b_b79k'
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()
        # ensure fp16 only on CUDA
        if self.half:
            self.model.half()

    def _preprocess_crop(self, crop_bgr: np.ndarray) -> torch.Tensor:
        # BGR -> RGB
        img = Image.fromarray(crop_bgr[..., ::-1])
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)  # type: ignore
        return tensor.half() if self.half else tensor

    @torch.no_grad()
    def _encode(self, batch: torch.Tensor) -> torch.Tensor:
        # open_clip.encode_image expects normalized tensors in model dtype
        feats = self.model.encode_image(batch)  # returns un-normalized features # type: ignore
        return feats

    def _to_unit(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1)

    @torch.no_grad()
    def get_features_for_crops(self, crops: List[np.ndarray]) -> np.ndarray:
        assert len(crops) > 0

        batch = torch.cat([self._preprocess_crop(c) for c in crops], dim=0)
        feats = self._encode(batch)
        feats = self._to_unit(feats)
        return feats.float().cpu().numpy()


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

    def get_features_for_crops(self, crops: list[np.ndarray]) -> np.ndarray:
        """
        Args:
            crops: list of cropped person images (H×W×3, BGR uint8)

        Returns:
            NxD normalized feature array in the same order as input crops
        """
        if len(crops) == 0:
            return np.zeros((0, 512), dtype=np.float32)

        batch = torch.cat([self.preprocess_crop(c) for c in crops], dim=0)
        with torch.no_grad():
            feats = self.extractor(batch)
        feats = F.normalize(feats, dim=1)
        return feats.cpu().numpy()


class ReIDColorHistogram:
    bins_L = 16
    bins_A = 12
    bins_B = 12

    def get_features_for_crops(self, crops: list[np.ndarray]) -> np.ndarray:
        # compute color histogram for each crop
        # histograms should be HSV histograms with 256 bins for H, 16 bins for S, 8 bins for V
        # return the histograms as a numpy array
        import cv2

        histograms = []
        for crop in crops:
            # crop in futher by 10% in all directions - removes background water and mostly leaves the sail
            h, w = crop.shape[:2]
            w_padding = w // 10
            h_padding = h // 10
            crop = crop[h_padding : h - h_padding, w_padding : w - w_padding]

            blur_crop = cv2.GaussianBlur(crop, (3, 3), 0)
            lab_crop = cv2.cvtColor(blur_crop, cv2.COLOR_BGR2LAB)
            hist = cv2.calcHist(
                [lab_crop], [0, 1, 2], None, [self.bins_L, self.bins_A, self.bins_B], [0, 256, 0, 256, 0, 256]
            )
            hist = hist.reshape(-1)
            # Normalize to unit length to be comparable across crops
            hist /= np.linalg.norm(hist) + 1e-6
            histograms.append(hist)
        return np.array(histograms)
