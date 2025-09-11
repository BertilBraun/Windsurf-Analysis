#!/usr/bin/env python3
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from typing import List

import open_clip


class ReID:
    def __init__(self, device: str | None = None):
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
