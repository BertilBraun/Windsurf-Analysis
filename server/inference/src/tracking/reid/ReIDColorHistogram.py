import numpy as np
from typing import Sequence

from server.inference.src.util.algebra import l1_normalize
from server.inference.src.util.similarity_helpers import HistogramEmbedding


class ReIDColorHistogram:
    bins_L = 16
    bins_A = 12
    bins_B = 12

    def get_features_for_crops(self, crops: list[np.ndarray]) -> Sequence[HistogramEmbedding]:
        # compute color histogram for each crop
        # histograms should be HSV histograms with 256 bins for H, 16 bins for S, 8 bins for V
        # return the histograms as a numpy array
        import cv2

        histograms = []
        for crop in crops:
            # crop in futher by 15% in all directions - removes background water and mostly leaves the sail
            h, w = crop.shape[:2]
            w_padding = int(w * 0.15)
            h_padding = int(h * 0.15)
            crop = crop[h_padding : h - h_padding, w_padding : w - w_padding]

            blur_crop = cv2.GaussianBlur(crop, (3, 3), 0)
            lab_crop = cv2.cvtColor(blur_crop, cv2.COLOR_BGR2LAB)
            hist = cv2.calcHist(
                [lab_crop], [0, 1, 2], None, [self.bins_L, self.bins_A, self.bins_B], [0, 256, 0, 256, 0, 256]
            )
            hist = hist.reshape(-1)
            # Normalize to unit length to be comparable across crops
            histograms.append(l1_normalize(hist))
        return [HistogramEmbedding(hist) for hist in histograms]
