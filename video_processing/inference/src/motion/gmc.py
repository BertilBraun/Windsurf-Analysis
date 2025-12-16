from typing import Any
import cv2
import copy
import numpy as np


class GMC:
    def __init__(self, downscale: int = 2):
        self.downscale = max(1, int(downscale))

        self.feature_params: dict[str, Any] = dict(
            maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3, useHarrisDetector=False, k=0.04
        )

        self.prevFrame: np.ndarray | None = None
        self.prevKeyPoints: np.ndarray | None = None

        self.initializedFirstFrame: bool = False

    def apply(self, raw_frame: np.ndarray) -> np.ndarray:
        """Run GMC on a single frame to compute the rigid transformation matrix compared to the previous frame.
        Returns the rigid transformation matrix (2x3 Matrix H).
        """
        # Initialize
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3)

        # Downscale image
        if self.downscale > 1.0:
            # frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))

        # find the keypoints
        keypoints = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)

        # Handle first frame
        if not self.initializedFirstFrame:
            # Initialize data
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)

            # Initialization done
            self.initializedFirstFrame = True

            return H

        assert self.prevFrame is not None
        assert self.prevKeyPoints is not None

        # find correspondences
        matchedKeypoints, status, err = cv2.calcOpticalFlowPyrLK(self.prevFrame, frame, self.prevKeyPoints, None)  # type: ignore

        # leave good correspondences only
        prevPoints: list[np.ndarray] = []
        currPoints: list[np.ndarray] = []

        for i in range(len(status)):
            if status[i]:
                prevPoints.append(self.prevKeyPoints[i])
                currPoints.append(matchedKeypoints[i])

        prevPointsNp = np.array(prevPoints)
        currPointsNp = np.array(currPoints)

        # Find rigid matrix
        if (prevPointsNp.shape[0] > 4) and (prevPointsNp.shape[0] == currPointsNp.shape[0]):
            H, inliers = cv2.estimateAffinePartial2D(prevPointsNp, currPointsNp, method=cv2.RANSAC)

            # Handle downscale
            if self.downscale > 1.0:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
        else:
            print('Warning: not enough matching points')

        # Store to next iteration
        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)

        return H
