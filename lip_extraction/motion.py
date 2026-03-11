"""Lip motion / clarity filtering utilities."""

import cv2
import numpy as np


class LipMotionFilter:
    """Decide whether a mouth ROI contains clear lip movement."""

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        # could load a small classifier here later

    def is_moving(self, roi_sequence: list) -> bool:
        """Simple optical-flow based motion detector over a list of ROIs."""
        if len(roi_sequence) < 2:
            return False
        prev = cv2.cvtColor(roi_sequence[0], cv2.COLOR_BGR2GRAY)
        total_flow = 0.0
        for img in roi_sequence[1:]:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            total_flow += np.mean(mag)
            prev = gray
        return total_flow > self.threshold
