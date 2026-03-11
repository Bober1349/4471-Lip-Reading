import unittest
import numpy as np
import cv2
from lip_extraction.detector import crop_mouth

class Dummy:
    pass

class DetectorTests(unittest.TestCase):
    def test_crop_mouth_dimensions(self):
        img = np.zeros((100,100,3), dtype=np.uint8)
        mouth = crop_mouth(img, (10, 10, 90, 90))
        # mouth should be lower part of the face box
        self.assertTrue(mouth.shape[0] > 0)
        self.assertEqual(mouth.shape[1], 80)

if __name__ == '__main__':
    unittest.main()
