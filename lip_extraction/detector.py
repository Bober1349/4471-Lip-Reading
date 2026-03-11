"""Face and mouth detection utilities."""

import cv2
from ultralytics import YOLO
import torch
import os

# Default weights directory relative to the project root
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")


class FaceDetector:
    """Wraps a face detection model (e.g. YOLO, MTCNN, etc.)"""

    def __init__(self, model_name: str = "yolo26n", device: str = "auto"):
        self.model_name = model_name
        
        # Auto-detect GPU if requested
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda:0"
                print(f"🎮 GPU detected: {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                print("💻 No GPU found, using CPU")
        else:
            self.device = device
        
        # Resolve model path: prefer weights/ folder, fall back to cwd
        model_file = f"{model_name}.pt"
        weights_path = os.path.join(WEIGHTS_DIR, model_file)
        model_path = weights_path if os.path.exists(weights_path) else model_file
        
        print(f"📦 Loading {model_name} from {model_path} on {self.device}...")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        print("✅ Model loaded successfully")

    def detect(self, frame):
        """Return a list of face bounding boxes in (x1, y1, x2, y2) format."""
        results = self.model(frame, conf=0.5, verbose=False, device=self.device)
        faces = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                faces.append((int(x1), int(y1), int(x2), int(y2)))
        return faces


def crop_mouth(frame, face_bbox):
    """Given a frame and a face bbox, return an image of the mouth region."""
    x1, y1, x2, y2 = face_bbox
    h = y2 - y1
    mouth_y1 = y1 + int(h * 0.55)
    mouth = frame[mouth_y1:y2, x1:x2]
    return mouth
