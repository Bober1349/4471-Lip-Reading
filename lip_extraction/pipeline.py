"""Pipeline orchestration for extracting "good" lip-reading clips."""

import cv2
from .detector import FaceDetector, crop_mouth
from .motion import LipMotionFilter


class ExtractionPipeline:
    def __init__(self, face_detector=None, motion_filter=None, min_face_area=10000, device="auto"):
        self.face_detector = face_detector or FaceDetector(device=device)
        self.motion_filter = motion_filter or LipMotionFilter()
        self.min_face_area = min_face_area
        self.device = device

    def process_video(self, path, clip_len=16, mouth_size=(96, 96)):
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"\n📹 Processing video: {total_frames} frames at {fps:.1f} FPS\n")
        
        good_clips = []
        mouth_seq = []
        frame_count = 0
        faces_detected = 0
        motion_checks = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 300 == 0:  # Print progress every 300 frames
                progress = (frame_count / total_frames) * 100
                print(f"  Frame {frame_count}/{total_frames} ({progress:.1f}%) | Faces: {faces_detected} | Clips: {len(good_clips)}")
            
            faces = self.face_detector.detect(frame)
            if not faces:
                mouth_seq.clear()
                continue
            
            faces_detected += 1
            # pick largest face
            face = max(faces, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
            area = (face[2]-face[0])*(face[3]-face[1])
            if area < self.min_face_area:
                mouth_seq.clear()
                continue
            
            mouth = crop_mouth(frame, face)
            # Resize to consistent size for optical flow calculation
            mouth = cv2.resize(mouth, mouth_size)
            mouth_seq.append(mouth)
            
            if len(mouth_seq) >= clip_len:
                motion_checks += 1
                if self.motion_filter.is_moving(mouth_seq):
                    good_clips.append(list(mouth_seq))
                    print(f"    ✓ Found good clip #{len(good_clips)}")
                mouth_seq.pop(0)
        
        cap.release()
        print(f"\n✅ Processing complete: {faces_detected} faces detected, {motion_checks} motion checks, {len(good_clips)} clips extracted\n")
        return good_clips
