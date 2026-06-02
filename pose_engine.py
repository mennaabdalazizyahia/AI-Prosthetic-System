#pip install mediapipe=0.10.14
import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import yaml

@dataclass
class BodyLandmarks:
    # Arms
    left_shoulder: Optional[Tuple[float, float]] = None
    left_elbow: Optional[Tuple[float, float]] = None
    left_wrist: Optional[Tuple[float, float]] = None
    right_shoulder: Optional[Tuple[float, float]] = None
    right_elbow: Optional[Tuple[float, float]] = None
    right_wrist: Optional[Tuple[float, float]] = None
    
    # legs
    left_hip: Optional[Tuple[float, float]] = None
    left_knee: Optional[Tuple[float, float]] = None
    left_ankle: Optional[Tuple[float, float]] = None
    right_hip: Optional[Tuple[float, float]] = None
    right_knee: Optional[Tuple[float, float]] = None
    right_ankle: Optional[Tuple[float, float]] = None
    
    left_pinky: Optional[Tuple[float, float]] = None
    right_pinky: Optional[Tuple[float, float]] = None
    nose: Optional[Tuple[float, float]] = None
    
    @property
    def all_points(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}

class PoseEngine:
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        pose_config = config['model']['pose']
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,  
            min_detection_confidence=pose_config['min_detection_confidence'],
            min_tracking_confidence=pose_config['min_tracking_confidence']
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def detect(self, image: np.ndarray) -> tuple[np.ndarray, Optional[BodyLandmarks]]:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            return image, None
        
        # Annotated_image
        annotated_image = image.copy()
        self.mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )
        
        # BodyLandmarks
        landmarks = results.pose_landmarks.landmark
        body = BodyLandmarks()
        
        # MediaPipe indices:
        # 11,12: shoulders, 13,14: elbows, 15,16: wrists
        # 23,24: hips, 25,26: knees, 27,28: ankles
        # 17,18: pinky (optional)
        # 0: nose
        
        def get_point(idx):
            if landmarks[idx].visibility > 0.5:
                return (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
            return None
        
        body.left_shoulder = get_point(11)
        body.left_elbow = get_point(13)
        body.left_wrist = get_point(15)
        body.right_shoulder = get_point(12)
        body.right_elbow = get_point(14)
        body.right_wrist = get_point(16)
        
        body.left_hip = get_point(23)
        body.left_knee = get_point(25)
        body.left_ankle = get_point(27)
        body.right_hip = get_point(24)
        body.right_knee = get_point(26)
        body.right_ankle = get_point(28)
        
        body.left_pinky = get_point(17) or get_point(19)
        body.right_pinky = get_point(18) or get_point(20)
        body.nose = get_point(0)
        
        return annotated_image, body
    
    def calculate_angles(self, body: BodyLandmarks) -> Dict:
        angles = {}
        
        # Right Elbow
        if all([body.right_shoulder, body.right_elbow, body.right_wrist]):
            angles['right_elbow'] = self._angle_between(
                body.right_shoulder, body.right_elbow, body.right_wrist
            )
        
        # Left Elbow
        if all([body.left_shoulder, body.left_elbow, body.left_wrist]):
            angles['left_elbow'] = self._angle_between(
                body.left_shoulder, body.left_elbow, body.left_wrist
            )
        
        # Right Knee
        if all([body.right_hip, body.right_knee, body.right_ankle]):
            angles['right_knee'] = self._angle_between(
                body.right_hip, body.right_knee, body.right_ankle
            )
        
        # Left Knee
        if all([body.left_hip, body.left_knee, body.left_ankle]):
            angles['left_knee'] = self._angle_between(
                body.left_hip, body.left_knee, body.left_ankle
            )
        
        return angles
    
    @staticmethod
    def _angle_between(p1, p2, p3) -> float:
        import math
        a = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        b = math.atan2(p3[1] - p2[1], p3[0] - p2[0])
        angle = abs(math.degrees(a - b))
        return min(360 - angle, angle) if angle > 180 else angle