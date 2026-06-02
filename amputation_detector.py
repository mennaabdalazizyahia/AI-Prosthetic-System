import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass
from enum import Enum

class AmputationType(Enum):
    NORMAL = "normal"
    ABOVE_ELBOW = "above_elbow"
    BELOW_ELBOW = "below_elbow"
    ABOVE_KNEE = "above_knee"
    BELOW_KNEE = "below_knee"
    HAND = "hand"
    FOOT = "foot"

@dataclass
class AmputationResult:
    left_arm: AmputationType
    right_arm: AmputationType
    left_leg: AmputationType
    right_leg: AmputationType
    confidence: Dict[str, float]
    
    def has_amputation(self) -> bool:
        return any([
            self.left_arm != AmputationType.NORMAL,
            self.right_arm != AmputationType.NORMAL,
            self.left_leg != AmputationType.NORMAL,
            self.right_leg != AmputationType.NORMAL
        ])
    
    def get_recommendations(self) -> List[Tuple[str, AmputationType]]:
        recs = []
        if self.left_arm != AmputationType.NORMAL:
            recs.append(("left_arm", self.left_arm))
        if self.right_arm != AmputationType.NORMAL:
            recs.append(("right_arm", self.right_arm))
        if self.left_leg != AmputationType.NORMAL:
            recs.append(("left_leg", self.left_leg))
        if self.right_leg != AmputationType.NORMAL:
            recs.append(("right_leg", self.right_leg))
        return recs

class AmputationDetector:
    
    def __init__(self, use_ai: bool = False, model_path: str = None):
        self.use_ai = use_ai
        if use_ai and model_path:
            self._load_ai_model(model_path)
    
    def _load_ai_model(self, model_path: str):
        import pickle
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"AI model loaded from {model_path}")
        except Exception as e:
            print(f"Could not load AI model: {e}")
            self.use_ai = False
    
    def detect(self, body_landmarks) -> AmputationResult:
        
        if self.use_ai and hasattr(self, 'model'):
            return self._detect_with_ai(body_landmarks)
        else:
            return self._detect_with_rules(body_landmarks)
    
    def _detect_with_rules(self, body_landmarks) -> AmputationResult:
        
        def has_point(point):
            return point is not None
        

        left_arm = AmputationType.NORMAL
        if has_point(body_landmarks.left_shoulder):
            if not has_point(body_landmarks.left_elbow):
                left_arm = AmputationType.ABOVE_ELBOW
            elif has_point(body_landmarks.left_elbow) and not has_point(body_landmarks.left_wrist):
                left_arm = AmputationType.BELOW_ELBOW
            elif has_point(body_landmarks.left_wrist) and not has_point(body_landmarks.left_pinky):
                left_arm = AmputationType.HAND
        
        right_arm = AmputationType.NORMAL
        if has_point(body_landmarks.right_shoulder):
            if not has_point(body_landmarks.right_elbow):
                right_arm = AmputationType.ABOVE_ELBOW
            elif has_point(body_landmarks.right_elbow) and not has_point(body_landmarks.right_wrist):
                right_arm = AmputationType.BELOW_ELBOW
            elif has_point(body_landmarks.right_wrist) and not has_point(body_landmarks.right_pinky):
                right_arm = AmputationType.HAND
        

        left_leg = AmputationType.NORMAL
        if has_point(body_landmarks.left_hip):
            if not has_point(body_landmarks.left_knee):
                left_leg = AmputationType.ABOVE_KNEE
            elif has_point(body_landmarks.left_knee) and not has_point(body_landmarks.left_ankle):
                left_leg = AmputationType.BELOW_KNEE
            elif has_point(body_landmarks.left_ankle) and not has_point(body_landmarks.left_pinky):
                left_leg = AmputationType.FOOT
        
        right_leg = AmputationType.NORMAL
        if has_point(body_landmarks.right_hip):
            if not has_point(body_landmarks.right_knee):
                right_leg = AmputationType.ABOVE_KNEE
            elif has_point(body_landmarks.right_knee) and not has_point(body_landmarks.right_ankle):
                right_leg = AmputationType.BELOW_KNEE
            elif has_point(body_landmarks.right_ankle) and not has_point(body_landmarks.right_pinky):
                right_leg = AmputationType.FOOT
        
        confidence = {
            'left_arm': 0.9 if left_arm != AmputationType.NORMAL else 0.95,
            'right_arm': 0.9 if right_arm != AmputationType.NORMAL else 0.95,
            'left_leg': 0.9 if left_leg != AmputationType.NORMAL else 0.95,
            'right_leg': 0.9 if right_leg != AmputationType.NORMAL else 0.95,
        }
        
        return AmputationResult(left_arm, right_arm, left_leg, right_leg, confidence)
    
    def _detect_with_ai(self, body_landmarks) -> AmputationResult:
        return self._detect_with_rules(body_landmarks)