import cv2
import numpy as np
from typing import Tuple, Optional
import math

class ProstheticVisualizer:
    
    def __init__(self, prosthetics_path: str = "prosthetics/"):
        self.prosthetics_path = prosthetics_path
        self.cache = {} 
        
    def load_prosthetic_image(self, amputation_type: str, side: str) -> Optional[np.ndarray]:
 
        filename = f"{amputation_type}_{side}.png"
        path = f"{self.prosthetics_path}/{filename}"
        
        if path in self.cache:
            return self.cache[path]
        
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                img = self._create_colored_prosthetic(amputation_type, side)
            self.cache[path] = img
            return img
        except Exception as e:
            print(f"Could not load {path}: {e}")
            return self._create_colored_prosthetic(amputation_type, side)
    
    def _create_colored_prosthetic(self, amputation_type: str, side: str) -> np.ndarray:
        colors = {
            'above_elbow': (255, 100, 0),
            'below_elbow': (0, 150, 255),
            'above_knee': (0, 100, 255),
            'below_knee': (100, 255, 0)
        }
        color = colors.get(amputation_type, (0, 255, 0))
        
        img = np.zeros((100, 300, 4), dtype=np.uint8)
        img[:, :, 0:3] = color
        img[:, :, 3] = 200  
        return img
    
    def overlay_prosthetic(
        self,
        image: np.ndarray,
        start_point: Tuple[int, int],
        end_point: Tuple[int, int],
        prosthetic_img: np.ndarray
    ) -> np.ndarray:
 
        result = image.copy()
        
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        length = int(math.sqrt(dx**2 + dy**2))
        angle = math.degrees(math.atan2(dy, dx))
        
        new_width = length
        new_height = int(prosthetic_img.shape[0] * (length / prosthetic_img.shape[1]))
        
        if new_width < 5 or new_height < 5:
            return result
        
        prosthetic_resized = cv2.resize(prosthetic_img, (new_width, new_height))
        
        center = (prosthetic_resized.shape[1] // 2, prosthetic_resized.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            prosthetic_resized, rotation_matrix,
            (prosthetic_resized.shape[1], prosthetic_resized.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        
        offset_x = start_point[0] - center[0]
        offset_y = start_point[1] - center[1]
        
        for y in range(rotated.shape[0]):
            for x in range(rotated.shape[1]):
                img_y = offset_y + y
                img_x = offset_x + x
                
                if 0 <= img_y < result.shape[0] and 0 <= img_x < result.shape[1]:
                    alpha = rotated[y, x, 3] / 255.0
                    if alpha > 0.1:
                        result[img_y, img_x] = (
                            result[img_y, img_x] * (1 - alpha) +
                            rotated[y, x, :3] * alpha
                        ).astype(np.uint8)
        
        return result
    
    def visualize_amputation(
        self,
        image: np.ndarray,
        body_landmarks,
        amputation_result,
        pose_engine
    ) -> np.ndarray:
        
        result = image.copy()
        
        recommendations = amputation_result.get_recommendations()
        
        for limb, amputation_type in recommendations:
            prosthetic_img = self.load_prosthetic_image(
                amputation_type.value,
                limb.split('_')[0]  # left or right
            )
            
            if prosthetic_img is None:
                continue
            
            if limb == 'left_arm':
                start = body_landmarks.left_shoulder
                if amputation_type.value == 'above_elbow':
                    end = body_landmarks.left_elbow or body_landmarks.left_shoulder
                else:
                    end = body_landmarks.left_wrist or body_landmarks.left_elbow
                    
            elif limb == 'right_arm':
                start = body_landmarks.right_shoulder
                if amputation_type.value == 'above_elbow':
                    end = body_landmarks.right_elbow or body_landmarks.right_shoulder
                else:
                    end = body_landmarks.right_wrist or body_landmarks.right_elbow
                    
            elif limb == 'left_leg':
                start = body_landmarks.left_hip
                if amputation_type.value == 'above_knee':
                    end = body_landmarks.left_knee or body_landmarks.left_hip
                else:
                    end = body_landmarks.left_ankle or body_landmarks.left_knee
                    
            elif limb == 'right_leg':
                start = body_landmarks.right_hip
                if amputation_type.value == 'above_knee':
                    end = body_landmarks.right_knee or body_landmarks.right_hip
                else:
                    end = body_landmarks.right_ankle or body_landmarks.right_knee
            else:
                continue
            
            if start and end:
                result = self.overlay_prosthetic(result, start, end, prosthetic_img)
        
        return result