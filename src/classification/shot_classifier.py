import numpy as np
import logging

class ShotClassifier:
    """
    Classifies shots based on player pose and racket-ball interaction.
    """
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def classify(self, kpts, ball_pos, player_bbox):
        """
        Shot Classification Rules (Padel):
        - Smash/Overhead: Wrist ABOVE shoulder
        - Serve: Wrist BELOW shoulder (Padel serves are underhand)
        - Forehand: Ball on dominant side
        - Backhand: Ball on non-dominant side
        """
        if kpts is None or len(kpts) == 0:
            return "Unknown", 0.0

        # Keypoints: 5: L_Shoulder, 6: R_Shoulder, 9: L_Wrist, 10: R_Wrist
        l_shoulder = kpts[5]
        r_shoulder = kpts[6]
        l_wrist = kpts[9]
        r_wrist = kpts[10]

        # Determine dominant side
        dist_l = np.sqrt((l_wrist[0] - ball_pos[0])**2 + (l_wrist[1] - ball_pos[1])**2)
        dist_r = np.sqrt((r_wrist[0] - ball_pos[0])**2 + (r_wrist[1] - ball_pos[1])**2)
        
        is_right_dominant = dist_r < dist_l
        active_wrist = r_wrist if is_right_dominant else l_wrist
        active_shoulder = r_shoulder if is_right_dominant else l_shoulder
        
        # Rule 1: Smash/Overhead (High reach)
        # In Padel, if the wrist is above the shoulder, it's a Smash or Bandeja
        if active_wrist[1] < active_shoulder[1]: 
            return "Smash", 0.85
        
        # Rule 2: Forehand/Backhand/Serve
        # We'll use torso position to distinguish sides
        torso_x = (l_shoulder[0] + r_shoulder[0]) / 2
        
        # Simplified: If the ball is very low, it might be a Serve, 
        # but without point-tracking, we'll focus on Forehand/Backhand.
        # We'll label low hits based on side.
        
        if is_right_dominant:
            if ball_pos[0] > torso_x:
                return "Forehand", 0.75
            else:
                return "Backhand", 0.7
        else:
            if ball_pos[0] < torso_x:
                return "Forehand", 0.75
            else:
                return "Backhand", 0.7

        return "Unknown", 0.5
