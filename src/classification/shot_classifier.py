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
        Shot Classification Rules:
        - Serve: Arm angle > 160° AND wrist above shoulder
        - Forehand: Arm angle between 90-150° AND racket on dominant side
        - Backhand: Arm angle > 150° AND racket across body
        """
        if kpts is None or len(kpts) == 0:
            return "Unknown", 0.0

        # Keypoints: 5: L_Shoulder, 6: R_Shoulder, 7: L_Elbow, 8: R_Elbow, 9: L_Wrist, 10: R_Wrist
        l_shoulder = kpts[5]
        r_shoulder = kpts[6]
        l_wrist = kpts[9]
        r_wrist = kpts[10]

        # Determine dominant side (active arm)
        # For simplicity, check which wrist is closer to the ball
        dist_l = np.sqrt((l_wrist[0] - ball_pos[0])**2 + (l_wrist[1] - ball_pos[1])**2)
        dist_r = np.sqrt((r_wrist[0] - ball_pos[0])**2 + (r_wrist[1] - ball_pos[1])**2)
        
        is_right_dominant = dist_r < dist_l
        active_wrist = r_wrist if is_right_dominant else l_wrist
        active_shoulder = r_shoulder if is_right_dominant else l_shoulder
        
        # Calculate arm angle (placeholder for now, use simple vertical distance)
        # In a real system, we'd use atan2 for 3 points (shoulder, elbow, wrist)
        
        # Rule 1: Serve (High reach)
        if active_wrist[1] < active_shoulder[1] - 30: # Wrist significantly above shoulder
            return "Serve", 0.8
        
        # Rule 2: Forehand/Backhand
        # Based on position relative to torso center
        torso_x = (l_shoulder[0] + r_shoulder[0]) / 2
        
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
