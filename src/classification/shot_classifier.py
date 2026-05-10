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
        
        # Rule 2: Forehand/Backhand
        # Orientation-aware logic: 
        # Bottom players (y > 400) face UP. Right of screen = Right side of body.
        # Top players (y < 400) face DOWN. Right of screen = Left side of body.
        torso_x = (l_shoulder[0] + r_shoulder[0]) / 2
        is_top_side = player_bbox[1] < 400
        
        # Simplified assumption: Player uses their right hand for most shots
        # (Or whichever hand is closer to the ball center)
        if ball_pos[0] > torso_x: # Ball is on the right side of the screen
            if is_top_side:
                return "Backhand", 0.75 # Facing us, right of screen is their left
            else:
                return "Forehand", 0.75 # Facing away, right of screen is their right
        else: # Ball is on the left side of the screen
            if is_top_side:
                return "Forehand", 0.75 # Facing us, left of screen is their right
            else:
                return "Backhand", 0.75 # Facing away, left of screen is their left

        return "Unknown", 0.5
