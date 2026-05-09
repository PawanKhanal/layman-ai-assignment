import numpy as np

class ShotClassifier:
    def __init__(self):
        self.ball_history = []
        self.racket_history = {} # track_id -> history
        self.hit_threshold = 50 # pixels

    def detect_hit(self, ball_pos, rackets):
        """
        ball_pos: (x, y)
        rackets: list of {'track_id': id, 'bbox': [x1, y1, x2, y2]}
        """
        for racket in rackets:
            r_bbox = racket['bbox']
            r_center = ((r_bbox[0] + r_bbox[2])/2, (r_bbox[1] + r_bbox[3])/2)
            dist = np.sqrt((ball_pos[0] - r_center[0])**2 + (ball_pos[1] - r_center[1])**2)
            
            if dist < self.hit_threshold:
                # Potential hit, check for velocity change in next frames or past frames
                return racket['track_id'], "Hit"
        return None, None

    def classify(self, player_pose, hit_pos, player_bbox):
        """
        player_pose: MediaPipe landmarks
        hit_pos: (x, y) ball position at hit
        player_bbox: [x1, y1, x2, y2]
        """
        if player_pose is None:
            return "Unknown"

        # Get keypoints (relative to crop)
        # For simplicity, let's use relative horizontal position to player center
        p_center_x = (player_bbox[0] + player_bbox[2]) / 2
        p_head_y = player_bbox[1] # Approximate head y

        # Smash/Serve: Ball is high
        if hit_pos[1] < p_head_y + (player_bbox[3] - player_bbox[1]) * 0.2:
            return "Smash/Serve"

        # Forehand vs Backhand
        # Assuming right-handed player for now. 
        # (A better way is to see which arm is extended)
        if hit_pos[0] > p_center_x:
            return "Forehand"
        else:
            return "Backhand"
