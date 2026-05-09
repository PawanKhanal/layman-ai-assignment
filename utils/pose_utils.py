import numpy as np

class PoseEstimator:
    """
    Wrapper for YOLOv8 Pose results.
    """
    def __init__(self):
        # We don't need a separate model here if we use YOLOv8-pose in main.py
        pass

    def get_pose_from_yolo(self, results, player_idx):
        """
        results: Ultralytics Results object
        player_idx: index of the player in the results.boxes
        """
        if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
            # keypoints.xy is [N, 17, 2]
            kpts = results[0].keypoints.xy.cpu().numpy()
            if player_idx < len(kpts):
                return kpts[player_idx]
        return None

    def classify_by_pose(self, kpts, hit_pos, player_bbox):
        """
        Simple classification using YOLO keypoints.
        kpts: [17, 2] array of (x, y) coordinates
        hit_pos: (x, y)
        """
        if kpts is None or len(kpts) == 0:
            return "Unknown"

        # 17 keypoints: 0: nose, 5: l_shoulder, 6: r_shoulder, 7: l_elbow, 8: r_elbow, 9: l_wrist, 10: r_wrist
        # Approximate head height using nose or shoulders
        nose = kpts[0]
        l_shoulder = kpts[5]
        r_shoulder = kpts[6]
        
        avg_shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2 if (l_shoulder[1] > 0 and r_shoulder[1] > 0) else player_bbox[1]

        # Smash/Serve: Hit is above shoulder/head
        if hit_pos[1] < avg_shoulder_y - 20: # 20 pixel margin
            return "Smash/Serve"

        # Forehand vs Backhand
        # Determine center of torso
        torso_x = (l_shoulder[0] + r_shoulder[0]) / 2 if (l_shoulder[0] > 0 and r_shoulder[0] > 0) else (player_bbox[0] + player_bbox[2]) / 2
        
        # Assuming right-handed: Forehand is on the right, Backhand on the left
        # This is a simplification.
        if hit_pos[0] > torso_x:
            return "Forehand"
        else:
            return "Backhand"
