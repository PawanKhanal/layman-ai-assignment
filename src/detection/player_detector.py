import cv2
import numpy as np
from ultralytics import YOLO
import logging

class PlayerDetector:
    """
    Handles player detection using YOLOv8 and pose estimation.
    """
    def __init__(self, model_path='yolov8n.pt', pose_model_path='yolov8n-pose.pt', conf=0.5):
        self.model = YOLO(model_path)
        self.pose_model = YOLO(pose_model_path)
        self.conf = conf
        self.logger = logging.getLogger(__name__)

    def detect(self, frame):
        """
        Detect players and return their bounding boxes and track IDs.
        """
        results = self.model.track(frame, persist=True, classes=[0], conf=self.conf, verbose=False)
        players = []
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            
            for box, tid, cf in zip(boxes, ids, confs):
                players.append({
                    'id': tid,
                    'bbox': box,
                    'conf': cf
                })
        return players

    def get_poses(self, frame, players):
        """
        Get poses for detected players.
        """
        # Run pose model on the whole frame
        pose_results = self.pose_model(frame, verbose=False)
        poses = {}
        
        if hasattr(pose_results[0], 'keypoints') and pose_results[0].keypoints is not None:
            kpts = pose_results[0].keypoints.xy.cpu().numpy()
            pose_boxes = pose_results[0].boxes.xyxy.cpu().numpy()
            
            for player in players:
                p_center = ((player['bbox'][0] + player['bbox'][2])/2, (player['bbox'][1] + player['bbox'][3])/2)
                
                # Match pose to player based on proximity
                best_idx = -1
                min_dist = float('inf')
                for i, pbox in enumerate(pose_boxes):
                    pb_center = ((pbox[0] + pbox[2])/2, (pbox[1] + pbox[3])/2)
                    d = np.sqrt((p_center[0] - pb_center[0])**2 + (p_center[1] - pb_center[1])**2)
                    if d < min_dist:
                        min_dist = d
                        best_idx = i
                
                if best_idx != -1 and min_dist < 100: # Threshold for matching
                    poses[player['id']] = kpts[best_idx]
                    
        return poses
