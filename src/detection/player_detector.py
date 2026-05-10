import cv2
import numpy as np
from ultralytics import YOLO
import logging

class PlayerDetector:
    """
    Handles player detection using YOLOv8 and pose estimation.
    """
    def __init__(self, config):
        self.config = config
        self.model = YOLO(config['detection']['player_model'])
        self.pose_model = YOLO(config['detection']['pose_model'])
        self.conf = config['detection']['confidence_threshold']
        self.roi = config['video'].get('roi', [0, 0, 1, 1])
        self.logger = logging.getLogger(__name__)

    def detect(self, frame):
        """
        Detect players and rackets, then return associated objects.
        """
        # Detect person (0) and tennis racket (38)
        results = self.model.track(frame, persist=True, classes=[0, 38], conf=self.conf, verbose=False)
        players = []
        rackets = []
        
        h, w = frame.shape[:2]
        y_min, x_min, y_max, x_max = [int(self.roi[0]*h), int(self.roi[1]*w), int(self.roi[2]*h), int(self.roi[3]*w)]

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, tid, cf, cls in zip(boxes, ids, confs, classes):
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                
                # Only keep objects in ROI
                if not (x_min <= cx <= x_max and y_min <= cy <= y_max):
                    continue

                if cls == 0: # Person
                    players.append({
                        'id': tid,
                        'bbox': box,
                        'conf': cf,
                        'racket_id': None
                    })
                elif cls == 38: # Tennis Racket
                    rackets.append({
                        'id': tid,
                        'bbox': box,
                        'conf': cf
                    })

        # Associate rackets with players based on proximity
        for racket in rackets:
            r_center = ((racket['bbox'][0] + racket['bbox'][2])/2, (racket['bbox'][1] + racket['bbox'][3])/2)
            best_p = None
            min_dist = 150 # Max distance to associate racket with hand
            
            for player in players:
                p_center = ((player['bbox'][0] + player['bbox'][2])/2, (player['bbox'][1] + player['bbox'][3])/2)
                d = np.sqrt((r_center[0] - p_center[0])**2 + (r_center[1] - p_center[1])**2)
                if d < min_dist:
                    min_dist = d
                    best_p = player
            
            if best_p:
                best_p['racket_bbox'] = racket['bbox']

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
