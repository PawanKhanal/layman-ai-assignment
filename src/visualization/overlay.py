import cv2
import numpy as np

class Visualizer:
    """
    Handles drawing annotations on video frames.
    """
    def __init__(self, config):
        self.config = config['visualization']
        self.colors = {
            'player': (0, 255, 0),
            'ball': (0, 255, 255),
            'shot': (0, 0, 255),
            'pose': (255, 0, 0)
        }

    def draw_player(self, frame, player):
        if not self.config['show_bboxes']:
            return
        
        bbox = player['bbox'].astype(int)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), self.colors['player'], 2)
        cv2.putText(frame, f"P{player['id']}", (bbox[0], bbox[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['player'], 2)

    def draw_ball(self, frame, ball):
        if ball is None:
            return
        center = tuple(map(int, ball['center']))
        cv2.circle(frame, center, 5, self.colors['ball'], -1)
        
        # Draw bounce indicator
        if ball.get('is_bounce'):
            cv2.putText(frame, "BOUNCE!", (center[0]-40, center[1]-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            cv2.circle(frame, center, 15, (255, 255, 255), 2)

    def draw_racket(self, frame, player):
        if 'racket_bbox' in player:
            rb = player['racket_bbox'].astype(int)
            cv2.rectangle(frame, (rb[0], rb[1]), (rb[2], rb[3]), (255, 255, 0), 2)
            cv2.putText(frame, "Racket", (rb[0], rb[1]-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    def draw_pose(self, frame, kpts):
        if not self.config['show_pose'] or kpts is None:
            return
        
        # Draw skeleton connections
        connections = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10)] # Shoulders and arms
        for start, end in connections:
            if kpts[start][0] > 0 and kpts[end][0] > 0:
                p1 = tuple(map(int, kpts[start]))
                p2 = tuple(map(int, kpts[end]))
                cv2.line(frame, p1, p2, self.colors['pose'], 2)
        
        # Draw keypoints
        for kp in kpts:
            if kp[0] > 0:
                cv2.circle(frame, tuple(map(int, kp)), 3, self.colors['pose'], -1)

    def draw_shot_info(self, frame, shot_type, count_summary):
        if not self.config['show_shot_text']:
            return
            
        # Draw current shot type prominently
        cv2.putText(frame, f"LATEST SHOT: {shot_type}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.colors['shot'], 3)
        
        # Draw summary table
        y = 100
        cv2.putText(frame, "Summary:", (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        for s_type, count in count_summary['breakdown'].items():
            y += 30
            cv2.putText(frame, f"- {s_type}: {count}", (60, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
