import cv2
import numpy as np
import logging

class BallTracker:
    """
    Simplified ball tracking using frame differencing and color filtering.
    """
    def __init__(self, config):
        self.config = config
        self.roi = config['video'].get('roi', [0, 0, 1, 1])
        self.prev_frame = None
        self.ball_history = [] # List of (x, y) coordinates
        self.max_history = 5
        self.logger = logging.getLogger(__name__)

    def track(self, frame):
        """
        Detect the ball and calculate velocity.
        """
        h, w = frame.shape[:2]
        y_min, x_min, y_max, x_max = [int(self.roi[0]*h), int(self.roi[1]*w), int(self.roi[2]*h), int(self.roi[3]*w)]
        
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Mask everything outside ROI
        mask = np.zeros_like(gray)
        mask[y_min:y_max, x_min:x_max] = 255
        
        diff = cv2.absdiff(self.prev_frame, gray)
        diff = cv2.bitwise_and(diff, mask)
        self.prev_frame = gray

        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_ball = None
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 10 < area < 300:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                
                if len(approx) > 5:
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    aspect_ratio = float(bw)/bh
                    if 0.7 < aspect_ratio < 1.3:
                        center = (x + bw//2, y + bh//2)
                        
                        # Calculate velocity and bounce
                        velocity = 0
                        is_bounce = False
                        if self.ball_history:
                            last_pos = self.ball_history[-1]
                            velocity = np.sqrt((center[0] - last_pos[0])**2 + (center[1] - last_pos[1])**2)
                            
                            # Bounce Detection: Sudden change in Y-direction (upwards)
                            if len(self.ball_history) >= 2:
                                p2 = self.ball_history[-1] # Previous
                                p1 = self.ball_history[-2] # Before previous
                                dy_prev = p2[1] - p1[1]
                                dy_curr = center[1] - p2[1]
                                
                                # If moving down (dy > 0) then up (dy < -5)
                                if dy_prev > 2 and dy_curr < -2:
                                    # Only count as bounce if in bottom 1/3 of ROI
                                    if center[1] > y_min + (y_max - y_min) * 0.6:
                                        is_bounce = True
                        
                        self.ball_history.append(center)
                        if len(self.ball_history) > self.max_history:
                            self.ball_history.pop(0)
                            
                        return {
                            'bbox': [x, y, x+bw, y+bh], 
                            'center': center,
                            'velocity': velocity,
                            'is_bounce': is_bounce
                        }

        return None
