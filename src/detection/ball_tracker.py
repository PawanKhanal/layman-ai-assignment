import cv2
import numpy as np
import logging

class BallTracker:
    """
    Simplified ball tracking using frame differencing and color filtering.
    """
    def __init__(self):
        self.prev_frame = None
        self.logger = logging.getLogger(__name__)

    def track(self, frame):
        """
        Detect the ball based on motion and shape.
        """
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(self.prev_frame, gray)
        self.prev_frame = gray

        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_ball = None
        min_dist_to_center = float('inf')
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 10 < area < 300: # Ball area range
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                
                # Ball should be roughly circular
                if len(approx) > 5:
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = float(w)/h
                    if 0.7 < aspect_ratio < 1.3:
                        center = (x + w//2, y + h//2)
                        return {'bbox': [x, y, x+w, y+h], 'center': center}

        return None
