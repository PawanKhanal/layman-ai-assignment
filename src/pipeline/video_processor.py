import cv2
import logging
import os

class VideoProcessor:
    """
    Handles video reading and writing.
    """
    def __init__(self, input_path, output_path, frame_skip=1, resize_factor=1.0):
        self.input_path = input_path
        self.output_path = output_path
        self.frame_skip = frame_skip
        self.resize_factor = resize_factor
        self.logger = logging.getLogger(__name__)
        
        self.cap = cv2.VideoCapture(input_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {input_path}")
            
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap_get_height := self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Output setup
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_w = int(self.width * resize_factor)
        out_h = int(cap_get_height * resize_factor)
        self.writer = cv2.VideoWriter(output_path, fourcc, self.fps, (out_w, out_h))

    def get_frames(self):
        """
        Generator for video frames.
        """
        frame_idx = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if frame_idx % self.frame_skip == 0:
                if self.resize_factor != 1.0:
                    frame = cv2.resize(frame, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
                yield frame_idx, frame
                
            frame_idx += 1

    def write_frame(self, frame):
        self.writer.write(frame)

    def release(self):
        self.cap.release()
        self.writer.release()
        self.logger.info(f"Processing complete. Video saved to {self.output_path}")
