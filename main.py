import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import argparse
import os
from dotenv import load_dotenv
from utils.pose_utils import PoseEstimator
from classification.shot_classifier import ShotClassifier

# Load environment variables
load_dotenv()

class PadelAnalytics:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.getenv("YOLO_MODEL", "yolov8m.pt")
        self.model = YOLO(model_path)
        self.pose_model = YOLO("yolov8m-pose.pt") # Pose model for players
        self.pose_estimator = PoseEstimator()
        self.shot_classifier = ShotClassifier()
        self.results_data = []
        self.shot_counts = {"Forehand": 0, "Backhand": 0, "Smash/Serve": 0}

    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Detection and Tracking (Ball, Racket, Players)
            results = self.model.track(frame, persist=True, classes=[0, 32, 38], verbose=False)
            
            # 2. Extract Detections
            detections = self.extract_detections(results, frame_idx)
            self.results_data.extend(detections)

            # Separate detections by class
            players = [d for d in detections if d['class'] == 'person']
            ball = next((d for d in detections if d['class'] == 'sports ball'), None)
            rackets = [d for d in detections if d['class'] == 'tennis racket']

            # 3. Shot Classification Logic
            if ball:
                ball_center = ((ball['bbox'][0] + ball['bbox'][2])/2, (ball['bbox'][1] + ball['bbox'][3])/2)
                racket_id, event = self.shot_classifier.detect_hit(ball_center, rackets)
                
                if event == "Hit":
                    best_player = self.get_nearest_player(ball_center, players)
                    if best_player:
                        # Get pose for this specific player
                        # We can either use a pose model on the whole frame or a crop
                        pose_results = self.pose_model(frame, verbose=False)
                        # Find the pose that matches our best_player track
                        # For simplicity, we'll find the pose closest to the player's center
                        p_center = ((best_player['bbox'][0] + best_player['bbox'][2])/2, (best_player['bbox'][1] + best_player['bbox'][3])/2)
                        
                        best_pose_idx = -1
                        min_pose_dist = float('inf')
                        if hasattr(pose_results[0], 'boxes'):
                            pose_boxes = pose_results[0].boxes.xyxy.cpu().numpy()
                            for i, pbox in enumerate(pose_boxes):
                                pb_center = ((pbox[0] + pbox[2])/2, (pbox[1] + pbox[3])/2)
                                d = np.sqrt((p_center[0] - pb_center[0])**2 + (p_center[1] - pb_center[1])**2)
                                if d < min_pose_dist:
                                    min_pose_dist = d
                                    best_pose_idx = i
                        
                        kpts = self.pose_estimator.get_pose_from_yolo(pose_results, best_pose_idx)
                        shot_type = self.pose_estimator.classify_by_pose(kpts, ball_center, best_player['bbox'])
                        
                        if shot_type in self.shot_counts:
                            self.shot_counts[shot_type] += 1
                            # Draw visual feedback for hit
                            cv2.putText(frame, f"SHOT: {shot_type}", (50, 100), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            # 4. Visualization
            annotated_frame = results[0].plot()
            # Overlay shot counts
            y_offset = 150
            for shot, count in self.shot_counts.items():
                cv2.putText(annotated_frame, f"{shot}: {count}", (50, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                y_offset += 40

            out.write(annotated_frame)

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"Processed frame {frame_idx}/{total_frames}")

        cap.release()
        out.release()
        self.save_results()

    def get_nearest_player(self, pos, players):
        min_dist = float('inf')
        nearest = None
        for p in players:
            p_center = ((p['bbox'][0] + p['bbox'][2])/2, (p['bbox'][1] + p['bbox'][3])/2)
            dist = np.sqrt((pos[0] - p_center[0])**2 + (pos[1] - p_center[1])**2)
            if dist < min_dist:
                min_dist = dist
                nearest = p
        return nearest

    def extract_detections(self, results, frame_idx):
        detections = []
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            cls = results[0].boxes.cls.cpu().numpy().astype(int)
            conf = results[0].boxes.conf.cpu().numpy()

            for box, id, c, cf in zip(boxes, ids, cls, conf):
                detections.append({
                    "frame": frame_idx,
                    "track_id": id,
                    "class": self.model.names[c],
                    "bbox": box.tolist(),
                    "confidence": cf
                })
        return detections

    def save_results(self):
        df = pd.DataFrame(self.results_data)
        df.to_csv("output_results.csv", index=False)
        df.to_json("output_results.json", orient="records")
        
        summary = {
            "total_shots": sum(self.shot_counts.values()),
            "shot_counts": self.shot_counts
        }
        with open("summary.json", "w") as f:
            import json
            json.dump(summary, f, indent=4)
            
        print("Results saved to output_results.csv, output_results.json and summary.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=os.getenv("INPUT_VIDEO_PATH", "input_video.mp4"), help="Path to input video")
    parser.add_argument("--output", type=str, default=os.getenv("OUTPUT_VIDEO_PATH", "output_video.mp4"), help="Path to output video")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input video {args.input} not found.")
    else:
        analytics = PadelAnalytics()
        analytics.process_video(args.input, args.output)
