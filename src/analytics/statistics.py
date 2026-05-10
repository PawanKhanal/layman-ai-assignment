import json
import pandas as pd
import logging
from collections import Counter

class StatisticsManager:
    """
    Manages shot counts, statistics, and report generation.
    """
    def __init__(self, cooldown_frames=30, start_time_str=None):
        self.shots = []
        self.cooldown_frames = cooldown_frames
        self.last_shot_frames = {} # player_id -> last_frame_idx
        self.logger = logging.getLogger(__name__)
        
        # Real-world time setup
        self.start_time = None
        if start_time_str:
            try:
                from datetime import datetime
                self.start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
            except Exception as e:
                self.logger.warning(f"Could not parse start_time: {e}")

    def add_shot(self, frame_idx, timestamp, shot_type, player_id, confidence):
        # Check cooldown
        if player_id in self.last_shot_frames:
            if frame_idx - self.last_shot_frames[player_id] < self.cooldown_frames:
                return False # Suppress

        # Calculate real world time if start_time is provided
        real_time_str = "N/A"
        if self.start_time:
            from datetime import timedelta
            shot_time = self.start_time + timedelta(seconds=timestamp)
            real_time_str = shot_time.strftime("%Y-%m-%d %H:%M:%S")

        shot = {
            "frame": int(frame_idx),
            "timestamp": round(float(timestamp), 2),
            "real_world_time": real_time_str,
            "shot_type": str(shot_type),
            "player_id": int(player_id),
            "confidence": round(float(confidence), 2)
        }
        self.shots.append(shot)
        self.last_shot_frames[player_id] = frame_idx
        self.logger.info(f"Shot detected: {shot_type} by Player {player_id} at frame {frame_idx}")
        return True

    def get_summary(self):
        counts = Counter([s['shot_type'] for s in self.shots])
        return {
            "total_shots": len(self.shots),
            "breakdown": dict(counts)
        }

    def save_reports(self, json_path, csv_path):
        # Save JSON
        summary = self.get_summary()
        report = {
            "statistics": summary,
            "shots": self.shots
        }
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=4)
            
        # Save CSV
        if self.shots:
            df = pd.DataFrame(self.shots)
            df.to_csv(csv_path, index=False)
            
        self.logger.info(f"Reports saved to {json_path} and {csv_path}")
