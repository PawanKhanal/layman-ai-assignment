import json
import pandas as pd
import logging
from collections import Counter

class StatisticsManager:
    """
    Manages shot counts, statistics, and report generation.
    """
    def __init__(self, cooldown_frames=30):
        self.shots = []
        self.cooldown_frames = cooldown_frames
        self.last_shot_frames = {} # player_id -> last_frame_idx
        self.logger = logging.getLogger(__name__)

    def add_shot(self, frame_idx, timestamp, shot_type, player_id, confidence):
        # Check cooldown
        if player_id in self.last_shot_frames:
            if frame_idx - self.last_shot_frames[player_id] < self.cooldown_frames:
                return False # Suppress

        shot = {
            "frame": int(frame_idx),
            "timestamp": round(float(timestamp), 2),
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
