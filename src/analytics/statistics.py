import json
import pandas as pd
import logging
from collections import Counter

class StatisticsManager:
    """
    Manages shot counts, statistics, and report generation.
    """
    def __init__(self):
        self.shots = []
        self.logger = logging.getLogger(__name__)

    def add_shot(self, frame_idx, timestamp, shot_type, player_id, confidence):
        shot = {
            "frame": int(frame_idx),
            "timestamp": round(float(timestamp), 2),
            "shot_type": str(shot_type),
            "player_id": int(player_id),
            "confidence": round(float(confidence), 2)
        }
        self.shots.append(shot)
        self.logger.info(f"Shot detected: {shot_type} by Player {player_id} at frame {frame_idx}")

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
