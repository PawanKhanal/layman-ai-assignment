import argparse
import yaml
import logging
import os
from pipeline.video_processor import VideoProcessor
from detection.player_detector import PlayerDetector
from detection.ball_tracker import BallTracker
from classification.shot_classifier import ShotClassifier
from analytics.statistics import StatisticsManager
from visualization.overlay import Visualizer

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("processing.log"),
            logging.StreamHandler()
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="Padel Shot Classification System")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--max_frames", type=int, default=None, help="Limit frames for testing")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("Main")

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize components
    video_proc = VideoProcessor(
        input_path=config['video']['input_path'],
        output_path=config['video']['output_path'],
        frame_skip=config['video']['frame_skip'],
        resize_factor=config['video']['resize_factor']
    )
    
    player_det = PlayerDetector(config)
    
    ball_track = BallTracker(config)
    shot_class = ShotClassifier(config)
    stats = StatisticsManager(
        cooldown_frames=config['shot_classification'].get('shot_buffer_frames', 30),
        start_time_str=config['video'].get('start_time')
    )
    viz = Visualizer(config)

    logger.info("Starting pipeline processing...")

    latest_shot = "None"
    
    try:
        for frame_idx, frame in video_proc.get_frames():
            if args.max_frames and frame_idx >= args.max_frames:
                break

            # 1. Detection
            players = player_det.detect(frame)
            poses = player_det.get_poses(frame, players)
            ball = ball_track.track(frame, players=players)

            # 2. Classification Logic
            if ball:
                # Find player closest to ball for hit analysis
                # (This is a simplification, ideally check racket intersection)
                for player in players:
                    if player['id'] in poses:
                        kpts = poses[player['id']]
                        ball_pos = ball['center']
                        
                        # Distance check for potential hit
                        p_center = ((player['bbox'][0] + player['bbox'][2])/2, (player['bbox'][1] + player['bbox'][3])/2)
                        dist = ((p_center[0] - ball_pos[0])**2 + (p_center[1] - ball_pos[1])**2)**0.5
                        
                        # Hit Detection: Check for ball-racket interaction
                        is_hit = False
                        if 'racket_bbox' in player:
                            rb = player['racket_bbox']
                            # Check if ball center is inside or very near racket bbox
                            if (rb[0]-10 <= ball_pos[0] <= rb[2]+10 and 
                                rb[1]-10 <= ball_pos[1] <= rb[3]+10):
                                is_hit = True
                        else:
                            # Fallback to proximity if racket not detected
                            if dist < 60: 
                                is_hit = True

                        if is_hit:
                            ball_vel = ball.get('velocity', 0)
                            # Real shot if speed is sufficient AND movement is somewhat horizontal
                            # (Filters out vertical hand-bounces)
                            min_speed = config['shot_classification'].get('min_ball_speed', 25.0)
                            is_real_shot = ball_vel > min_speed
                            if len(ball_track.ball_history) >= 2:
                                p1 = ball_track.ball_history[-2]
                                p2 = ball_track.ball_history[-1]
                                dx = abs(p2[0] - p1[0])
                                dy = abs(p2[1] - p1[1])
                                if dy > dx * 2: # Mostly vertical movement
                                    is_real_shot = False

                            if is_real_shot:
                                shot_type, conf = shot_class.classify(kpts, ball_pos, player['bbox'])
                                if shot_type != "Unknown":
                                    if stats.add_shot(frame_idx, frame_idx/video_proc.fps, shot_type, player['id'], conf):
                                        latest_shot = shot_type

            # 3. Visualization
            annotated_frame = frame.copy()
            for player in players:
                viz.draw_player(annotated_frame, player)
                viz.draw_racket(annotated_frame, player)
                if player['id'] in poses:
                    viz.draw_pose(annotated_frame, poses[player['id']])
            
            viz.draw_ball(annotated_frame, ball)
            viz.draw_shot_info(annotated_frame, latest_shot, stats.get_summary())

            # 4. Output
            video_proc.write_frame(annotated_frame)

            if frame_idx % 100 == 0:
                logger.info(f"Processed up to frame {frame_idx}")

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}", exc_info=True)
    finally:
        video_proc.release()
        stats.save_reports(
            json_path="outputs/shots/shot_predictions.json",
            csv_path="outputs/shots/shot_predictions.csv"
        )

if __name__ == "__main__":
    main()
