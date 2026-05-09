# Padel Game Analytics — Shot Classification System

## Project Overview
This project is a Computer Vision prototype designed to analyze Padel gameplay footage. It performs real-time detection and tracking of players, rackets, and the ball, while classifying key shots (Forehand, Backhand, and Smash/Serve).

## Core Features
- **Object Detection & Tracking**: Uses YOLOv8 for robust tracking of players, rackets, and balls.
- **Pose Estimation**: Leverages MediaPipe Pose to analyze player movements during hits.
- **Shot Classification**: A rule-based system that combines ball trajectory, racket proximity, and player pose to identify shots.
- **Analytics**: Generates a summary of shot counts and detailed frame-by-frame results in CSV/JSON formats.

## Tech Stack
- **Python 3.8+**
- **OpenCV**: Video processing and visualization.
- **YOLOv8 (Ultralytics)**: Object detection and tracking.
- **MediaPipe**: Pose estimation.
- **Pandas/NumPy**: Data handling and analysis.

## Setup Instructions
1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd padel-analytics
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the analysis**:
   ```bash
   python main.py --input path/to/video.mp4 --output output_video.mp4
   ```

## Methodology & Approach
1. **Detection**: We use a pre-trained YOLOv8 model to detect `person`, `sports ball`, and `tennis racket` classes.
2. **Hit Detection**: A hit is registered when the ball enters the bounding box of a racket and its velocity vector changes significantly.
3. **Classification**:
   - **Smash/Serve**: Identified if the hit occurs above the player's head level.
   - **Forehand/Backhand**: Determined by the horizontal position of the hit relative to the player's center and pose keypoints.
4. **Data Export**: All detections are saved to `output_results.csv` and a summary is generated in `summary.json`.

## Challenges Faced
- **Ball Speed**: High-speed balls are often blurred or missed in low FPS videos. We mitigate this using frame-by-frame tracking and proximity thresholds.
- **Racket Specificity**: Generic "tennis racket" models may sometimes miss Padel-specific rackets. Future improvements would include fine-tuning on a Padel-specific dataset.

## Future Improvements
- **Kalman Filter**: Implement a Kalman filter for more robust ball tracking through occlusions.
- **Action Recognition**: Use a 3D-CNN or LSTM on pose sequences for more accurate shot classification beyond simple rules.
- **Court Mapping**: Detect the court lines to analyze shot placement and bounce detection.
