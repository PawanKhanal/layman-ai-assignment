# Padel Shot Classification System

Computer vision system for analyzing Padel match footage. It detects players, tracks the ball, and classifies key shots like forehands, backhands, and serves.

## Quick Start

1. **Setup Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Analysis**:
   ```bash
   python src/main.py --config configs/default.yaml
   ```

## Key Features
- **Object Tracking**: Robust player tracking using YOLOv8.
- **Pose Estimation**: Joint analysis for identifying shot mechanics.
- **Shot Classification**: Rule-based logic for identifying Serve, Forehand, and Backhand.
- **Data Export**: Generates detailed JSON and CSV reports of the match.

## Project Structure
- `src/`: Core logic and processing modules.
- `configs/`: YAML configuration files.
- `data/`: Input and reference data.
- `models/`: Pre-trained YOLO and Pose models.
- `outputs/`: Processed videos and analytical reports.

## Methodology
The system follows a pipeline approach:
1. **Player Detection**: YOLOv8 locates players in each frame.
2. **Pose Analysis**: Extracting keypoints to determine body positioning.
3. **Collision Detection**: Monitoring ball trajectory relative to players.
4. **Classification**: Categorizing shots based on swing height and direction.

## Future Improvements
- Enhanced ball tracking using Kalman Filters.
- Court line detection for automated score tracking.
- Integration with a web-based dashboard for match reviews.
