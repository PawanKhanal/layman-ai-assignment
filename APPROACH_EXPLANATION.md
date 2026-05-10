# Padel Shot Classification: Approach & Methodology

## 1. Methodology
The system is built as a modular computer vision pipeline designed for real-time analysis of Padel match footage. The architecture is divided into four distinct layers:

*   **Detection Layer**: Uses **YOLOv8n** for player and racket detection. By explicitly detecting the racket, we ensure that shot events are tied to physical contact rather than just proximity to a player.
*   **Tracking Layer**: Implements a persistent tracking system for players and a motion-sensitive tracker for the ball. We use a **Region of Interest (ROI)** filter to focus exclusively on the main court, ignoring background matches and spectators.
*   **Classification Layer**: A rule-based heuristic engine that analyzes the spatial relationship between the ball, the player's torso, and their detected racket. It handles **Spatial Inversion** (Top vs. Bottom court) to ensure shot types are accurate regardless of the player's orientation.
*   **Analytics Layer**: Aggregates detections into a structured format (JSON/CSV) and generates a live visual overlay with shot counts and event indicators (like "BOUNCE").

---

## 2. Challenges Faced (The "Human" Side)
Developing this system required solving several "real-world" problems that a standard detection model wouldn't catch:

*   **The "Double-Hit" Problem**: Initially, the system logged 3–5 shots for every single swing because the ball stayed near the racket for several frames. I solved this by implementing a **Temporal Cooldown** and a **Velocity Impulse** check—ensuring only the initial impact is counted.
*   **Distinguishing Play from Preparation**: A major challenge was players bouncing the ball in their hand or tossing it to a teammate. These "casual" movements look like shots to a basic AI. I implemented a **"Body-Escape" Rule**: if the ball is moving slowly or is inside the player's personal space, it is ignored until it "breaks away" at match speed.
*   **Perspective & Orientation**: Players at the top of the court are facing the camera, while bottom players face away. This means their Left and Right are swapped. I wrote logic to detect the player's court side and **invert the classification rules**, so a Forehand is always a Forehand regardless of where the player stands.
*   **Codec & Compatibility**: We initially faced "unplayable video" errors due to missing H.264 libraries on the local system. We pivoted to a more robust **XVID/AVI** combination to ensure the results could be viewed on any standard Windows machine.

---

## 3. Future Improvements
If I were to take this system to a production level, I would implement:

*   **Deep Learning Classification**: Replace the current heuristic rules with a Temporal Shift Module (TSM) or a GRU-based neural network trained on thousands of Padel swings for near-perfect accuracy.
*   **Automated Court Calibration**: Use Hough Transforms or a specialized "Court-Net" to automatically detect the lines and net, removing the need for manual ROI configuration.
*   **Kalman Filter Tracking**: Use Kalman Filters to predict the ball's trajectory during "occlusions" (when the ball is hidden behind a player's body).
*   **Speed Measurement**: Convert "pixels-per-frame" into "kilometers-per-hour" by using the known dimensions of a Padel court to provide professional-grade match data.

---

## 4. Model Links
The models used in this project are standard YOLOv8 pre-trained weights. You can find them here:
*   **Player & Racket Model (YOLOv8n)**: [https://drive.google.com/drive/folders/1pedc0wXiyHWh5Q5CcVzocpBl9nyNIe0C?usp=sharing]

