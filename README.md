# Sports Video Analysis Pipeline

This project is a Python-based video processing pipeline designed to analyze sports videos, specifically focusing on tracking players and the ball, estimating camera movements, assigning team colors, and calculating player statistics such as speed and distance.

## Features

- **Object Tracking**: Utilizes a YOLO model to detect and track players, referees, and the ball across video frames.
- **Camera Movement Estimation**: Uses optical flow to adjust object positions based on camera movements.
- **Team Assignment**: Assigns players to teams using KMeans clustering based on jersey colors.
- **Ball Possession Detection**: Determines which player has possession of the ball by calculating proximity.
- **Speed and Distance Calculation**: Computes speed and distance metrics for each player and annotates them on video frames.
- **Perspective Transformation**: Transforms object positions to accurately reflect their locations on the court.

## Installation

1. **Create a Virtual Environment**:
   ```bash
   python -m venv fbaenv
   source fbaenv/bin/activate  # On Windows use `fbaenv\Scripts\Activate`

2. **Install Dependencies**:
Install PyTorch and Ultralytics for YOLO model support.


3. **Download Required Models**:
Use Roboflow or other sources to download annotated datasets.
Train a YOLOv5 model using Colab or similar platforms.


## Usage
1. **Prepare Video Files**:
Place your video files in the designated input directory.

2. **Run the Main Script**


3. **Output**:
The processed video with annotations will be saved to the specified output path.


## Modules Overview
main.py: Orchestrates the entire video processing workflow.

camera_movement_estimator.py: Estimates and adjusts for camera movement between frames.

player_ball_assigner.py: Assigns ball possession to players based on proximity.

speed_and_distance_estimator.py: Calculates and annotates speed and distance metrics for players.

team_assigner.py: Assigns team colors using KMeans clustering.

tracker.py: Handles object detection and tracking using YOLO models.

view_transformer.py: Performs perspective transformation of player positions.
Utilities

video_utils.py: Contains functions to read and save video files.

bbox_utils.py: Provides utility functions for bounding box calculations.


## Notes
Ensure that the YOLO model is properly trained and placed in the correct directory as specified in your configuration files.
Adjust parameters such as frame rate and window size in speed_and_distance_estimator.py as needed for your specific use case.
Acknowledgments
This project utilizes various open-source libraries and models. Special thanks to Roboflow for providing annotated datasets used in training.
text

## Output
![Screenshot (879)](https://github.com/user-attachments/assets/19560946-685f-4b88-b1d7-e3dff9856a14)

The output contains:
   1. ellipses around every player detected and referee
   2. Markers over the ball and player in possession of the ball
   3. Percentage of team possession
   4. Coordinates of camera movement
   5. Total distance each player has run
   6. Current speed fo each detected player




