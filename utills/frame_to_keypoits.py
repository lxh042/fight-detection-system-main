import numpy as np
import os

from ultralytics import YOLO


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, 'yolo11n-pose.pt')

# Load models
yoloModel = YOLO(YOLO_MODEL_PATH)

# Keypoints Extracting using YOLO
def extract_keypoints(frame):
    max_persons = 3
    num_keypoints = 17
    features_per_person = num_keypoints * 3
    results = yoloModel.predict(frame, verbose=False)
    result = results[0]
    frame_keypoints = []
    for result in results:
        keypoints = result.keypoints.data.cpu().numpy()
        for person_keypoints in keypoints:
            if person_keypoints.shape[0] != 17:
                continue
            flattened = person_keypoints.flatten()
            frame_keypoints.append(flattened)

    while len(frame_keypoints) < max_persons:
        frame_keypoints.append(np.zeros(features_per_person))

    frame_keypoints = frame_keypoints[:max_persons]
    flattened_frame_keypoints = np.array(frame_keypoints).flatten()
    return flattened_frame_keypoints, result
