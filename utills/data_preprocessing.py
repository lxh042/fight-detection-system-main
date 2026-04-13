# This function convert videos into keypoint using yolo11 model

import os
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.preprocessing import MinMaxScaler  # 新增导入

yoloModel = YOLO('yolo11n-pose.pt')



# This function extract 41 frames from each video and detect 3 person each video and then 
# extract total 153 keypoints per video 
# Output shape : (videos , 41, 153)
def extract_pose_features(root_folder, mode):
    """
    Extract pose features from videos.
    
    Parameters:
    - root_folder: Path to the  dataset.
    - mode: Either 'train' or 'test'.
    - model: Pre-trained YOLO model for pose estimation.

    Returns:
    - X: NumPy array of shape (num_videos, frames, features)
    - y: NumPy array of labels (1 for violence, 0 for non-violence)
    """
    sequenceKeyPoints = []
    labels = []

    max_persons = 3  # Fixed number of persons per frame
    num_keypoints = 17  # YOLO detects 17 keypoints per person
    features_per_person = num_keypoints * 3  # (x, y, confidence)
    fixed_frames = 41  # Fixed number of frames per video

    dataset_path = os.path.join(root_folder, mode)

    for category, label in zip(['violence', 'non-violence'], [1, 0]):
        category_path = os.path.join(dataset_path, category)

        for video_name in os.listdir(category_path):
            video_path = os.path.join(category_path, video_name)
            cap = cv2.VideoCapture(video_path)

            #2. Extracting Each Frame keypoints from video
            videoKeypoints = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = yoloModel.predict(frame, verbose=False)  # YOLO model inference

                #2. Extracting Each person keypoints from one frame
                frame_keypoints = []
                for result in results:
                    keypoints = result.keypoints.data.cpu().numpy()

                    for i, person_keypoints in enumerate(keypoints):
                        if person_keypoints.shape[0] != num_keypoints:
                            continue  # Skip if keypoints are missing

                        flattened = person_keypoints.flatten()
                        frame_keypoints.append(flattened)

                # Ensuring each frames detect exactly three persons
                print("Frame Keypoints Before" , len(frame_keypoints))
                while len(frame_keypoints) < max_persons:
                    frame_keypoints.append(np.zeros(features_per_person))

                # Trim to max_persons
                frame_keypoints = frame_keypoints[:max_persons]
                print("Frame Keypoints After" , len(frame_keypoints))
                # Flatten keypoints for all persons
                flattened_frame_keypoints = np.array(frame_keypoints).flatten()
                videoKeypoints.append(flattened_frame_keypoints)

            cap.release()
            cv2.destroyAllWindows()

            # Ensure each video has exactly 41 frames
            print("Frames Before" , len(videoKeypoints))
            while len(videoKeypoints) < fixed_frames:
                videoKeypoints.append(np.zeros_like(videoKeypoints[0]))
            videoKeypoints = videoKeypoints[:fixed_frames]
            print("Frames After" , len(videoKeypoints))

            scaler = MinMaxScaler()
            videoKeypoints = scaler.fit_transform(np.array(videoKeypoints))


            sequenceKeyPoints.append(np.array(videoKeypoints))

            labels.append(label)

    return np.array(sequenceKeyPoints), np.array(labels)

# 新增：特征缓存/加载工具与统一入口
def _cache_file_path(root_folder, mode, cache_dir="./preprocessed"):
    os.makedirs(cache_dir, exist_ok=True)
    safe_mode = str(mode).replace(os.sep, "_")
    return os.path.join(cache_dir, f"{safe_mode}_features_v1.npz")

def save_features(X: np.ndarray, y: np.ndarray, path: str):
    np.savez_compressed(path, X=X, y=y)

def load_features(path: str):
    if os.path.exists(path):
        data = np.load(path, allow_pickle=False)
        return data["X"], data["y"]
    return None, None

def prepare_features(root_folder, mode, cache_dir="./preprocessed", force_recompute=False):
    """
    优先从缓存加载已提取特征；如无缓存或强制重算，则进行特征提取并保存。
    返回：X, y
    """
    cache_path = _cache_file_path(root_folder, mode, cache_dir)
    if not force_recompute:
        X, y = load_features(cache_path)
        if X is not None and y is not None:
            return X, y

    X, y = extract_pose_features(root_folder, mode)
    save_features(X, y, cache_path)
    return X, y
