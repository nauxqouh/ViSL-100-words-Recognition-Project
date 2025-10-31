import os
import mediapipe as mp
import cv2
import src.MediaPipeProcess.keypoint_extract as md
import numpy as np
from config.settings import K
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

mp_holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def get_list_frame(source_path):
    """Extract keypoints sequences from one video."""
    
    frames_keypoints = []
    cap = cv2.VideoCapture(source_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // 100)
    
    while cap.isOpened():
        # read video frame
        success, image = cap.read()

        # skip empty frames
        if not success:
            break
        
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % step != 0:
            continue
        
        try:
            # MediaPipe Holistic processing
            _, results = md.mediapipe_detection(image, mp_holistic)
            pose_landmarks, left_hand_landmarks, right_hand_landmarks = md.extract_keypoints(results)
            frames_keypoints.append([left_hand_landmarks, right_hand_landmarks, pose_landmarks])
        except Exception as e:
            continue
        
    cap.release()
    return frames_keypoints
    
def concate_array(left_hand_landmarks, right_hand_landmarks, pose_landmarks):
    """ """
    a1 = np.array(left_hand_landmarks).reshape(-1)
    a2 = np.array(right_hand_landmarks).reshape(-1)
    a3 = np.array(pose_landmarks).reshape(-1)
    result = np.concatenate((a1, a2, a3), axis=None)
    return result

def check_zeros(list_landmarks):
    data = np.array(list_landmarks)
    if np.all(data == 0):
        return True
    return False
    
def write_data(output_dir, source_path, file_name):
    """Write keypoints sequence into numpy file from original video."""
    
    try:
        list_fr = get_list_frame(source_path)
        X = []
        list_idx = []
        for i in range(len(list_fr)):
            if check_zeros(list_fr[i][0]) and check_zeros(list_fr[i][1]):
                continue
            X.append(concate_array(list_fr[i][0], list_fr[i][1], list_fr[i][2]))
            list_idx.append(i)
        if len(X) == 0:
            print("no valid frame to save in: " + source_path)
            return
        
        # filtering frame
        X_new = np.array(X)
        kmeans = KMeans(n_clusters=K)
        kmeans.fit(X_new)
        cluster_centers = kmeans.cluster_centers_
        distances = cdist(X_new, cluster_centers, 'euclidean')
        nearest_indices = np.argmin(distances, axis=0)
        index = np.sort(nearest_indices)
        
        data = []
        for i in index:
            data.append(list_fr[list_idx[i]])
        data_save = np.asarray(data, dtype="object")
        np.save(os.path.join(output_dir, file_name), data_save)
        print("ok write npy from file: " + source_path) 
    except Exception as e:
        print(f"error write: {source_path} with {e}")
        
def write_data_v1(output_dir, source_path, file_name):
    """Write keypoints sequence into numpy file from original video."""
    
    try:
        list_fr = get_list_frame(source_path)
        data_save = np.asarray(list_fr, dtype="object")
        np.save(os.path.join(output_dir, file_name), data_save)
        print("ok write npy from file: " + source_path) 
    except Exception as e:
        print(f"error write: {source_path} with {e}")
        
def interpolate_keypoints(keypoints_sequence, target_len = 30):
    try:
        list_fr = keypoints_sequence.copy()
        X = []
        list_idx = []
        for i in range(len(list_fr)):
            if check_zeros(list_fr[i][0]) and check_zeros(list_fr[i][1]):
                continue
            X.append(concate_array(list_fr[i][0], list_fr[i][1], list_fr[i][2]))
            list_idx.append(i)
        if len(X) == 0:
            print("no valid frame to save")
            return
        
        # filtering frame
        X_new = np.array(X)
        kmeans = KMeans(n_clusters=target_len)
        kmeans.fit(X_new)
        cluster_centers = kmeans.cluster_centers_
        distances = cdist(X_new, cluster_centers, 'euclidean')
        nearest_indices = np.argmin(distances, axis=0)
        index = np.sort(nearest_indices)
        
        data = []
        for i in index:
            data.append(list_fr[list_idx[i]])
        data_save = np.asarray(data, dtype="object")
        return data_save
    except Exception as e:
        print(f"error process with {e}")
        return None