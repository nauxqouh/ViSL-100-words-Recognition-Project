import cv2
import mediapipe as mp
import numpy as np
from config.settings import N_HAND_LANDMARKS, N_POSE_LANDMARKS, UPPER_BODY_CONNECTIONS, N_LANDMARKS

def mediapipe_detection(image, model):
    """Convert color space and run Mediapipe model."""
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results, visibility_thres=0.5):
    """Extract all keypoints from one video frame."""
    
    # Upper pose
    pose_landmarks = []
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        for idx, i in enumerate(UPPER_BODY_CONNECTIONS):
            if i < len(landmarks):
                res = landmarks[i]
                if res.visibility < visibility_thres:         
                    pose_landmarks.append([0.0, 0.0, 0.0])
                else:
                    pose_landmarks.append([res.x, res.y, res.z])
    else:
        pose_landmarks = [[0.0, 0.0, 0.0]] * N_POSE_LANDMARKS
    
    # Left hand
    if results.left_hand_landmarks:
        left_hand_landmarks = [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
    else:
        left_hand_landmarks = [[0.0, 0.0, 0.0]] * N_HAND_LANDMARKS

    # Right hand
    if results.right_hand_landmarks:
        right_hand_landmarks = [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
    else:
        right_hand_landmarks = [[0.0, 0.0, 0.0]] * N_HAND_LANDMARKS
    
    return pose_landmarks, left_hand_landmarks, right_hand_landmarks


def plot_keypoints(list_landmarks):
    """Plot image with keypoints"""
    
    pass