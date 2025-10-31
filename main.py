import os
import tensorflow as tf
from config.settings import N_HAND_LANDMARKS, N_POSE_LANDMARKS, N_LANDMARKS, UPPER_BODY_CONNECTIONS, K
import cv2
import mediapipe as mp
from src.MediaPipeProcess.keypoint_extract import mediapipe_detection, extract_keypoints
import streamlit as st
import numpy as np
import time

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('Models/checkpoints/ViSL_model_v3.keras') 

@st.cache_data
def load_label_map():
    import json
    with open('logs/label_map.json', 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    inv_label_map = {v: k for k, v in label_map.items()}
    return label_map, inv_label_map

model = load_model()
label_map, inv_label_map = load_label_map()

def process_webcam_to_sequence():
    cap = cv2.VideoCapture(0)
    st.write("⏳ Đang chuẩn bị... Bắt đầu trong 3 giây...")
    time.sleep(3)
    
    st.write("🎥 Đang ghi hình trong 10 giây...")
    sequence = []
    start_time = time.time()

    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Không thể truy cập webcam")
            break
        elapsed_time = time.time() - start_time
        if elapsed_time > 10:
            break
        image, results = mediapipe_detection(frame, holistic)

        pose_landmarks, left_hand_landmarks, right_hand_landmarks = extract_keypoints(results)
        sequence.append([left_hand_landmarks, right_hand_landmarks, pose_landmarks])

        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)
        )
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
        )
        stframe.image(image, channels="BGR", caption="Webcam feed", use_container_width=True)

    cap.release()
    
    return sequence

def process_video_to_sequence(video_file):
    """
    Xử lý video upload để trích xuất chuỗi keypoints.
    """
    st.write("🎬 Đang xử lý video...")
    temp_path = "temp_video.mp4"
    with open(temp_path, "wb") as f:
        f.write(video_file.read())

    cap = cv2.VideoCapture(temp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // 100)
        
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    sequence = []
    stframe = st.empty()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % step != 0:
            continue
        
        image, results = mediapipe_detection(frame, holistic)
        pose_landmarks, left_hand_landmarks, right_hand_landmarks = extract_keypoints(results)
        sequence.append([left_hand_landmarks, right_hand_landmarks, pose_landmarks])
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)
        )
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
        )
        frame_count += 1
        if frame_count % 2 == 0:
            stframe.image(image, channels="BGR", caption=f"Đang xử lý frame {frame_count}", use_container_width=True)

    cap.release()
    st.success(f"Xử lý xong video ({frame_count} frames).")
    return sequence

    
st.set_page_config(page_title="VSL Prediction", layout="centered")
st.title("DỰ ĐOÁN NGÔN NGỮ KÝ HIỆU")

mp_holistic = mp.solutions.holistic
sequence = None
holistic =mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

input_mode = st.radio("Chọn nguồn đầu vào:", ["🎞️ Video file", "📷 Webcam"])

if input_mode == "🎞️ Video file":
    uploaded_file = st.file_uploader("📂 Chọn video để dự đoán", type=["mp4", "avi"])
    if uploaded_file is not None:
        try:
            sequence = process_video_to_sequence(uploaded_file)
        except:
            st.warning("Vui lòng tải lên file video hợp lệ.")
            
elif input_mode == "📷 Webcam":
    sequence = process_webcam_to_sequence()

if sequence is not None and len(sequence) > 0:
    st.success(f"Đã thu được {len(sequence)} frame keypoints.")
    
    # Lọc và chọn 20 frame đặc trưng
    from src.MediaPipeProcess.create_numpy_data import interpolate_keypoints
    sequence_filtered = interpolate_keypoints(sequence, 30)
    if sequence_filtered is None:
        st.error("Không có frame hợp lệ để xử lý.")
    else:
        st.success(f"{sequence_filtered.shape}")
        from src.MediaPipeProcess.create_numpy_data import concate_array
        processed_frames = [concate_array(f[0], f[1], f[2]) for f in sequence_filtered]
        X_input = np.array(processed_frames, dtype=np.float32)

        if X_input.shape != (30, 177):
            st.warning(f"Dữ liệu đầu vào không đúng shape (hiện tại: {X_input.shape})")
        else:
            st.success(np.expand_dims(X_input, axis=0))
            result = model.predict(np.expand_dims(X_input, axis=0))
            pred_idx = np.argmax(result, axis=1)[0]
            pred_label = inv_label_map[pred_idx]
            confidence = np.max(result)

            st.subheader("Kết quả dự đoán:")
            st.metric(label="Nhãn dự đoán", value=pred_label)
            st.progress(float(confidence))

else:
    st.error("Không có dữ liệu keypoints để xử lý.")
