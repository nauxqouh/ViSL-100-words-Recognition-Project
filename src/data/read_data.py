import os
from src.MediaPipeProcess.create_numpy_data import write_data, write_data_v1
import src.utils.utils as utils
from config.paths import data_dir, keypoint_dir
import cv2 

def capture_frame(source_path, des_dir):
    # Capture video frames as images
    cap = cv2.VideoCapture(source_path)
    os.makedirs(des_dir, exist_ok=True)
    print("Capture video frames processing...")
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            frame_id += 1
            break
        save_path = os.path.join(des_dir, f"frame_{frame_id}.jpg")
        cv2.imwrite(save_path, frame)
        print(f"Saved: {save_path}")
        frame_id += 1
    cap.release()
    print("Done saving frames")
    
def extract_keypoint_from_videos(root_dir, des_dir):
    for sub_dir in os.listdir(root_dir):
        path = os.path.join(root_dir, sub_dir)
        if os.path.isdir(path):
            utils.create_directory(des_dir, sub_dir)
            new_path = os.path.join(des_dir, sub_dir)
            list_file = utils.list_files(path)
            for i in range(len(list_file)):
                path_to_file = os.path.join(path, list_file[i])
                print(i+1)
                # write_data(new_path, path_to_file, str(i) + ".npy")
                write_data_v1(new_path, path_to_file, str(i) + ".npy")
                
if __name__ == "__main__":
    extract_keypoint_from_videos(data_dir, 'dataset/keypoints_v1')