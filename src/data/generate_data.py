import src.MediaPipeProcess.create_point as my_func
import numpy as np
import os
from src.utils.utils import create_directory
from config.paths import keypoint_dir, augmented_data_dir

def generate_data_v1(source_sample, path_to_save, n_samples):
    """Generate new data for one numpy file."""
    
    n_name = 0
    if n_name < n_samples:
        try:
            for index in range(n_name, n_samples):
                new_file = path_to_save + "/" + str(index) + ".npy"
                result = []
                frame_0 = my_func.create_frame_0(source_sample)
                # print(frame_0)
                result.append(frame_0)
                for t in range(1, 20):
                    frame_i = my_func.create_frame_t(t, source_sample, frame_0)
                    result.append(frame_i)
                data_save = np.asarray(result, dtype="object")
                np.save(new_file, data_save)
        except Exception as e:
            print(f"error when generate augmented data: {e}.")

def generate_data_v2(source_sample, path_to_save, n_samples):
    """Generate new data for one numpy file."""
    
    from src.MediaPipeProcess.create_numpy_data import interpolate_keypoints
    
    n_name = 0
    if n_name < n_samples:
        try:
            for index in range(n_name, n_samples):
                new_file = path_to_save + "/" + str(index) + ".npy"
                result = []
                frame_0 = my_func.create_frame_0(source_sample)
                # print(frame_0)
                result.append(frame_0)
                for t in range(1, source_sample.shape[0]):
                    frame_i = my_func.create_frame_t(t, source_sample, frame_0)
                    result.append(frame_i)   
                augmented_result = interpolate_keypoints(result)
                data_save = np.asarray(augmented_result, dtype="object")
                np.save(new_file, data_save)
        except Exception as e:
            print(f"error when generate augmented data: {e}.")
            
def generate_full_data(path_source, des_dir, n_samples):
    """Generate full data pipeline."""
    
    list_dir = os.listdir(path_source)
    for sub_dir in list_dir:
        try:
            create_directory(des_dir, sub_dir)
            for file_name in os.listdir(os.path.join(path_source, sub_dir)):
                path_file = os.path.join(os.path.join(path_source, sub_dir), file_name)
                if os.path.isfile(path_file):
                    source_sample = np.load(path_file, allow_pickle=True)
                    path_to_save = os.path.join(des_dir, sub_dir)
                    # generate_data_v1(source_sample, path_to_save, n_samples)
                    generate_data_v2(source_sample, path_to_save, n_samples)
                    print(f"ok generate new data for {sub_dir}")
        except Exception as e:
            print(f"error when generate full augmented data: {e}.")
            
if __name__ == "__main__":
    # generate_full_data(keypoint_dir, 'dataset/new_data_v2', 100)
    generate_full_data('dataset/keypoints_v1', 'dataset/new_data_v3', 100)