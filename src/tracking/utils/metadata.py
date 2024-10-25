import numpy as np
import json
import os
import argparse

def convert_opencv_to_opengl(w2c_opencv):
    """
    Convert extrinsics from OpenCV format to OpenGL format.
    """
    R = w2c_opencv[:3, :3]
    t = w2c_opencv[:3, 3].reshape(3, 1)
    R_opengl = R.T
    t_opengl = -R.T @ t
    w2c_opengl = np.hstack((R_opengl, t_opengl))
    w2c_opengl = np.vstack((w2c_opengl, np.array([0, 0, 0, 1])))

    return w2c_opengl

def load_camera_parameters(cam_dir):
    """Load camera parameters from a directory."""
    cam_extr_file = os.path.join(cam_dir, "camera_extrinsics.npy")
    cam_intr_file = os.path.join(cam_dir, "camera_params.npy")
    
    w2c = np.load(cam_extr_file)
    w2c = np.linalg.inv(w2c)
    w2c = convert_opencv_to_opengl(w2c)
    fx, fy, cx, cy = np.load(cam_intr_file)
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    return k, w2c

def extract_image_data(cam_dir, foreground_dir, step=1, start_index=0, num_images=200):
    """Extract image file names and associated data from a specified index and limit the number of images."""
    cam_id = os.path.basename(cam_dir)
    file_list = os.listdir(foreground_dir)
    image_list = [f for f in file_list if 'foreground' in f and f.endswith(".png")]
    image_list.sort(key=lambda n: int(n[:-4].split('_')[-1]) if n[:-4].split('_')[-1].isdigit() else 0)
    
    # Ensure start_index is within the range of image_list
    start_index = max(0, min(start_index, len(image_list) - 1))
    
    # If num_images is specified and within range, adjust the end_index
    if num_images is not None:
        end_index = start_index + step * num_images
    else:
        end_index = len(image_list)
    
    # Select every 'step'th image starting from 'start_index', up to 'end_index'
    image_list = image_list[start_index:end_index:step]
    
    return [os.path.join(cam_id, 'foreground', img) for img in image_list], cam_id

def main():
    argparser = argparse.ArgumentParser()
    args = argparser.parse_args()
    data_path = args.data_path
    data_list = os.listdir(data_path)
    cam_path = []
    foreground_path = []
    for item in data_list:
        if item.startswith("camera"):
            cam_path.append(os.path.join(data_path, item))
            foreground_path.append(os.path.join(data_path, item, 'foreground'))

    fn_list = []
    per_cam_k_list = []
    per_cam_w2c_list = []
    per_cam_id_list = []

    # Determine the minimum number of frames across all foreground directories
    min_num_images = float('inf')
    for foreground_dir in foreground_path:
        file_list = os.listdir(foreground_dir)
        image_list = [f for f in file_list if 'foreground' in f and f.endswith(".png")]
        num_images = len(image_list)
        if num_images < min_num_images:
            min_num_images = num_images

    # Process each camera directory
    for cam_dir, foreground_dir in zip(cam_path, foreground_path):
        k, w2c = load_camera_parameters(cam_dir)
        
        # Extract image data
        images, cam_id = extract_image_data(cam_dir, foreground_dir, num_images=200)
        k_list = [k] * len(images)
        w2c_list = [w2c] * len(images)
        cam_id_list = [cam_id] * len(images)
        
        # Append to the per-camera lists
        per_cam_k_list.append(k_list)
        per_cam_w2c_list.append(w2c_list)
        fn_list.append(images)
        per_cam_id_list.append(cam_id_list)
    
    meta = {
        'w': 1280,
        'h': 720,
        'k': np.array(per_cam_k_list).transpose(1, 0, 2, 3).tolist(),
        'w2c': np.array(per_cam_w2c_list).transpose(1, 0, 2, 3).tolist(),
        'fn': np.array(fn_list).transpose(1, 0).tolist(),
        'cam_id': np.array(per_cam_id_list).transpose(1, 0).tolist()
    }

    with open(os.path.join(data_path, 'train_meta.json'), 'w') as f:
        json.dump(meta, f)

if __name__ == "__main__":
    main()
