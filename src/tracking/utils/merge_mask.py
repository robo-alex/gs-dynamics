import cv2
import numpy as np
import os
import argparse

def merge(rgb, mask, foreground_path, idx):
    # Check if the dimensions of the mask and the image match
    if rgb.shape[:2] != mask.shape:
        raise ValueError("The size of the RGB image and the binary mask must match.")

    # Convert the binary mask to a 3-channel image (same dimensions as RGB image)
    mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Convert the binary mask to boolean
    mask_bool = mask_3channel.astype(bool)

    # Use the binary mask to select the foreground from them
    foreground = np.zeros_like(rgb)
    foreground[mask_bool] = rgb[mask_bool]

    cv2.imwrite(os.path.join(foreground_path, f'foreground_{idx:06d}.png'), foreground)
    print(f"Saved foreground_{idx:06d}.png to {foreground_path}")

def sort_key(filename):
    return int(filename.split('_')[-1].split('.')[0])

def process_images(data_path):
    data_list = os.listdir(data_path)

    # Initialize lists to hold paths
    cam_path = []
    seg_path = []
    foreground_path = []

    # Populate the lists with the corresponding paths
    for entry in data_list:
        if entry.startswith("camera"):
            base_path = os.path.join(data_path, entry)
            cam_path.append(base_path)
            seg_path.append(os.path.join(base_path, 'seg'))
            foreground_path.append(os.path.join(base_path, 'foreground'))

    # Retrieve and sort the RGB image paths
    rgb_image_list = sorted([f for f in os.listdir(cam_path[0]) if f.endswith(".jpg")], key=sort_key)
    
    # Determine the minimum number of frames across all segmentation directories
    min_num_frames = float('inf')
    for seg_dir in seg_path:
        mask_image_list = [f for f in os.listdir(seg_dir) if f.startswith("seg") and f.endswith(".png")]
        mask_image_list.sort(key=sort_key)
        num_frames = len(mask_image_list)
        if num_frames < min_num_frames:
            min_num_frames = num_frames

    # Process images and masks
    for cam_dir, seg_dir, foreground_dir in zip(cam_path, seg_path, foreground_path):
        # Create the foreground directory if it does not exist
        os.makedirs(foreground_dir, exist_ok=True)

        # Retrieve and sort the mask image paths, and limit to min_num_frames
        mask_image_list = sorted([f for f in os.listdir(seg_dir) if f.startswith("seg") and f.endswith(".png")], key=sort_key)[:min_num_frames]

        for idx, (rgb_filename, mask_filename) in enumerate(zip(rgb_image_list[:min_num_frames], mask_image_list)):
            rgb_image_path = os.path.join(cam_dir, rgb_filename)
            mask_image_path = os.path.join(seg_dir, mask_filename)

            rgb_image = cv2.imread(rgb_image_path)
            mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

            merge(rgb_image, mask_image, foreground_dir, idx)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    args = argparser.parse_args()
    data_path = args.data_path
    for i in range(0, 4):
        process_images(data_path)
