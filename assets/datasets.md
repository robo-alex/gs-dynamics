## Real World Datasets
```
- Data
  -- camera_*: Raw data captured from 4 RealSense cameras
    --- idx.jpg    : RGB images
    --- idx_depth.png  : Depth images
    --- seg/    : Segmentation masks (mask_idx.png)
    --- foreground/ : Foreground masks of the object (foreground_idx.png)
```

## Rope and Toy Animals
```
weight_im_values=50.0
weight_rigid_values=200.0
weight_iso_values=1000.0
num_knn_values=20

weight_seg=200.0
weight_soft_col_cons=0.01
weight_bg=200.0
weight_rot=4.0
scale_scene_radius=0.05
```

## Cloth
```
weight_im_values=50.0
weight_rigid_values=400.0
weight_iso_values=2000.0
num_knn_values=20

weight_seg=200.0
weight_soft_col_cons=0.01
weight_bg=200.0
weight_rot=4.0
scale_scene_radius=0.05
```