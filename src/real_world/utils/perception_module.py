import argparse
import numpy as np
import torch
import open3d as o3d
import cv2
from PIL import Image

from real_world.utils.pcd_utils import visualize_o3d, depth2fgpcd

from segment_anything import SamPredictor, sam_model_registry
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


class PerceptionModule:

    def __init__(self, vis_path, device="cuda:0"):
        self.device = device
        self.vis_path = vis_path
        self.det_model = None
        self.sam_model = None
        self.load_model()
    
    def load_model(self):
        if self.det_model is not None:
            print("Model already loaded")
            return
        device = self.device
        det_model = build_model(SLConfig.fromfile(
            '../third-party/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py'))
        checkpoint = torch.load('../weights/groundingdino_swinb_cogcoor.pth', map_location="cpu")
        det_model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        det_model.eval()
        det_model = det_model.to(device)

        sam = sam_model_registry["default"](checkpoint='../weights/sam_vit_h_4b8939.pth')
        sam_model = SamPredictor(sam)
        sam_model.model = sam_model.model.to(device)

        self.det_model = det_model
        self.sam_model = sam_model
    
    def del_model(self):
        del self.det_model
        torch.cuda.empty_cache()
        del self.sam_model
        torch.cuda.empty_cache()
        self.det_model = None
        self.sam_model = None

    def detect(self, image, captions, box_thresholds):  # captions: list
        image = Image.fromarray(image)
        for i, caption in enumerate(captions):
            caption = caption.lower()
            caption = caption.strip()
            if not caption.endswith("."):
                caption = caption + "."
            captions[i] = caption
        num_captions = len(captions)

        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_tensor, _ = transform(image, None)  # 3, h, w

        image_tensor = image_tensor[None].repeat(num_captions, 1, 1, 1).to(self.device)

        with torch.no_grad():
            outputs = self.det_model(image_tensor, captions=captions)
        logits = outputs["pred_logits"].sigmoid()  # (num_captions, nq, 256)
        boxes = outputs["pred_boxes"]  # (num_captions, nq, 4)

        # filter output
        if isinstance(box_thresholds, list):
            filt_mask = logits.max(dim=2)[0] > torch.tensor(box_thresholds).to(device=self.device, dtype=logits.dtype)[:, None]
        else:
            filt_mask = logits.max(dim=2)[0] > box_thresholds
        labels = torch.ones((*logits.shape[:2], 1)) * torch.arange(logits.shape[0])[:, None, None]  # (num_captions, nq, 1)
        labels = labels.to(device=self.device, dtype=logits.dtype)  # (num_captions, nq, 1)
        logits = logits[filt_mask] # num_filt, 256
        boxes = boxes[filt_mask] # num_filt, 4
        labels = labels[filt_mask].reshape(-1).to(torch.int64) # num_filt,
        scores = logits.max(dim=1)[0] # num_filt,

        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            print(f"Detected {captions[label.item()]} with confidence {round(score.item(), 3)} at location {box}")
        return boxes, scores, labels


    def segment(self, image, boxes, scores, labels, text_prompts):
        # load sam model
        self.sam_model.set_image(image)

        masks, _, _ = self.sam_model.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = self.sam_model.transform.apply_boxes_torch(boxes, image.shape[:2]), # (n_detection, 4)
            multimask_output = False,
        )

        masks = masks[:, 0, :, :] # (n_detection, H, W)
        # text_labels = ['background']
        text_labels = []
        for category in range(len(text_prompts)):
            text_labels = text_labels + [text_prompts[category].rstrip('.')] * (labels == category).sum().item()
        
        # remove masks where IoU are large
        num_masks = masks.shape[0]
        to_remove = []
        for i in range(num_masks):
            for j in range(i+1, num_masks):
                IoU = (masks[i] & masks[j]).sum().item() / (masks[i] | masks[j]).sum().item()
                if IoU > 0.9:
                    if scores[i].item() > scores[j].item():
                        to_remove.append(j)
                    else:
                        to_remove.append(i)
        to_remove = np.unique(to_remove)
        to_keep = np.setdiff1d(np.arange(num_masks), to_remove)
        to_keep = torch.from_numpy(to_keep).to(device=self.device, dtype=torch.int64)
        masks = masks[to_keep]
        text_labels = [text_labels[i] for i in to_keep]
        # text_labels.insert(0, 'background')
        
        aggr_mask = torch.zeros(masks[0].shape).to(device=self.device, dtype=torch.uint8)
        for obj_i in range(masks.shape[0]):
            aggr_mask[masks[obj_i]] = obj_i + 1

        # masks: (n_detection, H, W)
        # aggr_mask: (H, W)
        return (masks, aggr_mask, text_labels), (boxes, scores, labels)


    def get_tabletop_points_env(self, env, depth_threshold=[0, 2], obj_names=[], return_imgs=False):
        obs = env.get_obs(get_color=True, get_depth=True)
        R_list, t_list = env.get_extrinsics()
        intr_list = env.get_intrinsics()
        rgb_list = []
        depth_list = []
        for c in range(env.n_fixed_cameras):
            rgb_list.append(obs[f'color_{c}'][-1][:, :, ::-1])
            depth_list.append(obs[f'depth_{c}'][-1])
        bbox = env.get_bbox()
        return self.get_tabletop_points(rgb_list, depth_list, R_list, t_list, intr_list, bbox, depth_threshold, obj_names, return_imgs)


    def get_tabletop_points(self, rgb_list, depth_list, R_list, t_list, intr_list, bbox, depth_threshold=[0, 2], obj_names=[], return_imgs=False):
        stride = 1
        obj_list = ['table'] + [obj for obj in obj_names]
        text_prompts = [f"{obj}" for obj in obj_list]
        print('text prompts:', text_prompts)

        pcd_all = o3d.geometry.PointCloud()
        mask_objs_list = []
        for i in range(len(rgb_list)):
            intr = intr_list[i]
            R_cam2board = R_list[i]
            t_cam2board = t_list[i]
            depth = depth_list[i].copy().astype(np.float32)
            points = depth2fgpcd(depth, intr)
            points = points.reshape(depth.shape[0], depth.shape[1], 3)
            points = points[::stride, ::stride, :].reshape(-1, 3)
            mask = np.logical_and((depth > depth_threshold[0]), (depth < depth_threshold[1]))  # (H, W)
            mask = mask[::stride, ::stride].reshape(-1)
            img = rgb_list[i].copy()

            # detect and segment
            if len(obj_names) > 0:
                boxes, scores, labels = self.detect(img, text_prompts, box_thresholds=0.3)
                H, W = img.shape[0], img.shape[1]
                boxes = boxes * torch.Tensor([[W, H, W, H]]).to(device=self.device, dtype=boxes.dtype)
                boxes[:, :2] -= boxes[:, 2:] / 2  # xywh to xyxy
                boxes[:, 2:] += boxes[:, :2]  # xywh to xyxy
                segmentation_results, _ = self.segment(img, boxes, scores, labels, text_prompts)
                masks, _, text_labels = segmentation_results
                masks = masks.detach().cpu().numpy()

                mask_table = np.zeros(masks[0].shape, dtype=np.uint8)
                mask_objs = np.zeros(masks[0].shape, dtype=np.uint8)
                for obj_i in range(masks.shape[0]):
                    if text_labels[obj_i] == 'table':
                        mask_table = np.logical_or(mask_table, masks[obj_i])
                for obj_i in range(masks.shape[0]):
                    if text_labels[obj_i] in obj_names:
                        mask_table = np.logical_and(mask_table, ~masks[obj_i])
                        mask_objs = np.logical_or(mask_objs, masks[obj_i])
                mask_obj_and_background = 1 - mask_table
                cv2.imwrite(f'{self.vis_path}/{i}_rgb.png', img[:, :, ::-1].copy())
                cv2.imwrite(f'{self.vis_path}/{i}_mask_table.png', (mask_table * 255).astype(np.uint8))
                cv2.imwrite(f'{self.vis_path}/{i}_mask_objs.png', (mask_objs * 255).astype(np.uint8))
                cv2.imwrite(f'{self.vis_path}/{i}_mask_obj_and_background.png', (mask_obj_and_background * 255).astype(np.uint8))
                if return_imgs:
                    mask_objs_list.append(mask_objs)

                mask_obj_and_background = mask_obj_and_background.astype(bool)
                mask_obj_and_background = mask_obj_and_background[::stride, ::stride].reshape(-1)
                mask = np.logical_and(mask, mask_obj_and_background)

            points = points[mask].reshape(-1, 3)
            points = R_cam2board @ points.T + t_cam2board[:, None]
            points = points.T  # (N, 3)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            colors = img[::stride, ::stride, :].reshape(-1, 3).astype(np.float64)
            colors = colors[mask].reshape(-1, 3)
            pcd.colors = o3d.utility.Vector3dVector(colors / 255)
            # visualize_o3d([pcd])
            pcd_all += pcd
        pcd = pcd_all
        pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(min_bound=bbox[:, 0], max_bound=bbox[:, 1]))
        pcd = pcd.voxel_down_sample(voxel_size=0.005)
        outliers = None
        new_outlier = None
        rm_iter = 0
        while new_outlier is None or len(new_outlier.points) > 0:
            _, inlier_idx = pcd.remove_statistical_outlier(
                nb_neighbors = 25, std_ratio = 1.5 + rm_iter * 0.5
            )
            new_pcd = pcd.select_by_index(inlier_idx)
            new_outlier = pcd.select_by_index(inlier_idx, invert=True)
            if outliers is None:
                outliers = new_outlier
            else:
                outliers += new_outlier
            pcd = new_pcd
            rm_iter += 1
        # visualize_o3d([pcd])
        if return_imgs:
            return pcd, rgb_list, mask_objs_list
        return pcd
