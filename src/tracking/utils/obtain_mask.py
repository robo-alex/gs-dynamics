import argparse
import os
import numpy as np
import json
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from PIL import Image

nodename = os.uname().nodename

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import SamPredictor, sam_model_registry

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        # self.image_list = [f for f in os.listdir(image_dir) if f.startswith("color") and f.endswith(".png")]
        self.image_list = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
        self.image_list.sort(key=lambda n: int(n[:-4].split('_')[-1]))
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_list[idx])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image, _ = self.transform(image, None)
        return image, self.image_list[idx]

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def save_mask_data(output_dir, mask_list, img, box_list, label_list, filename):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    # plt.figure(figsize=(10, 10))
    # # plt.imshow(mask_img.numpy())
    # plt.axis('off')
    # plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    num = int(filename.split('_')[-1].split('.')[0])
    cv2.imwrite(os.path.join(output_dir, f"seg_{num:06}.png"), img)
    print(f"seg_{num:06}.png saved!")

    cv2.imwrite(os.path.join(output_dir, f"mask_{num:06}.png"), mask_img.numpy())
    print(f"mask_{num:06}.png saved!")

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, f"seg_{num:06}.json"), 'w') as f:
        json.dump(json_data, f)

def process_images(images, filenames, model, text_prompt, predictor, box_threshold, text_threshold, device, output_dir):
    # Place all images on the correct device
    images = [img.to(device) for img in images]

    # Run the grounding model on all images
    boxes_list, pred_phrases_list = [], []
    for image in images:
        boxes_filt, pred_phrases = get_grounding_output(
            model, image, text_prompt, box_threshold, text_threshold, device=device
        )
        boxes_list.append(boxes_filt)
        pred_phrases_list.append(pred_phrases)

    # Process results and save output
    for image, boxes_filt, filename in zip(images, boxes_list, filenames):
        # Continue with the SAM predictor
        image_pil, image = load_image(os.path.join(image_path, filename))
        image_cv2 = cv2.imread(os.path.join(image_path, filename))
        image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_cv2)
        
        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_cv2.shape[:2]).to(device)

        masks, _, _ = predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(device),
            multimask_output = False,
        )

        binary_mask = np.squeeze(masks.cpu().numpy()) > 0.5
        img = (binary_mask * 255).astype(np.uint8)
        if img.shape != (image_cv2.shape[0], image_cv2.shape[1]):
            img = img.sum(axis=0)
        save_mask_data(output_dir, masks, img, boxes_filt, pred_phrases, filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="path to raw data")
    parser.add_argument("--text_prompt", type=str, required=True, help="white nylon rope.")
    args = parser.parse_args()

    # cfg
    config_file = "../../third-party/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
    grounded_checkpoint = "../../weights/groundingdino_swinb_cogcoor.pth"
    sam_checkpoint = "../../weights/sam_vit_h_4b8939.pth"
    data_path = args.data_path
    text_prompt = args.text_prompt

    box_threshold = 0.3
    text_threshold = 0.25
    device = "cuda"

    for seq in os.listdir(data_path):
        image_path = os.path.join(data_path, seq)
        output_dir = image_path + "/seg"
        os.makedirs(output_dir, exist_ok=True)

        # Create dataset and dataloader
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        model = load_model(config_file, grounded_checkpoint, device=device)

        predictor = SamPredictor(sam_model_registry["default"](checkpoint=sam_checkpoint).to(device))

        dataset = ImageDataset(image_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)

        # Process all images
        for images, filenames in dataloader:
            process_images(images, filenames, model, text_prompt, predictor, box_threshold, text_threshold, device, output_dir)