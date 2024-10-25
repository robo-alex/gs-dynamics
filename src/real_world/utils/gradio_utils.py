import gradio as gr
import numpy as np
from PIL import Image, ImageDraw



def get_valid_mask(mask: np.ndarray):
    """Convert mask from gr.Image(0 to 255, RGBA) to binary mask.
    """
    if mask.ndim == 3:
        mask_pil = Image.fromarray(mask).convert('L')
        mask = np.array(mask_pil)
    if mask.max() == 255:
        mask = mask / 255
    return mask


def draw_points_on_image(image, points, intr, extr, z, radius_scale=0.006, intr_orig=None, extr_orig=None):
    same_cam = (intr is None and extr is None)
    if intr_orig is None:
        assert extr_orig is None
        intr_orig = intr
        extr_orig = extr
    overlay_rgba = Image.new("RGBA", image.size, 0)
    overlay_draw = ImageDraw.Draw(overlay_rgba)
    for point_key, point in points.items():

        t_color = (255, 100, 100)
        o_color = (255, 50, 50)

        rad_draw = int(image.size[0] * radius_scale) + 2

        p_start = point["start"]
        p_target = point["target"]

        if p_start is not None and p_target is not None:
            if same_cam:
                p_draw = int(p_start[0]), int(p_start[1])
                t_draw = int(p_target[0]), int(p_target[1])

            # 2d
            # pt = (p_target[0] - p_start[0], p_target[1] - p_start[1])
            # pt_norm = np.linalg.norm(pt)
            # pt_unit = (pt[0] / pt_norm, pt[1] / pt_norm)
            # pt_tang = (pt_unit[1], -pt_unit[0])
            # tt1 = (t_draw[0] + pt_tang[0] * 0.1 * pt_norm - pt_unit[0] * 0.1 * pt_norm,
            #        t_draw[1] + pt_tang[1] * 0.1 * pt_norm - pt_unit[1] * 0.1 * pt_norm)
            # tt2 = (t_draw[0] - pt_tang[0] * 0.1 * pt_norm - pt_unit[0] * 0.1 * pt_norm,
            #        t_draw[1] - pt_tang[1] * 0.1 * pt_norm - pt_unit[1] * 0.1 * pt_norm)

            # 3d
            p_start_3d = np.array([p_start[0], p_start[1], 1])
            p_target_3d = np.array([p_target[0], p_target[1], 1])
            p_start_3d = np.dot(np.linalg.inv(intr_orig), p_start_3d)
            p_target_3d = np.dot(np.linalg.inv(intr_orig), p_target_3d)
            p_start_3d = np.dot(np.linalg.inv(extr_orig), np.concatenate([p_start_3d, [1]]))
            p_target_3d = np.dot(np.linalg.inv(extr_orig), np.concatenate([p_target_3d, [1]]))
            camera_t = np.linalg.inv(extr_orig)[:3, 3]
            p_start_3d = (p_start_3d[:3] - camera_t) * (z - camera_t[2]) / (p_start_3d[2] - camera_t[2]) + camera_t
            p_target_3d = (p_target_3d[:3] - camera_t) * (z - camera_t[2]) / (p_target_3d[2] - camera_t[2]) + camera_t
            pt_3d = p_target_3d - p_start_3d
            pt_3d_norm = np.linalg.norm(pt_3d)
            pt_3d_unit = pt_3d / pt_3d_norm
            pt_3d_tang = np.array([pt_3d_unit[1], -pt_3d_unit[0], 0])
            tt1_3d = p_target_3d + pt_3d_tang * 0.02 - pt_3d_unit * 0.02
            tt2_3d = p_target_3d - pt_3d_tang * 0.02 - pt_3d_unit * 0.02
            tt1_3d = np.dot(extr, np.concatenate([tt1_3d, [1]]))[:3]
            tt2_3d = np.dot(extr, np.concatenate([tt2_3d, [1]]))[:3]
            tt1_3d = np.dot(intr, tt1_3d)
            tt2_3d = np.dot(intr, tt2_3d)
            tt1_3d = (tt1_3d[:2] / tt1_3d[2]).astype(int)
            tt2_3d = (tt2_3d[:2] / tt2_3d[2]).astype(int)
            tt1 = (tt1_3d[0], tt1_3d[1])
            tt2 = (tt2_3d[0], tt2_3d[1])

            tt1_draw = int(tt1[0]), int(tt1[1])
            tt2_draw = int(tt2[0]), int(tt2[1])

            if not same_cam:
                p_proj = np.dot(intr, np.dot(extr, np.concatenate([p_start_3d, [1]]))[:3])
                p_draw = (p_proj[:2] / p_proj[2]).astype(int)
                t_proj = np.dot(intr, np.dot(extr, np.concatenate([p_target_3d, [1]]))[:3])
                t_draw = (t_proj[:2] / t_proj[2]).astype(int)

            overlay_draw.line(
                (p_draw[0], p_draw[1], t_draw[0], t_draw[1]),
                fill=o_color,
                width=4,
            )
            
            overlay_draw.line(
                (t_draw[0], t_draw[1], tt1_draw[0], tt1_draw[1]),
                fill=o_color,
                width=4,
            )

            overlay_draw.line(
                (t_draw[0], t_draw[1], tt2_draw[0], tt2_draw[1]),
                fill=o_color,
                width=4,
            )

        if p_start is not None:
            if same_cam:
                p_draw = int(p_start[0]), int(p_start[1])
            else:
                # 3d
                p_start_3d = np.array([p_start[0], p_start[1], 1])
                p_start_3d = np.dot(np.linalg.inv(intr_orig), p_start_3d)
                p_start_3d = np.dot(np.linalg.inv(extr_orig), np.concatenate([p_start_3d, [1]]))
                camera_t = np.linalg.inv(extr_orig)[:3, 3]
                p_start_3d = (p_start_3d[:3] - camera_t) * (z - camera_t[2]) / (p_start_3d[2] - camera_t[2]) + camera_t
                
                p_proj = np.dot(intr, np.dot(extr, np.concatenate([p_start_3d, [1]]))[:3])
                p_draw = (p_proj[:2] / p_proj[2]).astype(int)
            
            overlay_draw.ellipse(
                (
                    p_draw[0] - rad_draw,
                    p_draw[1] - rad_draw,
                    p_draw[0] + rad_draw,
                    p_draw[1] + rad_draw,
                ),
                fill=t_color,
                outline=o_color,
                width=2,
            )

        if p_target is not None:
            assert p_start is not None

    return Image.alpha_composite(image.convert("RGBA"),
                                 overlay_rgba).convert("RGB")


def draw_raw_points_on_image(image,
                             points,
                             # curr_point=None,
                             # highlight_all=True,
                             radius_scale=0.002):
    overlay_rgba = Image.new("RGBA", image.size, 0)
    overlay_draw = ImageDraw.Draw(overlay_rgba)
    for p in range(points.shape[0]):
        point = points[p]
        # if ((curr_point is not None and curr_point == point_key)
        #         or highlight_all):
        #     p_color = (255, 0, 0)
        t_color = (150, 150, 255)
        o_color = (50, 50, 255)

        # else:
        #     p_color = (255, 0, 0, 35)
        #     t_color = (0, 0, 255, 35)

        rad_draw = int(image.size[0] * radius_scale)

        # p_start = point.get("start_temp", point["start"])
        # p_target = point["target"]

        # if p_start is not None and p_target is not None:
        #     p_draw = int(p_start[0]), int(p_start[1])
        #     t_draw = int(p_target[0]), int(p_target[1])

        #     overlay_draw.line(
        #         (p_draw[0], p_draw[1], t_draw[0], t_draw[1]),
        #         fill=(255, 255, 0),
        #         width=2,
        #     )

        # if p_start is not None:
        #     p_draw = int(p_start[0]), int(p_start[1])
        #     overlay_draw.ellipse(
        #         (
        #             p_draw[0] - rad_draw,
        #             p_draw[1] - rad_draw,
        #             p_draw[0] + rad_draw,
        #             p_draw[1] + rad_draw,
        #         ),
        #         fill=p_color,
        #     )

        #     if curr_point is not None and curr_point == point_key:
        #         # overlay_draw.text(p_draw, "p", font=font, align="center", fill=(0, 0, 0))
        #         overlay_draw.text(p_draw, "p", align="center", fill=(0, 0, 0))

        # if p_target is not None:
        t_draw = int(point[0]), int(point[1])
        overlay_draw.ellipse(
            (
                t_draw[0] - rad_draw,
                t_draw[1] - rad_draw,
                t_draw[0] + rad_draw,
                t_draw[1] + rad_draw,
            ),
            fill=t_color,
            outline=o_color,
        )

        # if curr_point is not None and curr_point == point_key:
        #     # overlay_draw.text(t_draw, "t", font=font, align="center", fill=(0, 0, 0))
        #     overlay_draw.text(t_draw, "t", align="center", fill=(0, 0, 0))

    return Image.alpha_composite(image.convert("RGBA"),
                                 overlay_rgba).convert("RGB")


def draw_mask_on_image(image, mask):
    im_mask = np.uint8(mask * 255)
    im_mask_rgba = np.concatenate(
        (
            np.tile(im_mask[..., None], [1, 1, 3]),
            45 * np.ones(
                (im_mask.shape[0], im_mask.shape[1], 1), dtype=np.uint8),
        ),
        axis=-1,
    )
    im_mask_rgba = Image.fromarray(im_mask_rgba).convert("RGBA")

    return Image.alpha_composite(image.convert("RGBA"),
                                 im_mask_rgba).convert("RGB")


def on_change_single_global_state(keys,
                                  value,
                                  global_state,
                                  map_transform=None):
    if map_transform is not None:
        value = map_transform(value)

    curr_state = global_state
    if isinstance(keys, str):
        last_key = keys

    else:
        for k in keys[:-1]:
            curr_state = curr_state[k]

        last_key = keys[-1]

    curr_state[last_key] = value
    return global_state


def get_latest_points_pair(points_dict):
    if not points_dict:
        return None
    point_idx = list(points_dict.keys())
    latest_point_idx = max(point_idx)
    return latest_point_idx