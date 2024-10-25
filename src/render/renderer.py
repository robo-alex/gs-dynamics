import torch

from diff_gaussian_rasterization import GaussianRasterizer
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera

class Renderer:

    def __init__(self, device):
        self.near = 0.01
        self.far = 100.0
        self.remove_background = False

        self.w = 1280
        self.h = 720

        self.device = device
    
    @torch.no_grad
    def render(self, w2c, k, timestep_data, bg=[0.7, 0.7, 0.7]):
        timestep_data = {k: v.to(self.device) for k, v in timestep_data.items()}
        cam = self.setup_camera(k, w2c, bg=bg)
        im, _, depth, = GaussianRasterizer(raster_settings=cam)(**timestep_data)
        return im, depth

    def setup_camera(self, k, w2c, bg):
        w, h = self.w, self.h
        near, far = self.near, self.far
        fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
        w2c = torch.tensor(w2c).cuda().float()
        cam_center = torch.inverse(w2c)[:3, 3]
        w2c = w2c.unsqueeze(0).transpose(1, 2)
        opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                    [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                    [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                    [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
        full_proj = w2c.bmm(opengl_proj)
        cam = Camera(
            image_height=h,
            image_width=w,
            tanfovx=w / (2 * fx),
            tanfovy=h / (2 * fy),
            bg=torch.tensor(bg, dtype=torch.float32, device=self.device),
            scale_modifier=1.0,
            viewmatrix=w2c,
            projmatrix=full_proj,
            sh_degree=0,
            campos=cam_center,
            prefiltered=False
        )
        return cam
