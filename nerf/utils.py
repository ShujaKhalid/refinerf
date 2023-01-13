from ntpath import join
import os
import glob
import tqdm
import math
import random
import warnings
import tensorboardX

import numpy as np
import pandas as pd

import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
import lpips

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR

import trimesh
import mcubes
from rich.console import Console
from torch_ema import ExponentialMovingAverage

from packaging import version as pver

# from dnerf.network_fxfy import LearnFocal
# from dnerf.network_pose import LearnPose


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, masks, N=-1, error_map=None, dynamic_iter=-1, dynamic_iters=-1):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    # FIXME
    if (N > 0):
        MODELS = 1  # ["static", "dynamic"]
    else:
        MODELS = 2  # ["static", "dynamic"]

    i, j = custom_meshgrid(torch.linspace(
        0, (W)-1, W, device=device), torch.linspace(0, (H)-1, H, device=device))

    # print("poses: {}".format(poses))
    # print("device: {}".format(device))
    # print("i (before): {}".format(i))
    # print("j.shape: {}".format(j.shape))
    i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
    # print("i (after): {}".format(i))
    j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H*W)

        if error_map is None:
            e = 0  # buffer
            # N = 2*N  # FIXME
            if (masks != None):
                # mask = masks[e:masks.shape[0]-e, -1].to(device)
                # mask = torch.amax(masks, -1).to(device)
                mask = masks.mean(1).to(device)

                thresh = 0.0  # training threshold
                coords_s = torch.where(mask == thresh)[0]
                coords_d = torch.where(mask > thresh)[0]  # For training
                coords_s_mask = torch.where(mask == 0.0)[0]
                coords_d_mask = torch.where(mask > 0.0)[0]  # For inference
                # print("\ncoords_s: {}".format(coords_s))
                # print("coords_d: {}".format(coords_d))
                # print("mask.unique: {}".format(np.unique(mask.cpu().numpy())))

                # inds = torch.cat([coords_s, coords_d], 0)
                cond = np.array([key for key in dynamic_iters if dynamic_iter >= dynamic_iters[key][0] and dynamic_iter <
                                 dynamic_iters[key][1]])
                if ('d1' in cond or 'd2' in cond or 'd3' in cond or 'd4' in cond):
                    # print("\n\n=======================================")
                    # print(
                    #     "DYNAMIC MODEL ACTIVATED!!! - (get_rays) - iter: {}".format(dynamic_iter))
                    # print("=======================================\n\n")
                    # if (coords_d.shape[-1]-1 >= N):
                    inds_s = torch.randint(
                        0, coords_s.shape[-1]-1, size=[0], device=device)  # may duplicate
                    inds_d = torch.randint(
                        0, coords_d.shape[-1]-1, size=[int(N)], device=device)  # may duplicate
                    # else:
                    #     inds_s = torch.randint(
                    #         0, 10, size=[5], device=device)  # may duplicate
                    #     inds_d = torch.randint(
                    #         0, 10, size=[5], device=device)  # may duplicate
                    #     coords_d = coords_s

                    coords_s = coords_s[inds_s]
                    coords_d = coords_d[inds_d]
                    inds = torch.cat([coords_d], 0)
                    results['inds_s'] = coords_s_mask
                    results['inds_d'] = coords_d_mask
                elif ('b1' in cond or 'b2' in cond or 'b3' in cond or 'b4' in cond):
                    # print("\n\n=======================================")
                    # print(
                    #     "COMBINED MODEL ACTIVATED!!! - (get_rays) - iter: {}".format(dynamic_iter))
                    # print("=======================================\n\n")
                    inds_s = torch.randint(
                        0, coords_s.shape[-1]-1, size=[int(N//2)], device=device)  # may duplicate
                    inds_d = torch.randint(
                        0, coords_d.shape[-1]-1, size=[int(N//2)], device=device)  # may duplicate

                    coords_s = coords_s[inds_s]
                    coords_d = coords_d[inds_d]
                    inds = torch.cat([coords_s, coords_d], 0)

                    results['inds_s'] = coords_s_mask
                    results['inds_d'] = coords_d_mask

                    results["both"] = True
                else:
                    # print("\n\n=======================================")
                    # print(
                    #     "STATIC MODEL ACTIVATED!!! - (get_rays) - iter: {}".format(dynamic_iter))
                    # print("=======================================\n\n")
                    inds_s = torch.randint(
                        0, coords_s.shape[-1]-1, size=[int(N)], device=device)  # may duplicate
                    inds_d = torch.randint(
                        0, 1, size=[0], device=device)  # may duplicate

                    coords_s = coords_s[inds_s]
                    coords_d = coords_d[inds_d]
                    inds = torch.cat([coords_s], 0)
                    results['inds_s'] = coords_s_mask
                    results['inds_d'] = coords_d_mask

                    # print("\ncoords_s: {}".format(coords_s))
                    # print("coords_d: {}".format(coords_d))

            else:
                # sk_debug - Random from anaywhere on grid
                # For dnerf datasets - not sure if required
                inds = torch.randint(
                    0, H*W, size=[N], device=device)  # may duplicate
                # results['inds_s'] = torch.Tensor([]).cuda()
                # results['inds_d'] = inds
                results['inds_s'] = inds
                results['inds_d'] = torch.Tensor([]).cuda()

            inds = inds.expand([B, inds.shape[0]])
        else:

            # weighted sample on a low-reso grid
            # [B, N], but in [0, 128*128)
            inds_coarse = torch.multinomial(
                error_map.to(device), N, replacement=False)

            # map to the original resolution with random perturb.
            # `//` will throw a warning in torch 1.10... anyway.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device)
                      * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device)
                      * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            # need this when updating error_map
            results['inds_coarse'] = inds_coarse

        # We're only using a very small set of points from
        # our meshgrid
        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        if (masks != None):
            # mask = masks[e:masks.shape[0]-e, -1].to(device)
            mask = masks.mean(1).to(device)

            coords_s = torch.where(mask == 0.0)[0]
            coords_d = torch.where(mask > 0.0)[0]

            # print("\n\ncoords_s.shape: {}".format(coords_s.shape))
            # print("coords_d.shape: {}".format(coords_d.shape))
            # print("mask.shape: {}".format(mask.shape))
            # print("masks.shape: {}".format(masks.shape))

            # no segmentation assistance
            # coords_s = torch.randint(
            #     0, coords_s.shape[-1]-1, size=[int(len(coords_s)+len(coords_d))], device=device)  # may duplicate
            # coords_d = torch.randint(
            #     0, coords_d.shape[-1]-1, size=[int(len(coords_s)+len(coords_d))], device=device)  # may duplicate

            # segmentation assisted
            inds = torch.cat([coords_s, coords_d], 0)

            results['inds_s'] = coords_s
            results['inds_d'] = coords_d
            # inds = torch.cat([coords_d], 0)

        else:
            # sk_debug - Random from anywhere on grid
            coords_s = torch.randint(
                0, H*W-1, size=[0], device=device)  # may duplicate
            coords_d = torch.randint(
                0, H*W-1, size=[H*W], device=device)  # may duplicate

            inds = torch.cat([coords_s, coords_d], 0)

            results['inds_s'] = coords_s
            results['inds_d'] = coords_d

            inds = torch.arange(
                H*W*MODELS, device=device).expand([B, H*W*MODELS])
            results['inds'] = inds

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)  # (B, N, 3)
    rays_o = poses[..., :3, 3]  # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d)  # [B, N, 3]

    # print("\nxs.requires_grad: {}".format(xs.requires_grad))
    # print("ys.requires_grad: {}".format(ys.requires_grad))
    # print("zs.requires_grad: {}".format(zs.requires_grad))
    # print("directions.requires_grad: {}".format(directions.requires_grad))
    # print("rays_d.requires_grad: {}".format(rays_d.requires_grad))
    # print("rays_o.requires_grad: {}".format(rays_o.requires_grad))
    # print("poses.requires_grad: {}".format(poses.requires_grad))
    # print("intrinsics.requires_grad: {}".format(intrinsics.requires_grad))
    # print("fx.requires_grad: {}".format(fx.requires_grad))
    # print("fy.requires_grad: {}".format(fy.requires_grad))
    # print("rays_o: {}".format(rays_o))
    # print("rays_d: {}".format(rays_d))
    # print("\nrays_o.shape: {}".format(rays_o.shape))
    # print("\nrays_d.shape: {}".format(rays_d.shape))

    # inds = torch.arange(4096, device=device).expand([B, 4096])
    # inds_s = torch.arange(4096, device=device).expand([B, 4096])
    # inds_d = torch.arange(4096, device=device).expand([B, 4096])
    results['inds'] = inds
    # results['inds_s'] = inds_s
    # results['inds_d'] = inds_d
    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return results


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def torch_vis_2d(x, renormalize=False):
    # x: [3, H, W] or [1, H, W] or [H, W]
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    if isinstance(x, torch.Tensor):
        if len(x.shape) == 3:
            x = x.permute(1, 2, 0).squeeze()
        x = x.detach().cpu().numpy()

    print(f'[torch_vis_2d] {x.shape}, {x.dtype}, {x.min()} ~ {x.max()}')

    x = x.astype(np.float32)

    # renormalize
    if renormalize:
        x = (x - x.min(axis=0, keepdims=True)) / \
            (x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8)

    plt.imshow(x)
    plt.show()


def extract_fields(bound_min, bound_max, resolution, query_func, S=128):

    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(S)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(S)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(S)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat(
                        [xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)  # [S, 3]
                    val = query_func(pts).reshape(len(xs), len(ys), len(
                        zs)).detach().cpu().numpy()  # [S, 1] --> [x, y, z]
                    u[xi * S: xi * S + len(xs), yi * S: yi * S +
                      len(ys), zi * S: zi * S + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    # print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)

    # print(u.shape, u.max(), u.min(), np.percentile(u, 50))

    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * \
        (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


class PSNRMeter:
    def __init__(self):
        self.V = 0
        self.SSIM = 0
        self.LPIPS = 0
        self.N = 0
        self.lpips_loss = lpips.LPIPS(net='alex')

    def im2tensor(self, img):
        return torch.Tensor(img.transpose(2, 0, 1))[None, ...]
        # return torch.Tensor(img.transpose(2, 0, 1) / 127.5 - 1.0)[None, ...]
        # return torch.Tensor(img / 127.5 - 1.0)[None, ...]

    def clear(self):
        self.V = 0
        self.SSIM = 0
        self.LPIPS = 0
        self.N = 0

    # def prepare_inputs(self, *inputs):
    #     outputs = []
    #     for i, inp in enumerate(inputs):
    #         if torch.is_tensor(inp):
    #             inp = inp.detach().cpu().numpy()
    #         outputs.append(inp)

    #     return outputs

    def update(self, preds, truths):
        # [B, N, 3] or [B, H, W, 3], range[0, 1]
        CALC_FLAG = True
        if (CALC_FLAG):
            time_samples = 1
            for time in range(time_samples):
                pred_f16 = np.squeeze(
                    preds[time].detach().cpu().numpy())
                truth_f16 = np.squeeze(
                    truths[time].detach().cpu().numpy())
                pred_int = (pred_f16*255.).astype(np.uint8)
                truth_int = (truth_f16*255.).astype(np.uint8)
                ssim = structural_similarity(
                    truth_f16, pred_f16, channel_axis=2)
                lpips = self.lpips_loss.forward(
                    self.im2tensor(truth_int), self.im2tensor(pred_int)).item()
                psnr = cv2.PSNR(truth_int, pred_int)
                #psnr = -10 * np.log10(np.mean((pred_int - truth_int) ** 2))
                # print("pred_f16.mean: {}".format(pred_f16.mean()))
                # print("truth_f16.mean: {}".format(truth_f16.mean()))

        else:
            ssim = 0
            lpips = 0

            # simplified since max_pixel_value is 1 here.
            psnr = -10 * np.log10(np.mean((preds - truths) ** 2))

        self.V += psnr
        self.SSIM += ssim
        self.LPIPS += lpips
        self.N += 1

    def measure_psnr(self):
        return self.V / self.N

    def measure_ssim(self):
        return self.SSIM / self.N
        # return 0

    def measure_lpips(self):
        return self.LPIPS / self.N
        # return 0

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"),
                          self.measure_psnr(), global_step)
        writer.add_scalar(os.path.join(prefix, "SSIM"),
                          self.measure_ssim(), global_step)
        writer.add_scalar(os.path.join(prefix, "LPIPS"),
                          self.measure_lpips(), global_step)

    def report(self):

        return f'PSNR = {self.measure_psnr():.6f} - SSIM = {self.measure_ssim():.6f} - LPIPS = {self.measure_lpips():.6f}'


class Trainer(object):
    def __init__(self,
                 name,  # name of this experiment
                 opt,  # extra conf
                 model,  # network
                 model_fxfy,  # network
                 model_pose,  # network
                 #  model_camera=None,  # camera_network
                 criterion=None,  # loss function, if None, assume inline implementation in train_step
                 optimizer_model=None,  # optimizer
                 optimizer_fxfy=None,  # optimizer
                 optimizer_pose=None,  # optimizer
                 #  optimizer_cam_model=None,  # optimizer
                 ema_decay=None,  # if use EMA, set the decay
                 lr_scheduler=None,  # scheduler
                 # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 metrics=[],
                 local_rank=0,  # which GPU am I
                 world_size=1,  # total num of GPUs
                 # device to use, usually setting to None is OK. (auto choose device)
                 device=None,
                 mute=False,  # whether to mute all print
                 fp16=False,  # amp optimize level
                 eval_interval=1,  # eval once every $ epoch
                 max_keep_ckpt=2,  # max num of saved ckpts in disk
                 workspace='workspace',  # workspace to save logs & ckpts
                 best_mode='min',  # the smaller/larger result, the better
                 use_loss_as_metric=True,  # use loss as the first metric
                 report_metric_at_train=False,  # also report metrics at training
                 use_checkpoint="latest",  # which ckpt to use at init time
                 use_tensorboardX=True,  # whether to use tensorboard for logging
                 # whether to call scheduler.step() after every train step
                 scheduler_update_every_step=False,
                 ):

        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.tensorboard_folder = opt.tensorboard_folder
        self.pred_intrinsics = opt.pred_intrinsics
        self.pred_extrinsics = opt.pred_extrinsics
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(
            f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        self.optimizer_func = optimizer_model
        # self.optimizer_cam_func = optimizer_cam_model
        self.scheduler_func = lr_scheduler

        # model_camera.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank])
        self.model = model
        self.model_fxfy = model_fxfy
        self.model_pose = model_pose

        # self.model_fxfy = LearnFocal(H=1080, W=1920).cuda()  # FIXME
        # self.model_pose = LearnPose(num_cams=24).cuda()  # FIXME
        # self.model_camera = model_camera

        self.model.to(self.device)
        self.model_fxfy.to(self.device)
        self.model_pose.to(self.device)

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        if optimizer_model is None:
            self.optimizer_model = optim.Adam(self.model.parameters(),
                                              lr=0.001, weight_decay=5e-4)  # naive adam
        else:
            self.opt_state = "static"
            self.optimizer_model = optimizer_model(
                self.model, self.opt_state)

        self.optimizer_fxfy = optim.Adam(self.model_fxfy.parameters(),
                                         lr=0.000025, weight_decay=5e-4)  # naive adam
        self.optimizer_pose = optim.Adam(self.model_pose.parameters(),
                                         lr=0.000025, weight_decay=5e-4)  # naive adam

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer_model, lr_lambda=lambda epoch: 1)  # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer_model)

        if ema_decay is not None:
            self.opt_state = "static"
            self.ema = ExponentialMovingAverage(
                self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        self.scheduler_focal = MultiStepLR(
            self.optimizer_fxfy, milestones=list(range(0, 10000, 100)), gamma=0.9)
        self.scheduler_pose = MultiStepLR(
            self.optimizer_pose, milestones=list(range(0, 10000, 100)), gamma=0.9)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics[0], or valid_loss
            "checkpoints": [],  # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)

        self.log(
            f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(
            f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(
                        f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else:  # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

        # clip loss prepare
        if opt.rand_pose >= 0:  # =0 means only using CLIP loss, >0 means a hybrid mode.
            from nerf.clip_utils import CLIPLoss
            self.clip_loss = CLIPLoss(self.device)
            # only support one text prompt now...
            self.clip_loss.prepare_text([self.opt.clip_text])

    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute:
                # print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr:
                print(*args, file=self.log_ptr)
                self.log_ptr.flush()  # write immediately to file

    # ------------------------------

    def train_step(self, data):

        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]

        # if there is no gt image, we train with CLIP loss.
        if 'images' not in data:

            B, N = rays_o.shape[:2]
            H, W = data['H'], data['W']

            # currently fix white bg, MUST force all rays!
            outputs = self.model.render(
                rays_o, rays_d, staged=False, bg_color=None, perturb=True, force_all_rays=True, **vars(self.opt))
            pred_rgb = outputs['image'].reshape(
                B, H, W, 3).permute(0, 3, 1, 2).contiguous()

            # [debug] uncomment to plot the images used in train_step
            # torch_vis_2d(pred_rgb[0])

            loss = self.clip_loss(pred_rgb)

            return pred_rgb, None, loss

        images = data['images']  # [B, N, 3/4]

        B, N, C = images.shape

        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

        if C == 3 or self.model.bg_radius > 0:
            bg_color = 1
        # train with random background color if not using a bg model and has alpha channel.
        else:
            # bg_color = torch.ones(3, device=self.device) # [3], fixed white background
            # bg_color = torch.rand(3, device=self.device) # [3], frame-wise random.
            # [N, 3], pixel-wise random.
            bg_color = torch.rand_like(images[..., :3])

        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + \
                bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images

        outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=bg_color,
                                    perturb=True, force_all_rays=False, **vars(self.opt))

        pred_rgb = outputs['image']

        # [B, N, 3] --> [B, N]
        loss = self.criterion(pred_rgb, gt_rgb).mean(-1)

        # special case for CCNeRF's rank-residual training
        if len(loss.shape) == 3:  # [K, B, N]
            loss = loss.mean(0)

        # update error_map
        if self.error_map is not None:
            index = data['index']  # [B]
            inds = data['inds_coarse']  # [B, N]

            # take out, this is an advanced indexing and the copy is unavoidable.
            error_map = self.error_map[index]  # [B, H * W]

            # [debug] uncomment to save and visualize error map
            # if self.global_step % 1001 == 0:
            #     tmp = error_map[0].view(128, 128).cpu().numpy()
            #     print(f'[write error map] {tmp.shape} {tmp.min()} ~ {tmp.max()}')
            #     tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
            #     cv2.imwrite(os.path.join(self.workspace, f'{self.global_step}.jpg'), (tmp * 255).astype(np.uint8))

            # [B, N], already in [0, 1]
            error = loss.detach().to(error_map.device)

            # ema update
            ema_error = 0.1 * error_map.gather(1, inds) + 0.9 * error
            error_map.scatter_(1, inds, ema_error)

            # put back
            self.error_map[index] = error_map

        loss = loss.mean()

        # extra loss
        # pred_weights_sum = outputs['weights_sum'] + 1e-8
        # loss_ws = - 1e-1 * pred_weights_sum * torch.log(pred_weights_sum) # entropy to encourage weights_sum to be 0 or 1.
        # loss = loss + loss_ws.mean()

        return pred_rgb, gt_rgb, loss

    def eval_step(self, data):

        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        images = data['images']  # [B, H, W, 3/4]
        B, H, W, C = images.shape

        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])

        # eval with fixed background color
        bg_color = 1
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + \
                bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images

        outputs = self.model.render(
            rays_o, rays_d, staged=True, bg_color=bg_color, perturb=False, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)

        loss = self.criterion(pred_rgb, gt_rgb).mean()

        return pred_rgb, pred_depth, gt_rgb, loss

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, bg_color=None, perturb=False):

        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        H, W = data['H'], data['W']

        if bg_color is not None:
            bg_color = bg_color.to(self.device)

        outputs = self.model.render(
            rays_o, rays_d, staged=True, bg_color=bg_color, perturb=perturb, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(-1, H, W, 3)
        pred_depth = outputs['depth'].reshape(-1, H, W)

        return pred_rgb, pred_depth

    def save_mesh(self, save_path=None, resolution=256, threshold=10):

        if save_path is None:
            save_path = os.path.join(
                self.workspace, 'meshes', f'{self.name}_{self.epoch}.ply')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        def query_func(pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    sigma = self.model.density(pts.to(self.device))['sigma']
            return sigma

        vertices, triangles = extract_geometry(
            self.model.aabb_infer[:3], self.model.aabb_infer[3:], resolution=resolution, threshold=threshold, query_func=query_func)

        # important, process=True leads to seg fault...
        mesh = trimesh.Trimesh(vertices, triangles, process=False)
        mesh.export(save_path)

        self.log(f"==> Finished saving mesh.")

    # ------------------------------

    def train(self, train_loader, valid_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            # self.writer = tensorboardX.SummaryWriter(
            #     os.path.join(self.workspace, "run", self.name, str(int(time.time()))))
            self.writer = tensorboardX.SummaryWriter(
                os.path.join(self.workspace, "run", self.name, self.tensorboard_folder))

        # mark untrained region (i.e., not covered by any camera from the training dataset)
        if self.model.cuda_ray:
            self.model.mark_untrained_grid(
                train_loader._data.poses, train_loader._data.intrinsics)

        # get a ref to error_map
        self.error_map = train_loader._data.error_map

        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)

        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                         bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()
        with torch.no_grad():

            for i, data in enumerate(loader):

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth = self.test_step(data)

                path = os.path.join(save_path, f'{name}_{i:04d}.png')
                path_depth = os.path.join(
                    save_path, f'{name}_{i:04d}_depth.png')

                # self.log(f"[INFO] saving test image to {path}")

                if self.opt.color_space == 'linear':
                    preds = linear_to_srgb(preds)

                pred = preds[0].detach().cpu().numpy()
                pred_depth = preds_depth[0].detach().cpu().numpy()

                cv2.imwrite(path, cv2.cvtColor(
                    (pred * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                cv2.imwrite(path_depth, (pred_depth * 255).astype(np.uint8))

                pbar.update(loader.batch_size)

        self.log(f"==> Finished Test.")

    # [GUI] just train for 16 steps, without any other overhead that may slow down rendering.
    def train_gui(self, train_loader, step=16):

        self.model.train()

        total_loss = torch.tensor([0], dtype=torch.float32, device=self.device)

        loader = iter(train_loader)

        # mark untrained grid
        if self.global_step == 0:
            self.model.mark_untrained_grid(
                train_loader._data.poses, train_loader._data.intrinsics)

        for _ in range(step):

            # mimic an infinite loop dataloader (in case the total dataset is smaller than step)
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(train_loader)
                data = next(loader)

            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()

            self.global_step += 1

            self.optimizer_model.zero_grad()
            self.optimizer_fxfy.zero_grad()
            self.optimizer_pose.zero_grad()
            # self.optimizer_cam_model.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss = self.train_step(data)

            self.scaler.scale(loss).backward()

            self.scaler.step(self.optimizer_model)
            # self.scaler.step(self.optimizer_fxfy)
            # self.scaler.step(self.optimizer_pose)
            # self.scaler.step(self.optimizer_cam_model)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            total_loss += loss.detach()

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss.item() / step

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        outputs = {
            'loss': average_loss,
            'lr': self.optimizer_model.param_groups[0]['lr'],
        }

        return outputs

    # [GUI] test on a single image

    def test_gui(self, pose, intrinsics, W, H, bg_color=None, spp=1, downscale=1):

        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)

        rays = get_rays(pose, intrinsics, rH, rW, -1)

        data = {
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'H': rH,
            'W': rW,
        }

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # here spp is used as perturb random seed!
                preds, preds_depth = self.test_step(
                    data, bg_color=bg_color, perturb=spp)

        if self.ema is not None:
            self.ema.restore()

        # interpolation to the original resolution
        if downscale != 1:
            # TODO: have to permute twice with torch...
            preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(
                H, W), mode='nearest').permute(0, 2, 3, 1).contiguous()
            preds_depth = F.interpolate(preds_depth.unsqueeze(
                1), size=(H, W), mode='nearest').squeeze(1)

        if self.opt.color_space == 'linear':
            preds = linear_to_srgb(preds)

        pred = preds[0].detach().cpu().numpy()
        pred_depth = preds_depth[0].detach().cpu().numpy()

        outputs = {
            'image': pred,
            'depth': pred_depth,
        }

        return outputs

    def train_one_epoch(self, loader):
        self.log(
            f"==> Start Training Epoch {self.epoch}, lr={self.optimizer_model.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()
        self.model_fxfy.train()
        self.model_pose.train()
        # self.model_camera.train()
        # self.optimizer_cam_model = self.optimizer_cam_func(self.model_camera)

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        print(" | self.global_step: {} | self.opt_state: {}".format(
            self.global_step,  self.opt_state))
        iter_states = eval(self.opt.dynamic_iters)
        # print(iter_states)
        # cond = np.array([self.global_step > u and self.global_step <
        #                  v for (u, v) in eval(self.opt.dynamic_iters)]).sum()
        cond = np.array([key for key in iter_states if self.global_step >= iter_states[key][0] and self.global_step <
                         iter_states[key][1]])
        # if ((self.global_step >= self.opt.max_static_iters) and self.opt_state != "dynamic"):
        if (('d1' in cond or 'd2' in cond or 'd3' in cond or 'd4' in cond) and self.opt_state != "dynamic"):
            # print("\n\n========================================")
            # print("DYNAMIC MODEL ACTIVATED!!! - (optimizer)")
            # print("========================================\n\n")
            self.opt_state = "dynamic"
            self.optimizer_model = self.optimizer_func(
                self.model, self.opt_state)
            # self.lr_scheduler = self.scheduler_func(self.optimizer)

            # for name, param in self.model.named_parameters():
            #     if param.requires_grad:
            #         print(name)

        elif (('b1' in cond or 'b2' in cond or 'b3' in cond or 'b4' in cond) and self.opt_state != "all"):
            # print("\n\n========================================")
            # print("COMBINED MODEL ACTIVATED!!! - (optimizer)")
            # print("========================================\n\n")
            self.opt_state = "all"
            self.optimizer_model = self.optimizer_func(
                self.model, self.opt_state)
            # self.lr_scheduler = self.scheduler_func(self.optimizer)
        elif ('b1' not in cond and 'b2' not in cond and 'b3' not in cond and 'b4' not in cond and 'd1' not in cond and 'd2' not in cond and 'd3' not in cond and 'd4' not in cond):
            # print("\n\n========================================")
            # print("STATIC MODEL ACTIVATED!!! - (optimizer)")
            # print("========================================\n\n")
            self.opt_state = "static"
            self.optimizer_model = self.optimizer_func(
                self.model, self.opt_state)
            # self.lr_scheduler = self.scheduler_func(self.optimizer)

        for data in loader:

            # update grid every 16 steps
            # FIXME: Not sure exactly how this works
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0 and self.global_step < 5000:  # FIXME: Add to defaults
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()

            self.local_step += 1
            self.global_step += 1

            self.optimizer_model.zero_grad()
            self.optimizer_fxfy.zero_grad()
            self.optimizer_pose.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss = self.train_step(data)

            self.scaler.scale(loss).backward()

            # Results in nan/inf errors
            # break

            # print("local_step: {}".format(self.local_step))
            # print("global_step: {}".format(self.global_step))

            self.scaler.step(self.optimizer_model)
            # self.scaler.step(self.optimizer_fxfy)
            # TODO: Add to config
            if (self.global_step <= 20000 and self.pred_extrinsics):
                self.scaler.step(self.optimizer_pose)
            if (self.global_step <= 3600 and self.pred_intrinsics):
                self.scaler.step(self.optimizer_fxfy)

            # print("\n\n\n model_fxfy")
            # for p in self.model_fxfy.parameters():
            #     print(p.name, p.data, p.requires_grad, p.grad, p.is_leaf)

            # print("\n\n\n model_pose")
            # for p in self.model_pose.parameters():
            #     print(p.name, p.data, p.grad, p.is_leaf)

            self.scaler.update()

            # print("\n\n\n model_parameters")
            # for p in self.model.parameters():
            #     print(p.name, p.data, p.requires_grad, p.grad, p.is_leaf)

            # self.optimizer_model.step()
            # self.optimizer_fxfy.step()
            # self.optimizer_pose.step()
            # self.optimizer_model.zero_grad()
            # self.optimizer_fxfy.zero_grad()
            # self.optimizer_pose.zero_grad()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)

                if self.use_tensorboardX:
                    self.writer.add_scalar(
                        "validation/loss", loss_val, self.global_step)
                    self.writer.add_scalar(
                        "validation/lr", self.optimizer_model.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(
                        f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer_model.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(
                        f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")

    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0

            for data in loader:
                self.local_step += 1

                # print("data.keys(): {}".format(data.keys()))
                # print("data.time: {}".format(data["time"]))

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, truths, loss = self.eval_step(data)

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size

                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(
                        self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(
                        self.device) for _ in range(self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                    truths_list = [torch.zeros_like(truths).to(self.device) for _ in range(
                        self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(truths_list, truths)
                    truths = torch.cat(truths_list, dim=0)

                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:

                    for metric in self.metrics:
                        metric.update(preds, truths)

                    # # save image
                    # save_path = os.path.join(
                    #     self.workspace, 'validation', f'{name}_{self.local_step:04d}.png')
                    # # save_path_depth = os.path.join(
                    # #     self.workspace, 'validation', f'{name}_{self.local_step:04d}_depth.png')
                    # save_path_gt = os.path.join(
                    #     self.workspace, 'validation', f'{name}_{self.local_step:04d}_gt.png')

                    # # self.log(f"==> Saving validation image to {save_path}")
                    # os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    # if self.opt.color_space == 'linear':
                    #     preds = linear_to_srgb(preds)

                    # pred = preds[0].detach().cpu().numpy()
                    # # pred_depth = preds_depth[0].detach().cpu().numpy()
                    # truth = truths[0].detach().cpu().numpy()

                    # cv2.imwrite(save_path, cv2.cvtColor(
                    #     (pred * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                    # # cv2.imwrite(save_path_depth,
                    # #             (pred_depth * 255).astype(np.uint8))
                    # cv2.imwrite(save_path_gt, cv2.cvtColor(
                    #     (truth * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                    # Save for overall calcs ========================================
                    # save image
                    EVAL_FLAG = True
                    if (EVAL_FLAG):
                        save_path = os.path.join(
                            "results", 'Ours', self.workspace, f'v{0:03d}_t{self.local_step-1:03d}.png')
                        save_path_gt = os.path.join(
                            "results", 'Ours', self.workspace, f'v{0:03d}_t{self.local_step-1:03d}_gt.png')
                        save_path_depth = os.path.join(
                            "results", 'Ours', self.workspace, f'v{0:03d}_t{self.local_step-1:03d}_depth.png')

                        # self.log(f"==> Saving validation image to {save_path}")
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        # os.makedirs(os.path.dirname(
                        #     save_path_gt), exist_ok=True)

                        if self.opt.color_space == 'linear':
                            preds = linear_to_srgb(preds)

                        pred = preds[0].detach().cpu().numpy()
                        depth = preds_depth[0].detach().cpu().numpy()
                        truth = truths[0].detach().cpu().numpy()
                        # print("\npreds.shape: {}".format(preds.shape))
                        # print("truth.shape: {}".format(truth.shape))
                        # print("preds.mean: {}".format(preds.mean()))
                        # print("truth.mean: {}".format(truth.mean()))
                        # print("loss_val: {}".format(loss_val))

                        cv2.imwrite(save_path, cv2.cvtColor(
                            (pred * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                        # cv2.imwrite(save_path_depth,
                        #             (depth * 255).astype(np.uint8))
                        # cv2.imwrite(save_path_gt, cv2.cvtColor(
                        #     (truth * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                        # cv2.imwrite(save_path_gt, cv2.cvtColor(
                        #     (truth * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                    # ===============================================================

                    pbar.set_description(
                        f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                    pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                # if max mode, use -result
                self.stats["results"].append(
                    result if self.best_mode == 'min' else - result)
            else:
                # if no metric, choose best by min loss
                self.stats["results"].append(average_loss)

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            state['mean_count'] = self.model.mean_count
            state['mean_density'] = self.model.mean_density

        if full:
            state['optimizer'] = self.optimizer_model.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()

        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{name}.pth"

            if remove_old:
                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = self.stats["checkpoints"].pop(0)
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

            torch.save(state, file_path)

        else:
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(
                        f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    # we don't consider continued training from the best ckpt, so we discard the unneeded density_grid to save some storage (especially important for dnerf)
                    if 'density_grid' in state['model']:
                        del state['model']['density_grid']

                    if self.ema is not None:
                        self.ema.restore()

                    torch.save(state, self.best_path)
            else:
                self.log(
                    f"[WARN] no evaluated results found, skip saving best checkpoint.")

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(
                glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log(
                    "[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(
            checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if self.ema is not None and 'ema' in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict['ema'])

        if self.model.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.model.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']

        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(
            f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")

        if self.optimizer_model and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer_model.load_state_dict(
                    checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")

        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(
                    checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")

        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")
