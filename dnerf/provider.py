import os
import cv2
import sys
import glob
import json
import tqdm
import numpy as np
from dnerf.network_camera import CameraNetwork
from scipy.spatial.transform import Slerp, Rotation
import trimesh
import torch
from torch.utils.data import DataLoader
sys.path.append("..")  # noqa: E501
from .utils import get_rays, srgb_to_linear
from utils.flow_utils import resize_flow
# torch.autograd.set_detect_anomaly(True)
# from dnerf.network_camera import CameraNetwork # Sent in from main_dnerf


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose


def visualize_poses(poses, size=0.1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [
                        a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


def rand_poses(size, device, radius=1, theta_range=[np.pi/3, 2*np.pi/3], phi_range=[0, 2*np.pi]):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, \pi]
        phi_range: [min, max], should be in [0, 2\pi]
    Return:
        poses: [size, 4, 4]
    '''

    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    thetas = torch.rand(size, device=device) * \
        (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(size, device=device) * \
        (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1)  # [B, 3]

    # lookat
    forward_vector = - normalize(centers)
    # confused at the coordinate system...
    up_vector = torch.FloatTensor(
        [0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(
        0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack(
        (right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses


class NeRFDataset:
    def __init__(self, opt, device, type='train', downscale=1, n_test=10):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type  # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.preload = opt.preload  # preload data into GPU
        # camera radius scale to make sure camera are inside the bounding box.
        self.scale = opt.scale
        self.offset = opt.offset  # camera offset
        # bounding box half length, also used as the radius to random sample poses.
        self.bound = opt.bound
        self.fp16 = opt.fp16  # if preload, load into fp16.

        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1
        # self.masks = self.masks_val if self.training else self.masks

        self.rand_pose = opt.rand_pose
        self.DYNAMIC_ITERS = eval(opt.dynamic_iters)
        self.DYNAMIC_ITER = 0

        # auto-detect transforms.json and split mode.
        if os.path.exists(os.path.join(self.root_path, 'transforms.json')):
            # manually split, use view-interpolation for test.
            self.mode = 'colmap'
        elif os.path.exists(os.path.join(self.root_path, 'transforms_train.json')):
            self.mode = 'blender'  # provided split
        else:
            raise NotImplementedError(
                f'[NeRFDataset] Cannot find transforms*.json under {self.root_path}')

        # load nerf-compatible format data.
        if self.mode == 'colmap':
            with open(os.path.join(self.root_path, 'transforms.json'), 'r') as f:
                transform = json.load(f)
        elif self.mode == 'blender':
            # load all splits (train/valid/test), this is what instant-ngp in fact does...
            if type == 'all':
                transform_paths = glob.glob(
                    os.path.join(self.root_path, '*.json'))
                transform = None
                for transform_path in transform_paths:
                    with open(transform_path, 'r') as f:
                        tmp_transform = json.load(f)
                        if transform is None:
                            transform = tmp_transform
                        else:
                            transform['frames'].extend(tmp_transform['frames'])
            # load train and val split
            elif type == 'trainval':
                with open(os.path.join(self.root_path, f'transforms_train.json'), 'r') as f:
                    transform = json.load(f)
                with open(os.path.join(self.root_path, f'transforms_val.json'), 'r') as f:
                    transform_val = json.load(f)
                transform['frames'].extend(transform_val['frames'])
            # only load one specified split
            else:
                with open(os.path.join(self.root_path, f'transforms_{type}.json'), 'r') as f:
                    transform = json.load(f)

        else:
            raise NotImplementedError(f'unknown dataset mode: {self.mode}')

        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // downscale
            self.W = int(transform['w']) // downscale
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None

        # read images
        frames = transform["frames"]
        # frames = sorted(frames, key=lambda d: d['file_path']) # why do I sort...

        # for colmap, manually interpolate a test set.
        if self.mode == 'colmap' and type == 'test':

            # choose two random poses, and interpolate between.
            f0, f1 = np.random.choice(frames, 2, replace=False)
            pose0 = nerf_matrix_to_ngp(np.array(
                f0['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset)  # [4, 4]
            pose1 = nerf_matrix_to_ngp(np.array(
                f1['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset)  # [4, 4]
            time0 = f0['time'] if 'time' in f0 else int(
                os.path.basename(f0['file_path'])[:-4])
            time1 = f1['time'] if 'time' in f1 else int(
                os.path.basename(f1['file_path'])[:-4])
            rots = Rotation.from_matrix(
                np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)

            self.poses = []
            self.images = None
            self.times = []
            for i in range(n_test + 1):
                ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                self.poses.append(pose)
                time = (1 - ratio) * time0 + ratio * time1
                self.times.append(time)

            # manually find max time to normalize
            if 'time' not in f0:
                max_time = 0
                for f in frames:
                    max_time = max(max_time, int(
                        os.path.basename(f['file_path'])[:-4]))
                self.times = [t / max_time for t in self.times]

        else:
            # for colmap, manually split a valid set (the first frame).
            if self.mode == 'colmap':
                if type == 'train':
                    frames = frames[1:]
                elif type == 'val':
                    frames = frames[:1]
                # else 'all' or 'trainval' : use all frames

            self.poses = []
            self.images = []
            self.times = []

            # print("frames: {}".format(frames))

            # assume frames are already sorted by time!
            for f in tqdm.tqdm(frames, desc=f'Loading {type} data'):
                f_path = os.path.join(self.root_path, f['file_path'])
                if self.mode == 'blender' and '.' not in os.path.basename(f_path):
                    f_path += '.png'  # so silly...

                # there are non-exist paths in fox...
                if not os.path.exists(f_path):
                    continue

                # print("f_path: {}".format(f_path))
                # print("f: {}".format(f['transform_matrix']))

                pose = np.array(f['transform_matrix'],
                                dtype=np.float32)  # [4, 4]
                pose = nerf_matrix_to_ngp(
                    pose, scale=self.scale, offset=self.offset)

                # [H, W, 3] o [H, W, 4]
                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED)
                if self.H is None or self.W is None:
                    self.H = image.shape[0] // downscale
                    self.W = image.shape[1] // downscale

                # add support for the alpha channel as a mask.
                if image.shape[-1] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                if image.shape[0] != self.H or image.shape[1] != self.W:
                    image = cv2.resize(image, (self.W, self.H),
                                       interpolation=cv2.INTER_AREA)

                image = image.astype(np.float32) / 255  # [H, W, 3/4]

                # frame time
                if 'time' in f:
                    time = f['time']
                else:
                    # assume frame index as time
                    time = int(os.path.basename(f['file_path'])[:-4])

                self.poses.append(pose)
                self.images.append(image)
                # print("image.shape: {}".format(image.shape))
                self.times.append(time)

        self.poses = torch.from_numpy(
            np.stack(self.poses, axis=0))  # [N, 4, 4]
        if self.images is not None:
            self.images = torch.from_numpy(
                np.stack(self.images, axis=0))  # [N, H, W, C]
        self.times = torch.from_numpy(np.asarray(
            self.times, dtype=np.float32)).view(-1, 1)  # [N, 1]

        # manual normalize
        if self.times.max() > 1:
            self.times = self.times / \
                (self.times.max() + 1e-8)  # normalize to [0, 1]

        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()
        #print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

        # initialize error_map
        if self.training and self.opt.error_map:
            # [B, 128 * 128], flattened for easy indexing, fixed resolution...
            self.error_map = torch.ones(
                [self.images.shape[0], 128 * 128], dtype=torch.float)
        else:
            self.error_map = None

        # [debug] uncomment to view all training poses.
        # visualize_poses(self.poses.numpy())

        # [debug] uncomment to view examples of randomly generated poses.
        # visualize_poses(rand_poses(100, self.device, radius=self.radius).cpu().numpy())
        self.FLOW_FLAG = True
        self.PRED_POSE = True
        if (self.FLOW_FLAG):
            # TODO: ADD the additional pre-reqs here
            basedir = self.root_path
            disp_dir = os.path.join(basedir, 'disp')
            sh = image.shape[:2]
            num_img = len(frames)  # FIXME

            dispfiles = [os.path.join(disp_dir, f)
                         for f in sorted(os.listdir(disp_dir)) if f.endswith('npy')]

            disp = [cv2.resize(np.load(f),
                               (sh[1], sh[0]),
                               interpolation=cv2.INTER_NEAREST) for f in dispfiles]
            disp = np.stack(disp, -1)

            # sk_debug: used to be `motion_masks`
            mask_dir = os.path.join(basedir, 'motion_masks')
            maskfiles = [os.path.join(mask_dir, f)
                         for f in sorted(os.listdir(mask_dir)) if f.endswith('png')]

            masks = [cv2.resize(cv2.imread(f)/255., (sh[1], sh[0]),
                                interpolation=cv2.INTER_NEAREST) for f in maskfiles]
            masks = np.stack(masks, -1)
            masks = np.float32(masks > 1e-3)

            # val
            mask_dir_val = os.path.join(basedir, 'motion_masks_val')
            maskfiles_val = [os.path.join(mask_dir_val, f)
                             for f in sorted(os.listdir(mask_dir_val)) if f.endswith('png')]

            masks_val = [cv2.resize(cv2.imread(f)/255., (sh[1], sh[0]),
                                    interpolation=cv2.INTER_NEAREST) for f in maskfiles_val]
            masks_val = np.stack(masks_val, -1)
            masks_val = np.float32(masks_val > 1e-3)

            flow_dir = os.path.join(basedir, 'flow')
            flows_f = []
            flow_masks_f = []
            flows_b = []
            flow_masks_b = []
            for i in range(num_img):
                if i == num_img - 1:
                    fwd_flow, fwd_mask = np.zeros(
                        (sh[0], sh[1], 2)), np.zeros((sh[0], sh[1]))
                else:
                    fwd_flow_path = os.path.join(flow_dir, '%05d_fwd.npz' % i)
                    fwd_data = np.load(fwd_flow_path)
                    fwd_flow, fwd_mask = fwd_data['flow'], fwd_data['mask']
                    fwd_flow = resize_flow(fwd_flow, sh[0], sh[1])
                    fwd_mask = np.float32(fwd_mask)
                    fwd_mask = cv2.resize(fwd_mask, (sh[1], sh[0]),
                                          interpolation=cv2.INTER_NEAREST)
                flows_f.append(fwd_flow)
                flow_masks_f.append(fwd_mask)

                if i == 0:
                    bwd_flow, bwd_mask = np.zeros(
                        (sh[0], sh[1], 2)), np.zeros((sh[0], sh[1]))
                else:
                    bwd_flow_path = os.path.join(flow_dir, '%05d_bwd.npz' % i)
                    bwd_data = np.load(bwd_flow_path)
                    bwd_flow, bwd_mask = bwd_data['flow'], bwd_data['mask']
                    bwd_flow = resize_flow(bwd_flow, sh[0], sh[1])
                    bwd_mask = np.float32(bwd_mask)
                    bwd_mask = cv2.resize(bwd_mask, (sh[1], sh[0]),
                                          interpolation=cv2.INTER_NEAREST)
                flows_b.append(bwd_flow)
                flow_masks_b.append(bwd_mask)

            flows_f = np.stack(flows_f, -1)
            flow_masks_f = np.stack(flow_masks_f, -1)
            flows_b = np.stack(flows_b, -1)
            flow_masks_b = np.stack(flow_masks_b, -1)

            imgs = self.images  # sk_debug

            # Get grid
            i, j = np.meshgrid(np.arange(self.W, dtype=np.float32),
                               np.arange(self.H, dtype=np.float32), indexing='xy')

            # print("i.shape: {}".format(i.shape))
            # print("j.shape: {}".format(j.shape))
            # print("flows_b.shape: {}".format(flows_b.shape))
            # print("flows_f.shape: {}".format(flows_f.shape))
            # print("flow_masks_b.shape: {}".format(flow_masks_b.shape))
            # print("flow_masks_f.shape: {}".format(flow_masks_f.shape))

            self.grid = np.empty((0, self.H, self.W, 8), np.float32)
            for idx in range(num_img):
                self.grid = np.concatenate((self.grid, np.stack([i,
                                                                 j,
                                                                 flows_f[:,
                                                                         :, 0, idx],
                                                                 flows_f[:,
                                                                         :, 1, idx],
                                                                 flow_masks_f[:,
                                                                              :, idx],
                                                                 flows_b[:,
                                                                         :, 0, idx],
                                                                 flows_b[:,
                                                                         :, 1, idx],
                                                                 flow_masks_b[:, :, idx]], -1)[None, ...]))

            # print("imgs.shape: {}".format(imgs.shape))
            # print("disp.shape: {}".format(disp.shape))
            # print("masks.shape: {}".format(masks.shape))
            # print("flows_f.shape: {}".format(flows_f.shape))
            # print("flow_masks_f.shape: {}".format(flow_masks_f.shape))
            # print("flows_b.shape: {}".format(flows_b.shape))
            # print("flow_masks_b.shape: {}".format(flow_masks_b.shape))

            self.flows_f = flows_f
            self.flow_masks_f = flow_masks_f
            self.flows_b = flows_b
            self.flow_masks_b = flow_masks_b
            self.masks = torch.Tensor(masks).to(self.device)
            self.masks_val = torch.Tensor(masks_val).to(self.device)
            self.disp = disp

            # FIXME: sk_debug
            # assert(imgs.shape[0] == disp.shape[-1])
            # assert(imgs.shape[0] == masks.shape[-1])
            # assert(imgs.shape[0] == flows_f.shape[-1])
            # assert(imgs.shape[0] == flow_masks_f.shape[-1])
            # assert(imgs.shape[1] == disp.shape[-1])
            # assert(imgs.shape[1] == masks.shape[-1])

        if self.preload:
            self.poses = self.poses.to(self.device)
            if self.images is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.fp16 and self.opt.color_space != 'linear':
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.images = self.images.to(dtype).to(self.device)
            if self.error_map is not None:
                self.error_map = self.error_map.to(self.device)
            self.times = self.times.to(self.device)

        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (
                transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (
                transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / \
                (2 * np.tan(transform['camera_angle_x'] / 2)
                 ) if 'camera_angle_x' in transform else None
            fl_y = self.H / \
                (2 * np.tan(transform['camera_angle_y'] / 2)
                 ) if 'camera_angle_y' in transform else None
            if fl_x is None:
                fl_x = fl_y
            if fl_y is None:
                fl_y = fl_x
        else:
            raise RuntimeError(
                'Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] /
              downscale) if 'cx' in transform else (self.W / 2)
        cy = (transform['cy'] /
              downscale) if 'cy' in transform else (self.H / 2)

        self.intrinsics = np.array([fl_x, fl_y, cx, cy])

        # camera predictions
        self.model_camera = opt.model_camera

    def collate(self, index):

        B = len(index)  # a list of length 1

        # random pose without gt images.
        if self.rand_pose == 0 or index[0] >= len(self.poses):

            poses = rand_poses(B, self.device, radius=self.radius)

            # sample a low-resolution but full image for CLIP
            # only in training, assert num_rays > 0
            s = np.sqrt(self.H * self.W / self.num_rays)
            rH, rW = int(self.H / s), int(self.W / s)
            rays = get_rays(poses, self.intrinsics / s, rH, rW, -1, -1, -1)

            return {
                'H': rH,
                'W': rW,
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],
            }

        poses = self.poses[index].to(self.device)  # [B, 4, 4]
        times = self.times[index].to(self.device)  # [B, 1]

        error_map = None if self.error_map is None else self.error_map[index]

        if (self.FLOW_FLAG):
            masks = torch.reshape(self.masks, (-1, self.masks.shape[2], self.masks.shape[3]))[
                :, :, index].to(self.device)  # [B, N]
            masks_val = torch.reshape(self.masks_val, (-1, self.masks.shape[2], self.masks.shape[3]))[
                :, :, index].to(self.device)  # [B, N]
            grid = torch.Tensor(self.grid)
            grid = torch.reshape(
                grid, (grid.shape[0], -1, grid.shape[-1]))
        else:
            masks = None
            masks_val = None
            grid = None

        if (self.PRED_POSE):
            poses_gt = poses
            intrinsics_gt = self.intrinsics
            fxfy_pred, poses_pred = self.model_camera(
                index)

            # cpu -> gpu
            self.intrinsics = torch.Tensor(
                self.intrinsics).to(self.device)  # [B, 2]
            # poses = torch.unsqueeze(poses_pred, 0).to(self.device)  # [B, 4, 4]
            # poses_pred = torch.unsqueeze(
            #     poses_pred, 0).to(self.device)  # [B, 4, 4]

            # Assign intrinsics here
            # self.intrinsics[:2] = fxfy_pred

            print()
            print()
            print("fxfy_actual: {}\nposes_actual: {}".format(
                intrinsics_gt, poses_gt))
            # print("fxfy_pred: {} - poses_pred: {}".format(fxfy_pred, poses_pred))
            print("fxfy_actual.shape: {}\nposes_actual.shape: {}".format(
                intrinsics_gt.shape, poses_gt.shape))
            # print(
            #     "fxfy_pred.shape: {} - poses_pred.shape: {}".format(fxfy.shape, poses_pred.shape))
            print("fxfy_new: {}\nposes_new: {}".format(
                self.intrinsics, poses))
            print(
                "fxfy_new.shape: {}\nposes_new.shape: {}".format(self.intrinsics.shape, poses.shape))
            print()

        if self.training:
            rays = get_rays(poses, self.intrinsics, self.H,
                            self.W, masks, self.num_rays, error_map, self.DYNAMIC_ITER, self.DYNAMIC_ITERS)  # sk_debug - added masks
        else:
            rays = get_rays(poses, self.intrinsics, self.H,
                            self.W, masks_val, self.num_rays, error_map, self.DYNAMIC_ITER, self.DYNAMIC_ITERS)  # sk_debug - added masks

        self.DYNAMIC_ITER += 1

        if ("inds_s" in rays and "inds_d" in rays):
            self.inds_s = rays["inds_s"]
            self.inds_d = rays["inds_d"]
        else:
            self.inds_s = 0
            self.inds_d = 0

        indices = rays["inds"] if self.training else -1

        if (self.FLOW_FLAG):
            grid = grid[:, indices, :]

        results = {
            'H': self.H,
            'W': self.W,
            'grids': grid,
            'intrinsics': self.intrinsics,
            'all_poses': self.poses,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'time': times,
            'poses': poses,
            'FLOW_FLAG': self.FLOW_FLAG,
            "inds_s": self.inds_s,
            "inds_d": self.inds_d
        }

        if self.images is not None:
            # [B, H, W, 3/4]
            # print("index: {}".format(index))
            i = index[0]
            images_b = self.images[i-1].to(self.device) if i - \
                1 > 0 else self.images[i].to(self.device)
            images_f = self.images[i+1].to(self.device) if i+1 < len(
                self.images) else self.images[index].to(self.device)   # [B, H, W, 3/4]
            images = self.images[index].to(self.device)  # [B, H, W, 3/4]

            if self.training:
                C = images.shape[-1]
                images = torch.gather(images.view(
                    B, -1, C), 1, torch.stack(C * [rays['inds']], -1))  # [B, N, 3/4]
                images_b = torch.gather(images_b.view(
                    B, -1, C), 1, torch.stack(C * [rays['inds']], -1))  # [B, N, 3/4]
                images_f = torch.gather(images_f.view(
                    B, -1, C), 1, torch.stack(C * [rays['inds']], -1))  # [B, N, 3/4]
            results['images'] = images
            results['images_b'] = images_b
            results['images_f'] = images_f

        if (self.FLOW_FLAG):
            if self.masks is not None:
                index = index[0]
                # print(index)
                # [B, H, W, 3/4]
                # print(self.masks[:, :, :, index].shape)
                # print(self.disp[:, :, index].shape)
                masks = self.masks[:, :, :, index]
                disp = torch.Tensor(self.disp[:, :, index]).to(
                    self.device)  # [B, H, W, 3/4]
                # print("masks.shape: {}".format(masks.shape))
                # print("disp.shape: {}".format(disp.shape))
                # print("images.shape: {}".format(images.shape))
                # print("rays['inds'].shape: {}".format(rays['inds'].shape))
                # print("disp.view(B, -1, 1).shape: {}".format(disp.view(1, -1, 1).shape))
                if self.training:
                    masks = torch.gather(masks.view(
                        B, -1, 3), 1, torch.stack(3 * [rays['inds']], -1))  # [B, N, 3/4]
                    disp = torch.gather(disp.view(
                        B, -1, 1), 1, torch.stack(1 * [rays['inds']], -1))  # [B, N, 3/4]
                results['disp'] = disp
                results['masks'] = masks

        # need inds to update error_map
        results['index'] = index
        results['num_img'] = len(self.images)
        if error_map is not None:
            results['inds_coarse'] = rays['inds_coarse']

        return results

    def dataloader(self):
        size = len(self.poses)
        if self.training and self.rand_pose > 0:
            # index >= size means we use random pose.
            size += size // self.rand_pose
        loader = DataLoader(list(range(size)), batch_size=1,
                            collate_fn=self.collate, shuffle=self.training, num_workers=0)
        # an ugly fix... we need to access error_map & poses in trainer.
        loader._data = self
        loader.has_gt = self.images is not None
        return loader
