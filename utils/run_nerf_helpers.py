import os
import torch
import imageio
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Misc utils
def img2mse(x, y, M=None):
    # print("x: {}".format(x.shape))
    # print("gt: {}".format(y.shape))
    # print("x.isnan: {}".format(x.isnan().sum()))
    # print("gt.isnan: {}".format(y.isnan().sum()))
    if M == None:
        return torch.mean((x - y) ** 2)
    else:
        return torch.sum((x - y) ** 2 * M) / (torch.sum(M) + 1e-8) / x.shape[-1]


def img2mae(x, y, M=None):
    if M == None:
        return torch.mean(torch.abs(x - y))
    else:
        return torch.sum(torch.abs(x - y) * M) / (torch.sum(M) + 1e-8) / x.shape[-1]


def L1(x, M=None):
    if M == None:
        return torch.mean(torch.abs(x))
    else:
        return torch.sum(torch.abs(x) * M) / (torch.sum(M) + 1e-8) / x.shape[-1]


def L2(x, M=None):
    if M == None:
        return torch.mean(x ** 2)
    else:
        return torch.sum((x ** 2) * M) / (torch.sum(M) + 1e-8) / x.shape[-1]


def entropy(x):
    return -torch.sum(x * torch.log(x + 1e-19)) / x.shape[0]


def mse2psnr(x): return -10. * torch.log(x) / \
    torch.log(torch.Tensor([10.]).to(device))


def to8b(x): return (255 * np.clip(x, 0, 1)).astype(np.uint8)


# Ray helpers
def get_rays_new(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    i, j = torch.meshgrid(torch.linspace(
        0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack(
        [(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """Normalized device coordinate rays.
    Space such that the canvas is a cube with sides [-1, 1] in each axis.
    Args:
      H: int. Height in pixels.
      W: int. Width in pixels.
      focal: float. Focal length of pinhole camera.
      near: float or array of shape[batch_size]. Near depth bound for the scene.
      rays_o: array of shape [batch_size, 3]. Camera origin.
      rays_d: array of shape [batch_size, 3]. Ray direction.
    Returns:
      rays_o: array of shape [batch_size, 3]. Camera origin in NDC.
      rays_d: array of shape [batch_size, 3]. Ray direction in NDC.
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1./(H/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1./(W/(2.*focal)) * \
        (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
    d1 = -1./(H/(2.*focal)) * \
        (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


def get_grid(H, W, num_img, flows_f, flow_masks_f, flows_b, flow_masks_b):

    # |--------------------|  |--------------------|
    # |       j            |  |       v            |
    # |   i   *            |  |   u   *            |
    # |                    |  |                    |
    # |--------------------|  |--------------------|

    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')

    grid = np.empty((0, H, W, 8), np.float32)
    for idx in range(num_img):
        grid = np.concatenate((grid, np.stack([i,
                                               j,
                                               flows_f[idx, :, :, 0],
                                               flows_f[idx, :, :, 1],
                                               flow_masks_f[idx, :, :],
                                               flows_b[idx, :, :, 0],
                                               flows_b[idx, :, :, 1],
                                               flow_masks_b[idx, :, :]], -1)[None, ...]))
    return grid


def NDC2world(pts, H, W, f):

    # NDC coordinate to world coordinate
    pts_z = 2 / (torch.clamp(pts[..., 2:], min=-1., max=1-1e-3) - 1)
    pts_x = - pts[..., 0:1] * pts_z * W / 2 / f
    pts_y = - pts[..., 1:2] * pts_z * H / 2 / f
    pts_world = torch.cat([pts_x, pts_y, pts_z], -1)

    return pts_world


def render_3d_point(H, W, f, pose, weights, pts):
    """Render 3D position along each ray and project it to the image plane.
    """

    c2w = pose
    w2c = c2w[:3, :3].transpose(0, 1)  # same as np.linalg.inv(c2w[:3, :3])

    # Rendered 3D position in NDC coordinate
    pts_map_NDC = torch.sum(weights[..., None] * pts, -2)

    # NDC coordinate to world coordinate
    pts_map_world = NDC2world(pts_map_NDC, H, W, f)

    # World coordinate to camera coordinate
    # Translate
    pts_map_world = pts_map_world - c2w[:, 3]
    # Rotate
    pts_map_cam = torch.sum(pts_map_world[..., None, :] * w2c[:3, :3], -1)

    # Camera coordinate to 2D image coordinate
    pts_plane = torch.cat([pts_map_cam[..., 0:1] / (- pts_map_cam[..., 2:]) * f + W * .5,
                           - pts_map_cam[..., 1:2] / (- pts_map_cam[..., 2:]) * f + H * .5],
                          -1)

    return pts_plane


def induce_flow(H, W, focal, pose_neighbor, weights, pts_3d_neighbor, pts_2d):

    # Render 3D position along each ray and project it to the neighbor frame's image plane.
    pts_2d_neighbor = render_3d_point(H, W, focal,
                                      pose_neighbor,
                                      weights,
                                      pts_3d_neighbor)
    induced_flow = pts_2d_neighbor - pts_2d

    return induced_flow


def compute_depth_loss(dyn_depth, gt_depth):

    t_d = torch.median(dyn_depth)
    s_d = torch.mean(torch.abs(dyn_depth - t_d))
    dyn_depth_norm = (dyn_depth - t_d) / s_d

    t_gt = torch.median(gt_depth)
    s_gt = torch.mean(torch.abs(gt_depth - t_gt))
    gt_depth_norm = (gt_depth - t_gt) / s_gt

    return torch.mean((dyn_depth_norm - gt_depth_norm) ** 2)


def normalize_depth(depth):
    return torch.clamp(depth / percentile(depth, 97), 0., 1.)


def percentile(t, q):
    """
    Return the ``q``-th percentile of the flattened input tensor's data.

    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """

    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result


def save_res(moviebase, ret, fps=None):

    if fps == None:
        if len(ret['rgbs']) < 25:
            fps = 4
        else:
            fps = 24

    for k in ret:
        if 'rgbs' in k:
            imageio.mimwrite(moviebase + k + '.mp4',
                             to8b(ret[k]), fps=fps, quality=8, macro_block_size=1)
            # imageio.mimsave(moviebase + k + '.gif',
            #                  to8b(ret[k]), format='gif', fps=fps)
        elif 'depths' in k:
            imageio.mimwrite(moviebase + k + '.mp4',
                             to8b(ret[k]), fps=fps, quality=8, macro_block_size=1)
            # imageio.mimsave(moviebase + k + '.gif',
            #                  to8b(ret[k]), format='gif', fps=fps)
        elif 'disps' in k:
            imageio.mimwrite(moviebase + k + '.mp4',
                             to8b(ret[k] / np.max(ret[k])), fps=fps, quality=8, macro_block_size=1)
            # imageio.mimsave(moviebase + k + '.gif',
            #                  to8b(ret[k] / np.max(ret[k])), format='gif', fps=fps)
        elif 'sceneflow_' in k:
            imageio.mimwrite(moviebase + k + '.mp4',
                             to8b(norm_sf(ret[k])), fps=fps, quality=8, macro_block_size=1)
            # imageio.mimsave(moviebase + k + '.gif',
            #                  to8b(norm_sf(ret[k])), format='gif', fps=fps)
        elif 'flows' in k:
            imageio.mimwrite(moviebase + k + '.mp4',
                             ret[k], fps=fps, quality=8, macro_block_size=1)
            # imageio.mimsave(moviebase + k + '.gif',
            #                  ret[k], format='gif', fps=fps)
        elif 'dynamicness' in k:
            imageio.mimwrite(moviebase + k + '.mp4',
                             to8b(ret[k]), fps=fps, quality=8, macro_block_size=1)
            # imageio.mimsave(moviebase + k + '.gif',
            #                  to8b(ret[k]), format='gif', fps=fps)
        elif 'disocclusions' in k:
            imageio.mimwrite(moviebase + k + '.mp4',
                             to8b(ret[k][..., 0]), fps=fps, quality=8, macro_block_size=1)
            # imageio.mimsave(moviebase + k + '.gif',
            #                  to8b(ret[k][..., 0]), format='gif', fps=fps)
        elif 'blending' in k:
            blending = ret[k][..., None]
            blending = np.moveaxis(blending, [0, 1, 2, 3], [1, 2, 0, 3])
            imageio.mimwrite(moviebase + k + '.mp4',
                             to8b(blending), fps=fps, quality=8, macro_block_size=1)
            # imageio.mimsave(moviebase + k + '.gif',
            #                  to8b(blending), format='gif', fps=fps)
        elif 'weights' in k:
            imageio.mimwrite(moviebase + k + '.mp4',
                             to8b(ret[k]), fps=fps, quality=8, macro_block_size=1)
        else:
            raise NotImplementedError


def norm_sf_channel(sf_ch):

    # Make sure zero scene flow is not shifted
    sf_ch[sf_ch >= 0] = sf_ch[sf_ch >= 0] / sf_ch.max() / 2
    sf_ch[sf_ch < 0] = sf_ch[sf_ch < 0] / np.abs(sf_ch.min()) / 2
    sf_ch = sf_ch + 0.5
    return sf_ch


def norm_sf(sf):

    sf = np.concatenate((norm_sf_channel(sf[..., 0:1]),
                         norm_sf_channel(sf[..., 1:2]),
                         norm_sf_channel(sf[..., 2:3])), -1)
    sf = np.moveaxis(sf, [0, 1, 2, 3], [1, 2, 0, 3])
    return sf


# Spatial smoothness (adapted from NSFF)
def compute_sf_smooth_s_loss(pts1, pts2, H, W, f):

    N_samples = pts1.shape[1]

    # NDC coordinate to world coordinate
    pts1_world = NDC2world(pts1[..., :int(N_samples * 0.95), :], H, W, f)
    pts2_world = NDC2world(pts2[..., :int(N_samples * 0.95), :], H, W, f)

    # scene flow in world coordinate
    scene_flow_world = pts1_world - pts2_world

    return L1(scene_flow_world[..., :-1, :] - scene_flow_world[..., 1:, :])


# Temporal smoothness
def compute_sf_smooth_loss(pts, pts_f, pts_b, H, W, f):

    N_samples = pts.shape[1]

    pts_world = NDC2world(pts[..., :int(N_samples * 0.9), :],   H, W, f)
    pts_f_world = NDC2world(pts_f[..., :int(N_samples * 0.9), :], H, W, f)
    pts_b_world = NDC2world(pts_b[..., :int(N_samples * 0.9), :], H, W, f)

    # scene flow in world coordinate
    sceneflow_f = pts_f_world - pts_world
    sceneflow_b = pts_b_world - pts_world

    # For a 3D point, its forward and backward sceneflow should be opposite.
    return L2(sceneflow_f + sceneflow_b)
