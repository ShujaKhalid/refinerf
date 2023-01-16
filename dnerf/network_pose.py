import torch
import random
import numpy as np
import torch.nn as nn


def convert3x4_4x4(input):
    """
    :param input:  (N, 3, 4) or (3, 4) torch or np
    :return:       (N, 4, 4) or (4, 4) torch or np
    """
    # if torch.is_tensor(input):
    if len(input.shape) == 3:
        output = torch.cat([input, torch.zeros_like(
            input[:, 0:1])], dim=1)  # (N, 4, 4)
        output[:, 3, 3] = 1.0
    else:
        output = torch.cat([input, torch.tensor(
            [[0, 0, 0, 1]], dtype=input.dtype, device=input.device)], dim=0)  # (4, 4)

    return output


def vec2skew(v):
    """
    :param v:  (3, ) torch tensor
    :return:   (3, 3)
    """
    zero = torch.zeros(1, dtype=torch.float32, device=v.device)
    skew_v0 = torch.cat([zero,    -v[2:3],   v[1:2]])  # (3, 1)
    skew_v1 = torch.cat([v[2:3],   zero,    -v[0:1]])
    skew_v2 = torch.cat([-v[1:2],   v[0:1],   zero])
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=0)  # (3, 3)
    return skew_v  # (3, 3)


def Exp(r):
    """so(3) vector to SO(3) matrix
    :param r: (3, ) axis-angle, torch tensor
    :return:  (3, 3)
    """
    skew_r = vec2skew(r)  # (3, 3)
    norm_r = r.norm() + 1e-15
    eye = torch.eye(3, dtype=torch.float32, device=r.device)
    R = eye + (torch.sin(norm_r) / norm_r) * skew_r + \
        ((1 - torch.cos(norm_r)) / norm_r**2) * (skew_r @ skew_r)
    return R


def make_c2w(r, t):
    """
    :param r:  (3, ) axis-angle             torch tensor
    :param t:  (3, ) translation vector     torch tensor
    :return:   (4, 4)
    """
    R = Exp(r)  # (3, 3)
    c2w = torch.cat([R, t.unsqueeze(1)], dim=1)  # (3, 4)
    c2w = convert3x4_4x4(c2w)  # (4, 4)
    return c2w


class LearnPose(nn.Module):
    def __init__(self, num_cams, noise_pct, learn_R=True, learn_t=True):
        super(LearnPose, self).__init__()
        self.num_cams = num_cams
        self.r = nn.Parameter(torch.zeros(
            size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_R)  # (N, 3)
        self.t = nn.Parameter(torch.zeros(
            size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_t)  # (N, 3)
        self.noise_pct = noise_pct
        self.c2w_noise = nn.Parameter(torch.ones(
            size=(4, 4), dtype=torch.float32), requires_grad=True)  # (N, 3)
        self.c2w_signs = torch.rand_like(self.c2w_noise).cuda()
        self.c2w_signs[self.c2w_signs > 0.5] = 1
        self.c2w_signs[self.c2w_signs <= 0.5] = -1

        # # test
        # self.c2w_test = torch.unsqueeze(
        #     torch.eye(4, dtype=torch.float32), dim=0).cuda()  # (N, 3)

    def forward(self, cam_id, poses_gt):

        NOISE_ABLATION = True

        # c2w = (c2w + torch.rand_like(c2w))/2
        if NOISE_ABLATION:
            # self.c2w_noise = poses_gt + torch.rand_like(poses_gt)
            c2w_noise = self.c2w_noise*poses_gt*self.noise_pct
            c2w_noise = c2w_noise * self.c2w_signs
            c2w = poses_gt + c2w_noise

            print(
                "\nError - R: {:.4f} - t: {:.4f} - all: {:.4f}".format(
                    torch.sum(torch.abs(c2w_noise[:3, :3])),
                    torch.sum(torch.abs(c2w_noise[:3, -1])),
                    torch.sum(torch.abs(c2w_noise))))
        else:
            r = torch.squeeze(self.r[cam_id])  # (3, ) axis-angle
            t = torch.squeeze(self.t[cam_id])  # (3, )
            c2w = make_c2w(r, t)  # (4, 4)
            c2w = torch.unsqueeze(c2w, dim=0)

        # bypass
        # c2w = self.c2w_test
        # c2w = torch.unsqueeze(c2w, dim=0)
        return c2w
