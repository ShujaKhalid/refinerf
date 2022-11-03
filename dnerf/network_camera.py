from re import X
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer


class CameraNetwork(NeRFRenderer):
    def __init__(self,
                 h,
                 w,
                 num_cams,
                 learn_R=True,
                 learn_t=True,
                 learn_fx=True,
                 learn_fy=True,
                 **kwargs,
                 ):
        super().__init__(**kwargs)

        # pose & intrinsics
        self.num_cams = num_cams
        self.h, self.w = h, w
        self.r = nn.Parameter(torch.zeros(
            size=(self.num_cams, 3), dtype=torch.float32), requires_grad=learn_R)  # (N, 3)
        self.t = nn.Parameter(torch.zeros(
            size=(self.num_cams, 3), dtype=torch.float32), requires_grad=learn_t)  # (N, 3)
        self.fx = nn.Parameter(torch.tensor(
            1.0, dtype=torch.float32), requires_grad=learn_fx)  # (1, )
        self.fy = nn.Parameter(torch.tensor(
            1.0, dtype=torch.float32), requires_grad=learn_fy)  # (1, )
        # self.r = nn.Parameter(torch.randn(
        #     num_cams, 3, dtype=torch.float32), requires_grad=learn_R)  # (N, 3)
        # self.t = nn.Parameter(torch.randn(
        #     num_cams, 3, dtype=torch.float32), requires_grad=learn_t)  # (N, 3)
        # self.fx = nn.Parameter(torch.tensor(
        #     1.0, dtype=torch.float32), requires_grad=learn_fx)  # (1, )
        # self.fy = nn.Parameter(torch.tensor(
        #     1.0, dtype=torch.float32), requires_grad=learn_fy)  # (1, )

    def vec2skew(self, v):
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

    def Exp(self, r):
        """so(3) vector to SO(3) matrix
        :param r: (3, ) axis-angle, torch tensor
        :return:  (3, 3)
        """
        skew_r = self.vec2skew(r)  # (3, 3)
        norm_r = r.norm() + 1e-15
        eye = torch.eye(3, dtype=torch.float32, device=r.device)
        R = eye + (torch.sin(norm_r) / norm_r) * skew_r + \
            ((1 - torch.cos(norm_r)) / norm_r**2) * (skew_r @ skew_r)
        return R

    def make_c2w(self, r, t):
        """
        :param r:  (3, ) axis-angle             torch tensor
        :param t:  (3, ) translation vector     torch tensor
        :return:   (4, 4)
        """
        R = self.Exp(r)  # (3, 3)
        c2w = torch.cat([R, t.unsqueeze(1)], dim=1)  # (3, 4)
        c2w = self.convert3x4_4x4(c2w)  # (4, 4)
        return c2w

    def convert3x4_4x4(self, input):
        """
        :param input:  (N, 3, 4) or (3, 4) torch or np
        :return:       (N, 4, 4) or (4, 4) torch or np
        """
        if torch.is_tensor(input):
            if len(input.shape) == 3:
                output = torch.cat([input, torch.zeros_like(
                    input[:, 0:1])], dim=1)  # (N, 4, 4)
                output[:, 3, 3] = 1.0
            else:
                output = torch.cat([input, torch.tensor(
                    [[0, 0, 0, 1]], dtype=input.dtype, device=input.device)], dim=0)  # (4, 4)
        else:
            if len(input.shape) == 3:
                output = np.concatenate(
                    [input, np.zeros_like(input[:, 0:1])], axis=1)  # (N, 4, 4)
                output[:, 3, 3] = 1.0
            else:
                output = np.concatenate(
                    [input, np.array([[0, 0, 0, 1]], dtype=input.dtype)], axis=0)  # (4, 4)
                output[3, 3] = 1.0
        return output

    def forward(self, cam_id):

        fxfy = self.run_fxfy_network(self.h, self.w)
        pose = self.run_pose_network(cam_id)

        return fxfy, pose

    def run_pose_network(self, cam_id):
        r = torch.squeeze(self.r[cam_id])  # (3, ) axis-angle
        t = torch.squeeze(self.t[cam_id])  # (3, )

        c2w = self.make_c2w(r, t)  # (4, 4)
        return c2w

    def run_fxfy_network(self, h, w):
        fxfy = torch.stack([self.fx**2 * w, self.fy**2 * h])
        return fxfy

    # optimizer utils
    def get_params(self, lr_pose=0.001, lr_fxfy=0.001):
        params = [
            {'params': self.r, 'lr': lr_pose},
            {'params': self.t, 'lr': lr_pose},
            {'params': self.fx, 'lr': lr_fxfy},
            {'params': self.fy, 'lr': lr_fxfy},
        ]
        return params
