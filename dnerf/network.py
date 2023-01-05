from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="tiledgrid",  # tiledgrid (position_encoding)
                 # sphere_harmonics (direction_-encoding)
                 encoding_dir="sphere_harmonics",
                 encoding_time="frequency",  # frequency
                 encoding_deform="tiledgrid",  # "hashgrid" seems worse
                 encoding_bg="hashgrid",
                 num_layers=2,
                 hidden_dim=256,
                 geo_feat_dim=64,  # change me
                 num_layers_color=3,
                 hidden_dim_color=256,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 # a deeper MLP is very necessary for performance.
                 num_layers_deform=3,
                 hidden_dim_deform=256,
                 bound=1,
                 w=None,
                 h=None,
                 num_cams=None,
                 learn_R=True,
                 learn_t=True,
                 learn_fx=True,
                 learn_fy=True,
                 **kwargs,
                 ):
        super().__init__(bound, **kwargs)

        # Camera params
        self.num_cams = num_cams  # FIXME
        self.h, self.w = h, w
        self.r = nn.Parameter(torch.zeros(
            size=(1, 3), dtype=torch.float32), requires_grad=learn_R)  # (N, 3)
        self.t = nn.Parameter(torch.zeros(
            size=(1, 3), dtype=torch.float32), requires_grad=learn_t)  # (N, 3)
        self.fx = nn.Parameter(torch.tensor(
            1.0, dtype=torch.float32), requires_grad=learn_fx)  # (1, )
        self.fy = nn.Parameter(torch.tensor(
            1.0, dtype=torch.float32), requires_grad=learn_fy)  # (1, )

        # ==================
        # STATIC
        # ==================

        # sigma network ============================================
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        # ==================
        # Frequency encoding
        # ==================
        # static
        self.encoder_s_fact = 10   # 10 works
        self.encoder_dir_s_fact = 4  # 10 works
        # dynamic
        self.encoder_d_fact = 10   # 10 works
        self.encoder_dir_d_fact = 4  # 10 works
        self.encoder_d_constant = 1
        self.encoder_deform = 3
        self.encoder_time = 0

        if (encoding == "hashgrid"):
            self.encoder_s, self.in_dim_s = get_encoder(
                encoding, desired_resolution=self.encoder_s_fact*bound)
            self.encoder_d, self.in_dim_d = get_encoder(
                encoding, desired_resolution=self.encoder_d_fact*bound)
        elif (encoding == "tiledgrid"):
            self.encoder_s, self.in_dim_s = get_encoder(
                encoding, multires=self.encoder_s_fact)
            self.encoder_d, self.in_dim_d = get_encoder(
                encoding, multires=self.encoder_d_fact)
        else:
            self.encoder_s, self.in_dim_s = get_encoder(
                encoding, multires=10)
            self.encoder_d, self.in_dim_d = get_encoder(
                encoding, multires=10)

        sigma_s_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim_s
            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim  # 1 sigma + features for color
            else:
                out_dim = hidden_dim

            sigma_s_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_s_net = nn.ModuleList(sigma_s_net)

        # color network ============================================
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color

        if (encoding_dir == "hashgrid"):
            self.encoder_dir_s, self.in_dim_dir_s = get_encoder(
                encoding_dir, degree=self.encoder_dir_s_fact)
            self.encoder_dir_d, self.in_dim_dir_d = get_encoder(
                encoding_dir, degree=self.encoder_dir_d_fact)
        elif (encoding_dir == "tiledgrid"):
            self.encoder_dir_s, self.in_dim_dir_s = get_encoder(
                encoding_dir, multires=self.encoder_dir_s_fact)
            self.encoder_dir_d, self.in_dim_dir_d = get_encoder(
                encoding_dir, multires=self.encoder_dir_d_fact)
        else:
            self.encoder_dir_s, self.in_dim_dir_s = get_encoder(
                encoding_dir, multires=4)
            self.encoder_dir_d, self.in_dim_dir_d = get_encoder(
                encoding_dir, multires=4)

        color_s_net = []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_dir_s + self.geo_feat_dim
            else:
                in_dim = hidden_dim

            if l == num_layers_color - 1:
                out_dim = 3  # 3 rgb
            else:
                out_dim = hidden_dim

            color_s_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_s_net = nn.ModuleList(color_s_net)

        # ==================
        # DYNAMIC
        # ==================

        # Added for dynamic NeRF ============================================
        print("\nINITIALIZING DYNAMIC MODEL!!!\n")
        self.input_ch = 63
        self.input_ch_time = 257
        # self.D = 8  # FIXME: used to be 8!
        # self.W = 256  # FIXME: used to be 256!
        # self.skips = [4]
        # self.pts_linears = nn.ModuleList(
        #     [nn.Linear(self.input_ch, self.W)] + [nn.Linear(self.W, self.W) if i not in self.skips else nn.Linear(self.W + self.input_ch, self.W) for i in range(self.D-1)])
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(self.input_ch_views + self.W, self.W//2)])

        # deformation network ============================================
        self.num_layers_deform = num_layers_deform
        self.hidden_dim_deform = hidden_dim_deform

        if (encoding_deform == "hashgrid"):
            self.encoder_deform, self.in_dim_deform = get_encoder(
                encoding_deform, desired_resolution=bound)
        elif (encoding_deform == "tiledgrid"):
            self.encoder_deform, self.in_dim_deform = get_encoder(
                encoding_deform, multires=self.encoder_deform)  # FIXME: used to be 10
        else:
            self.encoder_deform, self.in_dim_deform = get_encoder(
                encoding_deform, multires=self.encoder_deform)  # FIXME: used to be 10

        if (encoding_time == "hashgrid"):
            self.encoder_time, self.in_dim_time = get_encoder(
                encoding_time, desired_resolution=bound)
        elif (encoding_time == "tiledgrid"):
            self.encoder_time, self.in_dim_time = get_encoder(
                encoding_time, input_dim=1, multires=self.encoder_time)  # FIXME: used to be 6
        else:
            self.encoder_time, self.in_dim_time = get_encoder(
                encoding_time, input_dim=1, multires=self.encoder_time)  # FIXME: used to be 6

        # print("\nin_dim_deform: {}".format(self.in_dim_deform))
        # print("in_dim_time: {}".format(self.in_dim_time))
        # print("in_dim_dir_d: {}".format(self.in_dim_dir_d))
        # print("geo_feat_dim: {}".format(self.geo_feat_dim))

        deform_d_net = []
        for l in range(num_layers_deform):
            if l == 0:
                in_dim = self.in_dim_deform + self.in_dim_time  # grid dim + time
            else:
                in_dim = hidden_dim_deform

            if l == num_layers_deform - 1:
                out_dim = 3  # deformation for xyz
            else:
                out_dim = hidden_dim_deform

            deform_d_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.deform_d_net = nn.ModuleList(deform_d_net)

        # sigma network ============================================
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        sigma_d_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim_d + self.in_dim_time + \
                    self.in_dim_deform  # concat everything
                # in_dim = self.in_dim_deform + self.in_dim_time + \
                #     self.in_dim_deform  # concat everything
            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim  # 1 sigma + features for color
            else:
                out_dim = hidden_dim

            sigma_d_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_d_net = nn.ModuleList(sigma_d_net)

        # color network ============================================
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color

        color_d_net = []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_dir_d + self.geo_feat_dim
            else:
                in_dim = hidden_dim

            if l == num_layers_color - 1:
                out_dim = 3  # 3 rgb
            else:
                out_dim = hidden_dim

            color_d_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_d_net = nn.ModuleList(color_d_net)

        # self.sf_net = nn.Linear(self.input_ch + self.input_ch_time, 6)
        # self.blend_net = nn.Linear(self.input_ch + self.input_ch_time, 1)

    def forward(self, x, d, t, svd):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # t: [1, 1], in [0, 1]
        # svd: [1], in ["static", "dynamic"]
        if (svd == "static"):
            sigma, rgbs = self.run_snerf(x, d)
            return sigma, rgbs
        elif (svd == "dynamic"):
            sigma, rgbs, deform, blend, sf = self.run_dnerf(x, d, t)
            return sigma, rgbs, deform, blend, sf
        elif (svd == "camera"):
            #fxfy = self.run_fxfy_network(self.h, self.w)
            #pose = self.run_pose_network()
            fxfy = nn.Parameter(torch.zeros(
                size=(1, 2), dtype=torch.float32), requires_grad=True)
            pose = nn.Parameter(torch.zeros(
                size=(4, 4), dtype=torch.float32), requires_grad=True)
            return fxfy, pose
        else:
            raise Exception("Run NeRF in either `static` or `dynamic` mode")

    def run_snerf(self, x, d):

        # sigma
        h = self.encoder_s(x, bound=self.bound)

        for l in range(self.num_layers):
            h = self.sigma_s_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        d = self.encoder_dir_s(d)

        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_s_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)

        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return sigma, rgbs

    def run_dnerf(self, x, d, t):
        # static
        # deform
        enc_ori_x = self.encoder_deform(x, bound=self.bound)  # [N, C]
        enc_t = self.encoder_time(t)  # [1, 1] --> [1, C']
        # print("\nt: {}".format(t))
        # print("x.mean: {}".format(x.mean()))
        if enc_t.shape[0] == 1:
            enc_t = enc_t.repeat(x.shape[0], 1)  # [1, C'] --> [N, C']

        deform = torch.cat([enc_ori_x, enc_t], dim=1)  # [N, C + C']
        # sf = torch.tanh(self.sf_net(deform))
        # blending = torch.sigmoid(self.blend_net(deform))
        # FIXME
        sf = deform[..., :6]
        blending = deform[..., 0]
        # print("x.shape: {}".format(x.shape))
        # print("enc_t.shape: {}".format(enc_t.shape))
        # print("enc_ori_x.shape: {}".format(enc_ori_x.shape))
        # print("deform.shape: {}".format(deform.shape))

        for l in range(self.num_layers_deform):
            deform = self.deform_d_net[l](deform)
            if l != self.num_layers_deform - 1:
                deform = F.relu(deform, inplace=True)

        # print(x)
        # print()
        # print(deform)
        # x_def = deform.float()  # FIXME: x + deform
        x_def = x + deform  # FIXME: x + deform
        # print("\nx_def.shape: {}".format(x_def.shape))

        # sigma
        # 4096 -> dim=32
        # 4096 -> dim=32
        x_def = self.encoder_d(
            x_def, bound=self.encoder_d_constant*self.bound)
        # x_def = self.encoder_deform(x_def, bound=self.bound)
        # print("x_def.shape: {}".format(x_def.shape))
        # print("enc_ori_x.shape: {}".format(enc_ori_x.shape))
        # print("enc_t.shape: {}".format(enc_t.shape))
        # TODO: Added -> confirm
        if (len(x.shape) == 3):
            h = torch.cat([x_def, enc_ori_x, enc_t], dim=-1)
        else:
            h = torch.cat([x_def, enc_ori_x, enc_t], dim=1)
        for l in range(self.num_layers):
            h = self.sigma_d_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        d = self.encoder_dir_d(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_d_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)

        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return sigma, rgbs, deform, blending, sf

    def density(self, x, t):
        # x: [N, 3], in [-bound, bound]
        # t: [1, 1], in [0, 1]

        results = {}
        # print("density_t: {}".format(t))
        # print("enc_t: {}".format(enc_t))

        # deformation
        enc_ori_x = self.encoder_deform(x, bound=self.bound)  # [N, C]
        enc_t = self.encoder_time(t)  # [1, 1] --> [1, C']
        if enc_t.shape[0] == 1:
            enc_t = enc_t.repeat(x.shape[0], 1)  # [1, C'] --> [N, C']
        # print("t: {}".format(t))
        # print("enc_ori_x.shape: {}".format(enc_ori_x.shape))
        # print("enc_t.shape: {}".format(enc_t.shape))

        deform = torch.cat([enc_ori_x, enc_t], dim=1)  # [N, C + C']
        for l in range(self.num_layers_deform):
            deform = self.deform_d_net[l](deform)
            if l != self.num_layers_deform - 1:
                deform = F.relu(deform, inplace=True)

        x = x + deform
        results['deform'] = deform

        # sigma
        x = self.encoder_d(x, bound=self.bound)  # FIXME
        # x = self.encoder_deform(x, bound=self.bound)
        h = torch.cat([x, enc_ori_x, enc_t], dim=1)
        for l in range(self.num_layers):
            h = self.sigma_d_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        results['sigma'] = sigma
        results['geo_feat'] = geo_feat

        return results

    # def background(self, x, d):
    #     # x: [N, 2], in [-1, 1]

    #     h = self.encoder_bg(x)  # [N, C]
    #     d = self.encoder_dir(d)

    #     h = torch.cat([d, h], dim=-1)
    #     for l in range(self.num_layers_bg):
    #         h = self.bg_d_net[l](h)
    #         if l != self.num_layers_bg - 1:
    #             h = F.relu(h, inplace=True)

    #     # sigmoid activation for rgb
    #     rgbs = torch.sigmoid(h)

    #     return rgbs

    # allow masked inference
    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # t: [1, 1], in [0, 1]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        if mask is not None:
            rgbs = torch.zeros(
                mask.shape[0], 3, dtype=x.dtype, device=x.device)  # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        d = self.encoder_dir_d(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_d_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)

        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype)  # fp16 --> fp32
        else:
            rgbs = h

        return rgbs

    # def vec2skew(self, v):
    #     """
    #     :param v:  (3, ) torch tensor
    #     :return:   (3, 3)
    #     """
    #     zero = torch.zeros(1, dtype=torch.float32, device=v.device)
    #     skew_v0 = torch.cat([zero,    -v[2:3],   v[1:2]])  # (3, 1)
    #     skew_v1 = torch.cat([v[2:3],   zero,    -v[0:1]])
    #     skew_v2 = torch.cat([-v[1:2],   v[0:1],   zero])
    #     skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=0)  # (3, 3)
    #     return skew_v  # (3, 3)

    # def Exp(self, r):
    #     """so(3) vector to SO(3) matrix
    #     :param r: (3, ) axis-angle, torch tensor
    #     :return:  (3, 3)
    #     """
    #     skew_r = self.vec2skew(r)  # (3, 3)
    #     norm_r = r.norm() + 1e-15
    #     eye = torch.eye(3, dtype=torch.float32, device=r.device)
    #     R = eye + (torch.sin(norm_r) / norm_r) * skew_r + \
    #         ((1 - torch.cos(norm_r)) / norm_r**2) * (skew_r @ skew_r)
    #     return R

    # def make_c2w(self, r, t):
    #     """
    #     :param r:  (3, ) axis-angle             torch tensor
    #     :param t:  (3, ) translation vector     torch tensor
    #     :return:   (4, 4)
    #     """
    #     R = self.Exp(r)  # (3, 3)
    #     c2w = torch.cat([R, t.unsqueeze(1)], dim=1)  # (3, 4)
    #     c2w = self.convert3x4_4x4(c2w)  # (4, 4)
    #     return c2w

    # def run_pose_network(self, cam_id=0):
    #     r = torch.squeeze(self.r[cam_id])  # (3, ) axis-angle
    #     t = torch.squeeze(self.t[cam_id])  # (3, )

    #     c2w = self.make_c2w(r, t)  # (4, 4)
    #     return c2w

    # def run_fxfy_network(self, h, w):
    #     fxfy = torch.stack([self.fx**2 * w, self.fy**2 * h])
    #     return fxfy

    # def convert3x4_4x4(self, input):
    #     """
    #     :param input:  (N, 3, 4) or (3, 4) torch or np
    #     :return:       (N, 4, 4) or (4, 4) torch or np
    #     """
    #     if torch.is_tensor(input):
    #         if len(input.shape) == 3:
    #             output = torch.cat([input, torch.zeros_like(
    #                 input[:, 0:1])], dim=1)  # (N, 4, 4)
    #             output[:, 3, 3] = 1.0
    #         else:
    #             output = torch.cat([input, torch.tensor(
    #                 [[0, 0, 0, 1]], dtype=input.dtype, device=input.device)], dim=0)  # (4, 4)
    #     else:
    #         if len(input.shape) == 3:
    #             output = np.concatenate(
    #                 [input, np.zeros_like(input[:, 0:1])], axis=1)  # (N, 4, 4)
    #             output[:, 3, 3] = 1.0
    #         else:
    #             output = np.concatenate(
    #                 [input, np.array([[0, 0, 0, 1]], dtype=input.dtype)], axis=0)  # (4, 4)
    #             output[3, 3] = 1.0
    #     return output

    # optimizer utils
    def get_params(self, lr, lr_net, svd):
        if (svd == "static"):
            params = [
                {'params': self.encoder_s.parameters(), 'lr': lr},
                {'params': self.encoder_dir_s.parameters(), 'lr': lr},
                {'params': self.sigma_s_net.parameters(), 'lr': lr_net},
                {'params': self.color_s_net.parameters(), 'lr': lr_net},
                {'params': self.r, 'lr': lr},
                {'params': self.t, 'lr': lr},
                {'params': self.fx, 'lr': lr},
                {'params': self.fy, 'lr': lr},
            ]
            if self.bg_radius > 0:
                params.append(
                    {'params': self.encoder_bg.parameters(), 'lr': lr})
                params.append(
                    {'params': self.bg_s_net.parameters(), 'lr': lr_net})
        elif (svd == "dynamic"):
            params = [
                {'params': self.encoder_d.parameters(), 'lr': lr},
                {'params': self.encoder_dir_d.parameters(), 'lr': lr},
                # {'params': self.sigma_s_net.parameters(), 'lr': lr_net},
                # {'params': self.color_s_net.parameters(), 'lr': lr_net},
                {'params': self.encoder_deform.parameters(), 'lr': lr_net},
                {'params': self.encoder_time.parameters(), 'lr': lr_net},
                {'params': self.sigma_d_net.parameters(), 'lr': lr_net},
                {'params': self.color_d_net.parameters(), 'lr': lr_net},
                {'params': self.deform_d_net.parameters(), 'lr': lr_net},
                {'params': self.r, 'lr': lr},
                {'params': self.t, 'lr': lr},
                {'params': self.fx, 'lr': lr},
                {'params': self.fy, 'lr': lr},
                # {'params': self.blend_net.parameters(), 'lr': lr_net},
                # {'params': self.sf_net.parameters(), 'lr': lr_net},
            ]
            if self.bg_radius > 0:
                params.append(
                    {'params': self.encoder_bg.parameters(), 'lr': lr})
                params.append(
                    {'params': self.bg_s_net.parameters(), 'lr': lr_net})
        elif (svd == "camera"):
            params = [
                {'params': self.r, 'lr': lr},
                {'params': self.t, 'lr': lr},
                {'params': self.fx, 'lr': lr},
                {'params': self.fy, 'lr': lr},
            ]
        elif (svd == "all"):
            params = [
                {'params': self.encoder_s.parameters(), 'lr': lr},
                {'params': self.encoder_dir_s.parameters(), 'lr': lr},
                {'params': self.encoder_d.parameters(), 'lr': lr},
                {'params': self.encoder_dir_d.parameters(), 'lr': lr},
                {'params': self.encoder_deform.parameters(), 'lr': lr},
                {'params': self.encoder_time.parameters(), 'lr': lr},
                {'params': self.sigma_s_net.parameters(), 'lr': lr_net},
                {'params': self.color_s_net.parameters(), 'lr': lr_net},
                {'params': self.sigma_d_net.parameters(), 'lr': lr_net},
                {'params': self.color_d_net.parameters(), 'lr': lr_net},
                {'params': self.deform_d_net.parameters(), 'lr': lr_net},
                # {'params': self.blend_net.parameters(), 'lr': lr_net},
                # {'params': self.sf_net.parameters(), 'lr': lr_net},
            ]
            if self.bg_radius > 0:
                params.append(
                    {'params': self.encoder_bg.parameters(), 'lr': lr})
                params.append(
                    {'params': self.bg_s_net.parameters(), 'lr': lr_net})

        else:
            raise Exception("Run NeRF in either `static` or `dynamic` mode")
        return params
