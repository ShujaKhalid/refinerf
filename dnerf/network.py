from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="tiledgrid",
                 encoding_dir="sphere_harmonics",
                 encoding_time="frequency",
                 encoding_deform="frequency",  # "hashgrid" seems worse
                 encoding_bg="hashgrid",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=12,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 # a deeper MLP is very necessary for performance.
                 num_layers_deform=3,
                 hidden_dim_deform=128,
                 bound=1,
                 **kwargs,
                 ):
        super().__init__(bound, **kwargs)

        # ==================
        # STATIC
        # ==================

        # deformation network ============================================
        # self.num_layers_deform = num_layers_deform
        # self.hidden_dim_deform = hidden_dim_deform
        # self.encoder_deform, self.in_dim_deform = get_encoder(
        #     encoding_deform, multires=10)
        # self.encoder_time, self.in_dim_time = get_encoder(
        #     encoding_time, input_dim=1, multires=6)

        # print("self.in_dim_deform: {}".format(self.in_dim_deform))
        # print("self.in_dim_time: {}".format(self.in_dim_time))

        # deform_s_net = []
        # for l in range(num_layers_deform):
        #     if l == 0:
        #         in_dim = self.in_dim_deform + self.in_dim_time  # grid dim + time
        #     else:
        #         in_dim = hidden_dim_deform

        #     if l == num_layers_deform - 1:
        #         out_dim = 3  # deformation for xyz
        #     else:
        #         out_dim = hidden_dim_deform

        #     deform_s_net.append(nn.Linear(in_dim, out_dim, bias=False))

        # self.deform_s_net = nn.ModuleList(deform_s_net)

        # sigma network ============================================
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.encoder_s, self.in_dim_s = get_encoder(
            encoding, desired_resolution=2048 * bound)
        self.encoder_d, self.in_dim_d = get_encoder(
            encoding, desired_resolution=2048 * bound)

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
        self.encoder_dir_s, self.in_dim_dir_s = get_encoder(encoding_dir)
        self.encoder_dir_d, self.in_dim_dir_d = get_encoder(encoding_dir)

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

        # # background network ============================================
        # if self.bg_radius > 0:
        #     self.num_layers_bg = num_layers_bg
        #     self.hidden_dim_bg = hidden_dim_bg
        #     self.encoder_bg, self.in_dim_bg = get_encoder(
        #         encoding_bg, input_dim=2, num_levels=4, log2_hashmap_size=19, desired_resolution=2048)  # much smaller hashgrid

        #     # print("self.in_dim_bg: {}".format(self.in_dim_bg))
        #     # print("self.in_dim_dir: {}".format(self.in_dim_dir))

        #     bg_s_net = []
        #     for l in range(num_layers_bg):
        #         if l == 0:
        #             in_dim = self.in_dim_bg + self.in_dim_dir
        #         else:
        #             in_dim = hidden_dim_bg

        #         if l == num_layers_bg - 1:
        #             out_dim = 3  # 3 rgb
        #         else:
        #             out_dim = hidden_dim_bg

        #         bg_s_net.append(nn.Linear(in_dim, out_dim, bias=False))

        #     self.bg_s_net = nn.ModuleList(bg_s_net)
        # else:
        #     self.bg_s_net = None

        # ==================
        # DYNAMIC
        # ==================

        # Added for dynamic NeRF ============================================
        print("\nINITIALIZING DYNAMIC MODEL!!!\n")
        self.input_ch = 3
        self.D = 8  # FIXME: used to be 8!
        self.W = 256  # FIXME: used to be 256!
        self.skips = [4]
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, self.W)] + [nn.Linear(self.W, self.W) if i not in self.skips else nn.Linear(self.W + self.input_ch, self.W) for i in range(self.D-1)])

        # deformation network ============================================
        self.num_layers_deform = num_layers_deform
        self.hidden_dim_deform = hidden_dim_deform
        self.encoder_deform, self.in_dim_deform = get_encoder(
            encoding_deform, multires=10)  # FIXME: used to be 10
        self.encoder_time, self.in_dim_time = get_encoder(
            encoding_time, input_dim=1, multires=12)  # FIXME: used to be 6

        print("\nin_dim_deform: {}".format(self.in_dim_deform))
        print("in_dim_time: {}".format(self.in_dim_time))
        print("in_dim_dir_d: {}".format(self.in_dim_dir_d))
        print("geo_feat_dim: {}".format(self.geo_feat_dim))

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

        # # background network ============================================
        # if self.bg_radius > 0:
        #     self.num_layers_bg = num_layers_bg
        #     self.hidden_dim_bg = hidden_dim_bg
        #     self.encoder_bg, self.in_dim_bg = get_encoder(
        #         encoding_bg, input_dim=2, num_levels=4, log2_hashmap_size=19, desired_resolution=2048)  # much smaller hashgrid

        #     bg_d_net = []
        #     for l in range(num_layers_bg):
        #         if l == 0:
        #             in_dim = self.in_dim_bg + self.in_dim_dir
        #         else:
        #             in_dim = hidden_dim_bg

        #         if l == num_layers_bg - 1:
        #             out_dim = 3  # 3 rgb
        #         else:
        #             out_dim = hidden_dim_bg

        #         bg_d_net.append(nn.Linear(in_dim, out_dim, bias=False))

        #     self.bg_d_net = nn.ModuleList(bg_d_net)
        # else:
        #     self.bg_d_net = None

        self.sf_net = nn.Linear(self.W, 6)
        self.blend_net = nn.Linear(self.W, 1)

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

        # print("\nt: {}".format(t))
        # print("enc_ori_x.shape: {}".format(enc_ori_x.shape))
        # print("enc_t.shape: {}\n".format(enc_t.shape))

        # # TODO: Added -> confirm
        # if (len(x.shape) == 3):
        #     enc_t = torch.unsqueeze(
        #         enc_t, -1).repeat(1, 1, enc_t.shape[1])
        #     deform = torch.cat([enc_ori_x, enc_t], dim=-1)  # [N, C + C']
        # else:
        deform = torch.cat([enc_ori_x, enc_t], dim=1)  # [N, C + C']
        # print("x.shape: {}".format(x.shape))
        # print("enc_t.shape: {}".format(enc_t.shape))
        # print("enc_ori_x.shape: {}".format(enc_ori_x.shape))
        # print("deform.shape: {}".format(deform.shape))

        for l in range(self.num_layers_deform):
            deform = self.deform_d_net[l](deform)
            if l != self.num_layers_deform - 1:
                deform = F.relu(deform, inplace=True)

        x_def = x + deform  # FIXME: x + deform

        # sigma
        x_def = self.encoder_d(x_def, bound=self.bound)
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

        # dynamic
        input_pts, _ = x, d
        h = input_pts
        for i, _ in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        sf = torch.tanh(self.sf_net(h))
        blending = torch.sigmoid(self.blend_net(h))

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
        x = self.encoder_d(x, bound=self.bound)
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

    # optimizer utils
    def get_params(self, lr, lr_net, svd):
        if (svd == "static"):
            params = [
                {'params': self.encoder_s.parameters(), 'lr': lr},
                {'params': self.encoder_dir_s.parameters(), 'lr': lr},
                {'params': self.sigma_s_net.parameters(), 'lr': lr_net},
                {'params': self.color_s_net.parameters(), 'lr': lr_net},
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
                {'params': self.encoder_deform.parameters(), 'lr': lr},
                {'params': self.encoder_time.parameters(), 'lr': lr},
                {'params': self.sigma_d_net.parameters(), 'lr': lr_net},
                {'params': self.color_d_net.parameters(), 'lr': lr_net},
                {'params': self.deform_d_net.parameters(), 'lr': lr_net},
                {'params': self.blend_net.parameters(), 'lr': lr_net},
                {'params': self.sf_net.parameters(), 'lr': lr_net},
            ]
            if self.bg_radius > 0:
                params.append(
                    {'params': self.encoder_bg.parameters(), 'lr': lr})
                params.append(
                    {'params': self.bg_s_net.parameters(), 'lr': lr_net})
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
                {'params': self.blend_net.parameters(), 'lr': lr_net},
                {'params': self.sf_net.parameters(), 'lr': lr_net},
            ]
            if self.bg_radius > 0:
                params.append(
                    {'params': self.encoder_bg.parameters(), 'lr': lr})
                params.append(
                    {'params': self.bg_s_net.parameters(), 'lr': lr_net})
        else:
            raise Exception("Run NeRF in either `static` or `dynamic` mode")
        return params
