from nerf.utils import *
from nerf.utils import Trainer as _Trainer


class Trainer(_Trainer):
    def __init__(self,
                 name,  # name of this experiment
                 opt,  # extra conf
                 model,  # network
                 criterion=None,  # loss function, if None, assume inline implementation in train_step
                 optimizer=None,  # optimizer
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

        self.optimizer_fn = optimizer
        self.lr_scheduler_fn = lr_scheduler

        super().__init__(name, opt, model, criterion, optimizer, ema_decay, lr_scheduler, metrics, local_rank, world_size, device, mute, fp16, eval_interval,
                         max_keep_ckpt, workspace, best_mode, use_loss_as_metric, report_metric_at_train, use_checkpoint, use_tensorboardX, scheduler_update_every_step)

    # ------------------------------

    def train_step(self, data):

        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        time = data['time']  # [B, 1]

        print("data.keys(): {}".format(data.keys()))
        print("data[masks]: {}".format(data["masks"].shape))

        # TODO:
        # Get masks here and

        # if there is no gt image, we train with CLIP loss.
        if 'images' not in data:

            B, N = rays_o.shape[:2]
            H, W = data['H'], data['W']

            # currently fix white bg, MUST force all rays!
            outputs = self.model.render(rays_o, rays_d, time, staged=False,
                                        bg_color=None, perturb=True, force_all_rays=True, **vars(self.opt))
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

        outputs = self.model.render(rays_o, rays_d, time, staged=False,
                                    bg_color=bg_color, perturb=True, force_all_rays=False, **vars(self.opt))

        pred_rgb = outputs['image']

        # TODO: Get outputs here
        depth = outputs['depth']
        image = outputs['image']
        blend = outputs['blend']
        sigmas_s = outputs['sigmas_s']
        rgbs_s = outputs['rgbs_s']
        sigmas_d = outputs['sigmas_d']
        rgbs_d = outputs['rgbs_d']
        deform_s = outputs['deform_s']
        deform_d = outputs['deform_d']
        sf = outputs['sf']

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

        # deform regularization
        if 'deform' in outputs and outputs['deform'] is not None:
            loss = loss + 1e-3 * outputs['deform'].abs().mean()

        return pred_rgb, gt_rgb, loss

    def eval_step(self, data):

        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        time = data['time']  # [B, 1]
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
            rays_o, rays_d, time, staged=True, bg_color=bg_color, perturb=False, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)

        loss = self.criterion(pred_rgb, gt_rgb).mean()

        return pred_rgb, pred_depth, gt_rgb, loss

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, bg_color=None, perturb=False):

        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        time = data['time']  # [B, 1]
        H, W = data['H'], data['W']

        if bg_color is not None:
            bg_color = bg_color.to(self.device)

        outputs = self.model.render(
            rays_o, rays_d, time, staged=True, bg_color=bg_color, perturb=perturb, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(-1, H, W, 3)
        pred_depth = outputs['depth'].reshape(-1, H, W)

        return pred_rgb, pred_depth

    # [GUI] test on a single image
    def test_gui(self, pose, intrinsics, W, H, time=0, bg_color=None, spp=1, downscale=1):

        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)

        rays = get_rays(pose, intrinsics, rH, rW, -1)

        data = {
            # from scalar to [1, 1] tensor.
            'time': torch.FloatTensor([[time]]).to(self.device),
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

    def save_mesh(self, time, save_path=None, resolution=256, threshold=10):
        # time: scalar in [0, 1]
        time = torch.FloatTensor([[time]]).to(self.device)

        if save_path is None:
            save_path = os.path.join(
                self.workspace, 'meshes', f'{self.name}_{self.epoch}.ply')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        def query_func(pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    sigma = self.model.density(
                        pts.to(self.device), time)['sigma']
            return sigma

        vertices, triangles = extract_geometry(
            self.model.aabb_infer[:3], self.model.aabb_infer[3:], resolution=resolution, threshold=threshold, query_func=query_func)

        # important, process=True leads to seg fault...
        mesh = trimesh.Trimesh(vertices, triangles, process=False)
        mesh.export(save_path)

        self.log(f"==> Finished saving mesh.")


def raw2outputs(raw_s,
                raw_d,
                blending,
                z_vals,
                rays_d,
                raw_noise_std):
    """Transforms model's predictions to semantically meaningful values.

    Args:
      raw_d: [num_rays, num_samples along ray, 4]. Prediction from Dynamic model.
      raw_s: [num_rays, num_samples along ray, 4]. Prediction from Static model.
      z_vals: [num_rays, num_samples along ray]. Integration time.
      rays_d: [num_rays, 3]. Direction of each ray.

    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
      disp_map: [num_rays]. Disparity map. Inverse of depth map.
      acc_map: [num_rays]. Sum of weights along each ray.
      weights: [num_rays, num_samples]. Weights assigned to each sampled color.
      depth_map: [num_rays]. Estimated distance to object.
    """
    # Function for computing density from model prediction. This value is
    # strictly between [0, 1].
    def raw2alpha(raw, dists, act_fn=F.relu): return 1.0 - \
        torch.exp(-act_fn(raw) * dists)

    # Compute 'distance' (in time) between each integration time along a ray.
    dists = z_vals[..., 1:] - z_vals[..., :-1]

    # The 'distance' from the last integration time is infinity.
    dists = torch.cat(
        [dists, torch.Tensor([1e10]).expand(dists[..., :1].shape).cuda()],
        -1)  # [N_rays, N_samples]

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # Extract RGB of each sample position along each ray.
    rgb_d = torch.sigmoid(raw_d[..., :3])  # [N_rays, N_samples, 3]
    rgb_s = torch.sigmoid(raw_s[..., :3])  # [N_rays, N_samples, 3]

    # Add noise to model's predictions for density. Can be used to
    # regularize network during training (prevents floater artifacts).
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw_d[..., 3].shape) * raw_noise_std

    # Predict density of each sample along each ray. Higher values imply
    # higher likelihood of being absorbed at this point.
    alpha_d = raw2alpha(raw_d[..., 3] + noise,
                        dists).cuda()  # [N_rays, N_samples]
    alpha_s = raw2alpha(raw_s[..., 3] + noise,
                        dists).cuda()  # [N_rays, N_samples]
    alphas = 1. - (1. - alpha_s) * (1. - alpha_d)  # [N_rays, N_samples]

    T_d = torch.cumprod(torch.cat(
        [torch.ones((alpha_d.shape[0], 1)).cuda(), 1. - alpha_d + 1e-10], -1), -1)[:, :-1]
    T_s = torch.cumprod(torch.cat(
        [torch.ones((alpha_s.shape[0], 1)).cuda(), 1. - alpha_s + 1e-10], -1), -1)[:, :-1]

    blending = torch.squeeze(blending)
    T_full = torch.cumprod(torch.cat([torch.ones((alpha_d.shape[0], 1)).cuda(), (
        1. - alpha_d * blending).cuda() * (1. - alpha_s * (1. - blending)).cuda() + 1e-10], -1), -1)[:, :-1]
    # T_full = torch.cumprod(torch.cat([torch.ones((alpha_d.shape[0], 1)), torch.pow(1. - alpha_d + 1e-10, blending) * torch.pow(1. - alpha_s + 1e-10, 1. - blending)], -1), -1)[:, :-1]
    # T_full = torch.cumprod(torch.cat([torch.ones((alpha_d.shape[0], 1)), (1. - alpha_d) * (1. - alpha_s) + 1e-10], -1), -1)[:, :-1]

    # Compute weight for RGB of each sample along each ray.  A cumprod() is
    # used to express the idea of the ray not having reflected up to this
    # sample yet.
    weights_d = alpha_d * T_d
    weights_s = alpha_s * T_s
    weights_full = (alpha_d * blending + alpha_s * (1. - blending)) * T_full
    # weights_full = alphas * T_full

    # Computed weighted color of each sample along each ray.
    rgb_map_d = torch.sum(weights_d[..., None] * rgb_d, -2)
    rgb_map_s = torch.sum(weights_s[..., None] * rgb_s, -2)
    rgb_map_full = torch.sum(
        (T_full * alpha_d * blending)[..., None] * rgb_d +
        (T_full * alpha_s * (1. - blending))[..., None] * rgb_s, -2)

    # Estimated depth map is expected distance.
    depth_map_d = torch.sum(weights_d * z_vals, -1)
    depth_map_s = torch.sum(weights_s * z_vals, -1)
    depth_map_full = torch.sum(weights_full * z_vals, -1)

    # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
    acc_map_d = torch.sum(weights_d, -1)
    acc_map_s = torch.sum(weights_s, -1)
    acc_map_full = torch.sum(weights_full, -1)

    # Computed dynamicness
    dynamicness_map = torch.sum(weights_full * blending, -1)
    # dynamicness_map = 1 - T_d[..., -1]

    return rgb_map_full, depth_map_full, acc_map_full, weights_full, \
        rgb_map_s, depth_map_s, acc_map_s, weights_s, \
        rgb_map_d, depth_map_d, acc_map_d, weights_d, dynamicness_map


def raw2outputs_d(raw_d,
                  z_vals,
                  rays_d,
                  raw_noise_std):

    # Function for computing density from model prediction. This value is
    # strictly between [0, 1].
    def raw2alpha(raw, dists, act_fn=F.relu): return 1.0 - \
        torch.exp(-act_fn(raw) * dists)

    # Compute 'distance' (in time) between each integration time along a ray.
    dists = z_vals[..., 1:] - z_vals[..., :-1]

    # The 'distance' from the last integration time is infinity.
    dists = torch.cat(
        [dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)],
        -1)  # [N_rays, N_samples]

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # Extract RGB of each sample position along each ray.
    rgb_d = torch.sigmoid(raw_d[..., :3])  # [N_rays, N_samples, 3]

    # Add noise to model's predictions for density. Can be used to
    # regularize network during training (prevents floater artifacts).
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw_d[..., 3].shape) * raw_noise_std

    # Predict density of each sample along each ray. Higher values imply
    # higher likelihood of being absorbed at this point.
    alpha_d = raw2alpha(raw_d[..., 3] + noise, dists)  # [N_rays, N_samples]

    T_d = torch.cumprod(torch.cat(
        [torch.ones((alpha_d.shape[0], 1)), 1. - alpha_d + 1e-10], -1), -1)[:, :-1]
    # Compute weight for RGB of each sample along each ray.  A cumprod() is
    # used to express the idea of the ray not having reflected up to this
    # sample yet.
    weights_d = alpha_d * T_d

    # Computed weighted color of each sample along each ray.
    rgb_map_d = torch.sum(weights_d[..., None] * rgb_d, -2)

    return rgb_map_d, weights_d
