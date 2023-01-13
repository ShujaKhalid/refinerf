from dataclasses import dataclass
from nerf.utils import *
from nerf.utils import Trainer as _Trainer
from utils.run_nerf_helpers import *
from torchviz import make_dot, make_dot_from_trace
# from dnerf.network_camera import CameraNetwork # Sent in from main_dnerf


class Trainer(_Trainer):
    def __init__(self,
                 name,  # name of this experiment
                 opt,  # extra conf
                 model,  # network
                 model_fxfy,
                 model_pose,
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

        self.optimizer_fn = optimizer_model
        self.lr_scheduler_fn = lr_scheduler

        super().__init__(name, opt, model, model_fxfy, model_pose, criterion, optimizer_model, optimizer_fxfy, optimizer_pose, ema_decay, lr_scheduler, metrics, local_rank, world_size, device, mute, fp16, eval_interval,
                         max_keep_ckpt, workspace, best_mode, use_loss_as_metric, report_metric_at_train, use_checkpoint, use_tensorboardX, scheduler_update_every_step)

    # ------------------------------

    def render_prereqs(self, data):

        results = data
        B = 1
        self.PRED_INTRINSICS = data['PRED_INTRINSICS']
        self.PRED_EXTRINSICS = data['PRED_EXTRINSICS']
        self.FLOW_FLAG = data['FLOW_FLAG']
        self.TRAIN_FLAG = data['TRAIN_FLAG']
        self.DYNAMIC_ITER = data['dynamic_iter']
        self.DYNAMIC_ITERS = data['dynamic_iters']
        self.poses = data['poses']  # COLMAP APPROXIMATION
        self.intrinsics = data['intrinsics']  # COLMAP_APPROXIMATION
        self.error_map = data['error_map']
        self.num_rays = data['num_rays']
        self.masks = data['masks']
        self.masks_val = data['masks_val']
        self.all_masks = data['all_masks']
        self.all_masks_val = data['all_masks_val']
        self.times = data['time']
        self.H = data['H']
        self.W = data['W']
        self.index = data['index']
        self.grid = data['grid']
        self.images = data['images']
        self.disp = data['disp']

        if (self.PRED_INTRINSICS or self.PRED_EXTRINSICS):
            # print("\n\nPREDICTING POSES!\n\n")
            poses_gt = self.poses
            intrinsics_gt = self.intrinsics
            # fxfy_pred, poses_pred = self.model_camera(self.index)
            # fxfy_pred, poses_pred = self.model(None, None, None, svd="camera")

            # print("fxfy: {}\nposes: {}".format(fxfy_pred, poses_pred))

            # assignments
            INTRINSICS_FLAG = self.PRED_INTRINSICS
            EXTRINSICS_FLAG = self.PRED_EXTRINSICS
            if (INTRINSICS_FLAG):
                fxfy_pred = self.model_fxfy(fxfy_gt=self.intrinsics)
                self.intrinsics = fxfy_pred
            if (EXTRINSICS_FLAG):
                poses_pred = self.model_pose(self.index, poses_gt)
                self.poses = poses_pred
            # self.poses[0, 0, 0] = poses_pred[0, 0]
            # self.poses = torch.unsqueeze(
            #     poses_pred, 0)  # [B, 4, 4]
            # poses_pred = torch.unsqueeze(
            #     poses_pred, 0).to(self.device)  # [B, 4, 4]

            # print()
            # print("\nfxfy_actual: {}\nposes_actual:\n{}".format(
            #     intrinsics_gt, poses_gt))
            # # print("fxfy_pred: {} - poses_pred: {}".format(fxfy_pred, poses_pred))
            # # print("\nfxfy_actual.shape: {}\nposes_actual.shape: {}\n".format(
            # #     intrinsics_gt.shape, poses_gt.shape))
            # print("\self.intrinsics: {}\nposes_new:\n{}".format(
            #     self.intrinsics, poses_pred))
            # # print(
            # #     "\self.intrinsics.shape: {}\nposes_new.shape: {}\n".format(self.intrinsics, poses_pred.shape))
            # print()
            # print("\nfxfy_new: {}\nposes_new:\n{}".format(
            #     fxfy_pred, poses_pred))
            # # print(
            # #     "\nfxfy_new.shape: {}\nposes_new.shape: {}\n".format(fxfy_pred.shape, poses_pred.shape))
            # print()
            # self.intrinsics = [416.44504027, 429.45316301, 240, 125]
            #self.intrinsics = [1270, 640, 240, 125]

            # print("\n\nfxfy_actual: {}".format(intrinsics_gt))
            # print("self.intrinsics: {}".format(self.intrinsics))
            # print()

            # print("\n\nextrinsics_actual: {}".format(poses_gt))
            # print("self.poses: {}".format(self.poses))
            # print()

        # bypass rays for testing
        # rays = {}
        # rays["inds"] = torch.ones(1, 4096, dtype=torch.long).to(self.device)
        # rays["inds_s"] = torch.ones(117280, dtype=torch.int16).to(self.device)
        # rays["inds_d"] = torch.ones(12320, dtype=torch.int16).to(self.device)
        # rays["rays_o"] = torch.ones(
        #     1, 4096, 3, dtype=torch.float32).to(self.device)
        # rays["rays_d"] = torch.ones(
        #     1, 4096, 3, dtype=torch.float32).to(self.device)

        if self.TRAIN_FLAG:
            rays = get_rays(self.poses, self.intrinsics, self.H,
                            self.W, self.masks, self.num_rays, self.error_map, self.DYNAMIC_ITER, self.DYNAMIC_ITERS)  # sk_debug - added masks
        else:
            rays = get_rays(self.poses, self.intrinsics, self.H,
                            self.W, self.masks_val, self.num_rays, self.error_map, self.DYNAMIC_ITER, self.DYNAMIC_ITERS)  # sk_debug - added masks

        # self.DYNAMIC_ITER += 1
        # print("rays['inds'].shape: {}".format(rays["inds"].shape))
        # print("rays['inds_s'].shape: {}".format(rays["inds_s"].shape))
        # print("rays['inds_d'].shape: {}".format(rays["inds_d"].shape))
        # print("rays['rays_o'].shape: {}".format(rays["rays_o"].shape))
        # print("rays['rays_d'].shape: {}".format(rays["rays_d"].shape))
        # print("rays['inds'].shape: {}".format(rays["inds"]))
        # print("rays['inds_s'].shape: {}".format(rays["inds_s"]))
        # print("rays['inds_d'].shape: {}".format(rays["inds_d"]))
        # print("rays['rays_o'].shape: {}".format(rays["rays_o"]))
        # print("rays['rays_d'].shape: {}".format(rays["rays_d"]))

        if ("inds_s" in rays and "inds_d" in rays):
            self.inds_s = rays["inds_s"]
            self.inds_d = rays["inds_d"]
        else:
            self.inds_s = 0
            self.inds_d = 0

        results['inds_s'] = self.inds_s
        results['inds_d'] = self.inds_d

        indices = rays["inds"] if self.TRAIN_FLAG else -1

        if (self.FLOW_FLAG):
            grid = self.grid[:, indices, :]
        else:
            grid = self.grid

        results['rays_o'] = rays['rays_o']
        results['rays_d'] = rays['rays_d']

        if self.images is not None:
            # [B, H, W, 3/4]
            # print("index: {}".format(index))
            i = self.index[0]
            images_b = self.images[i-1].to(self.device) if i - \
                1 > 0 else self.images[i].to(self.device)
            images_f = self.images[i+1].to(self.device) if i+1 < len(
                self.images) else self.images[self.index].to(self.device)   # [B, H, W, 3/4]
            images = self.images[self.index].to(self.device)  # [B, H, W, 3/4]

            if self.TRAIN_FLAG:
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
                i = self.index[0]
                # [B, H, W, 3/4]
                # print(self.masks[:, :, :, index].shape)
                # print(self.disp[:, :, index].shape)
                masks = self.all_masks[:, :, :, i]
                disp = torch.Tensor(self.disp[:, :, i]).to(
                    self.device)  # [B, H, W, 3/4]
                # print("masks.shape: {}".format(masks.shape))
                # print("disp.shape: {}".format(disp.shape))
                # print("images.shape: {}".format(images.shape))
                # print("rays['inds'].shape: {}".format(rays['inds'].shape))
                # print("disp.view(B, -1, 1).shape: {}".format(disp.view(1, -1, 1).shape))
                if self.TRAIN_FLAG:
                    masks = torch.gather(masks.view(
                        B, -1, 3), 1, torch.stack(3 * [rays['inds']], -1))  # [B, N, 3/4]
                    disp = torch.gather(disp.view(
                        B, -1, 1), 1, torch.stack(1 * [rays['inds']], -1))  # [B, N, 3/4]
                results['disp'] = disp
                results['masks'] = masks

        # need inds to update error_map
        results['index'] = self.index
        results['grid'] = grid
        if self.error_map is not None:
            results['inds_coarse'] = rays['inds_coarse']

        return results

    def train_step(self, data):

        data = self.render_prereqs(data)

        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        time = data['time']  # [B, 1]

        # print("data.keys(): {}".format(data.keys()))
        # print("data[masks]: {}".format(data["masks"].shape))

        # if there is no gt image, we train with CLIP loss.
        if 'images' not in data:

            B, N = rays_o.shape[:2]
            H, W = data['H'], data['W']

            # currently fix white bg, MUST force all rays!
            outputs = self.model.render(rays_o, rays_d, time, staged=False,
                                        bg_color=None, perturb=True, force_all_rays=True, **vars(self.opt))
            pred_rgb = outputs['image'].reshape(
                B, H, W, 3).permute(0, 3, 1, 2).contiguous()
            pred_rgb_b = outputs['image_b'].reshape(
                B, H, W, 3).permute(0, 3, 1, 2).contiguous()
            pred_rgb_f = outputs['image_f'].reshape(
                B, H, W, 3).permute(0, 3, 1, 2).contiguous()

            # [debug] uncomment to plot the images used in train_step
            # torch_vis_2d(pred_rgb[0])

            loss = self.clip_loss(pred_rgb)

            return pred_rgb, None, loss

        images = data['images']  # [B, N, 3/4]
        images_b = data['images_b']  # [B, N, 3/4]
        images_f = data['images_f']  # [B, N, 3/4]

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
            gt_rgb_b = images_b
            gt_rgb_f = images_f

        self.opt.inds_s = data['inds_s']
        self.opt.inds_d = data['inds_d']

        ret = self.model.render(rays_o, rays_d, time, staged=False,
                                bg_color=bg_color, perturb=True, force_all_rays=False, **vars(self.opt))

        pred_rgb = ret['image']

        # TODO: Get outputs here
        # depth_s = outputs['depth_s']
        # depth_d = outputs['depth_d']
        # image_s = outputs['image']
        # blend = outputs['blending']
        # sigmas_s = outputs['sigmas_s']
        # sigmas_d = outputs['sigmas_d']
        # pts_f = outputs['raw_pts_f']
        # pts_b = outputs['raw_pts_b']
        # rgbs_s = outputs['rgbs_s']
        # rgbs_d = outputs['rgbs_d']
        # deform_d = outputs['deform_d']
        # sceneflow_f = outputs['sceneflow_f']
        # sceneflow_b = outputs['sceneflow_b']
        # weights_sum_s = outputs['weights_sum_s']
        # weights_sum_d = outputs['weights_sum_d']
        # sceneflow_b_f = outputs['sceneflow_b_f']
        # rgb_map_d_b = outputs['rgb_map_d_b']
        # acc_map_d_b = outputs['acc_map_d_b']
        # rgbs_d_f = outputs['rgb_map_d_f']
        # sceneflow_f_b = outputs['sceneflow_f_b']
        # acc_map_d_f = outputs['acc_map_d_f']
        # raw_pts_b_b = outputs['raw_pts_b_b']
        # raw_pts_f_f = outputs['raw_pts_f_f']
        # image_d_b_b = outputs['rgb_map_d_b_b']
        # rgb_map_d_f_f = outputs['rgb_map_d_f_f']
        # FLOW_FLAG = False
        if (data["FLOW_FLAG"]):  # FIXME
            # Calculate losses here
            chain_5frames = True  # FIXME: Make conditionally appropriate
            loss = 0
            loss_dict = {}
            decay_iteration = 25  # FIXME
            num_img = data['num_img']
            img_i = data['index']
            batch_invdepth = data['disp'].cuda()
            poses = data['all_poses']
            # FIXME - 0 index probs
            batch_mask = data['masks'].cuda()
            batch_grid = data['grids'].cuda()
            # print("batch_mask.shape: {}".format(batch_mask.shape))
            Temp = 1. / (10 ** (1 // (decay_iteration * 1000)))  # FIXME
            focal = data['intrinsics'][0]  # FIXME
            H, W = data['H'], data['W']
            args = {
                'dynamic_loss_lambda': 1.0,
                'static_loss_lambda': 1.0,
                'full_loss_lambda': 3.0,
                'depth_loss_lambda': 0.04,
                'order_loss_lambda': 0.1,
                'flow_loss_lambda': 0.02,
                'slow_loss_lambda': 0.01,
                'smooth_loss_lambda': 0.1,
                'consistency_loss_lambda': 1.0,
                'mask_loss_lambda': 0.1,
                'sparse_loss_lambda': 0.001,
                'deform_loss_lambda': 1.0
            }

            # # TODO: Combined loss
            # if ("rgb_map_full" in ret):
            #     img_loss = img2mse(ret['rgb_map_full'], gt_rgb)
            #     psnr = mse2psnr(img_loss)
            #     loss_dict['psnr'] = psnr
            #     loss_dict['img_loss'] = img_loss
            #     loss += args['full_loss_lambda'] * loss_dict['img_loss']

            # # [Include] Deformation Loss
            # if ("deform" in ret):
            #     loss_dict['deform_loss'] = ret['deform'].abs().mean()
            #     loss += args['deform_loss_lambda'] * loss_dict['deform_loss']

            # [Include] Compute MSE loss between rgb_s and true RGB.
            if ("rgb_map_s" in ret):
                # print("ret['rgb_map_s']: {}".format(ret['rgb_map_s'].shape))
                # print("ret['gt_rgb']: {}".format(gt_rgb.shape))
                img_s_loss = img2mse(
                    ret['rgb_map_s'], gt_rgb)
                psnr_s = mse2psnr(img_s_loss)
                loss_dict['psnr_s'] = psnr_s
                loss_dict['img_s_loss'] = img_s_loss
                loss += args['static_loss_lambda'] * loss_dict['img_s_loss']

            # [Include] Compute MSE loss between rgb_d and true RGB.
            if ("rgb_map_d" in ret):
                # print("ret['rgb_map_d']: {}".format(ret['rgb_map_d'].shape))
                # print("ret['gt_rgb']: {}".format(gt_rgb.shape))
                img_d_loss = img2mse(
                    ret['rgb_map_d'], gt_rgb)
                psnr_d = mse2psnr(img_d_loss)
                loss_dict['psnr_d'] = psnr_d
                loss_dict['img_d_loss'] = img_d_loss
                loss += args['dynamic_loss_lambda'] * loss_dict['img_d_loss']

            # [Include] Compute MSE loss between rgb_d_f and true RGB.
            if ("rgb_map_d_f" in ret):
                img_d_f_loss = img2mse(
                    ret['rgb_map_d_f'], gt_rgb_f)
                psnr_d_f = mse2psnr(img_d_f_loss)
                loss_dict['psnr_d_f'] = psnr_d_f
                loss_dict['img_d_f_loss'] = img_d_f_loss
                loss += args['dynamic_loss_lambda'] * loss_dict['img_d_f_loss']

            # [Include] Compute MSE loss between rgb_d_b and true RGB.
            if ("rgb_map_d_b" in ret):
                img_d_b_loss = img2mse(
                    ret['rgb_map_d_b'], gt_rgb_b)
                psnr_d_b = mse2psnr(img_d_b_loss)
                loss_dict['psnr_d_b'] = psnr_d_b
                loss_dict['img_d_b_loss'] = img_d_b_loss
                loss += args['dynamic_loss_lambda'] * loss_dict['img_d_b_loss']

            # print("\nDYNAMIC_loss_dict: {}\n".format(loss_dict))

            # # Motion loss.
            # # FIXME: No idea...
            # # SOLVE: Get weights_d from composite_rays cuda function
            # # Compute EPE between induced flow and true flow (forward flow).
            # # The last frame does not have forward flow.
            # if ('weights_d' in ret and 'raw_pts_f' in ret):
            #     # print("ret['weights_d']: {}".format(ret['weights_d'].shape))
            #     # print("ret['raw_pts_f'].shape: {}".format(
            #     #     ret['raw_pts_f'].shape))
            #     if img_i < num_img - 1:
            #         pts_f = ret['raw_pts_f']
            #         weight = ret['weights_d']
            #         pose_f = poses[img_i + 1, :3, :4]
            #         induced_flow_f = induce_flow(
            #             H, W, focal, pose_f, weight, pts_f, batch_grid[img_i, :, :, :2])
            #         flow_f_loss = img2mae(
            #             induced_flow_f, batch_grid[img_i, :, :, 2:4], batch_grid[img_i, :, :, 4:5])
            #         loss_dict['flow_f_loss'] = flow_f_loss
            #         loss += args['flow_loss_lambda'] * \
            #             Temp * loss_dict['flow_f_loss']

            # # Compuate EPE between induced flow and true flow (backward flow).
            # # The first frame does not have backward flow.
            # # FIXME: No idea...
            # if ('weights_d' in ret and 'raw_pts_b' in ret):
            #     if img_i > 0:
            #         pts_b = ret['raw_pts_b']
            #         weight = ret['weights_d']
            #         pose_b = poses[img_i - 1, :3, :4]
            #         induced_flow_b = induce_flow(
            #             H, W, focal, pose_b, weight, pts_b, batch_grid[img_i, :, :, :2])
            #         flow_b_loss = img2mae(
            #             induced_flow_b, batch_grid[img_i, :, :, 5:7], batch_grid[img_i, :, :, 7:8])
            #         loss_dict['flow_b_loss'] = flow_b_loss
            #         loss += args['flow_loss_lambda'] * \
            #             Temp * loss_dict['flow_b_loss']

            # # [Include] Slow scene flow. The forward and backward sceneflow should be small.
            # if ('sceneflow_f' in ret and 'sceneflow_b' in ret):
            #     slow_loss = L1(ret['sceneflow_b']) + \
            #         L1(ret['sceneflow_f'])
            #     loss_dict['slow_loss'] = slow_loss
            #     loss += args['slow_loss_lambda'] * loss_dict['slow_loss']

            # # Smooth scene flow. The summation of the forward and backward sceneflow should be small.
            # if ('raw_pts' in ret and 'raw_pts_b' in ret and 'raw_pts_f' in ret):
            #     smooth_loss = compute_sf_smooth_loss(ret['raw_pts'],
            #                                          ret['raw_pts_f'],
            #                                          ret['raw_pts_b'],
            #                                          H, W, focal)
            #     loss_dict['smooth_loss'] = smooth_loss
            #     loss += args['smooth_loss_lambda'] * loss_dict['smooth_loss']

            # # Spatial smooth scene flow. (loss adapted from NSFF)
            # if ('raw_pts' in ret and 'raw_pts_b' in ret and 'raw_pts_f' in ret):
            #     sp_smooth_loss = compute_sf_smooth_s_loss(ret['raw_pts'], ret['raw_pts_f'], H, W, focal) \
            #         + compute_sf_smooth_s_loss(ret['raw_pts'],
            #                                    ret['raw_pts_b'], H, W, focal)
            #     loss_dict['sp_smooth_loss'] = sp_smooth_loss
            #     loss += args['smooth_loss_lambda'] * \
            #         loss_dict['sp_smooth_loss']

            # [Include] Consistency loss.
            if ('sceneflow_f' in ret and 'sceneflow_b' in ret and 'sceneflow_f_b' in ret):
                consistency_loss = L1(ret['sceneflow_f'] + ret['sceneflow_f_b']) + \
                    L1(ret['sceneflow_b'] + ret['sceneflow_b_f'])
                loss_dict['consistency_loss'] = consistency_loss
                loss += args['consistency_loss_lambda'] * \
                    loss_dict['consistency_loss']

            # FIXME: Blending has incorrect dimensions & Get dynamicness_map
            # Mask loss.
            # if ('blending' in ret and 'dynamicness_map' in ret and 'sceneflow_f_b' in ret):
            #     print("ret['blending'].shape: {}".format(
            #         ret['blending']).shape)
            #     print("batch_mask[:, 0]: {}".format(batch_mask[:, 0].shape))
            #     mask_loss = L1(ret['blending'][batch_mask[:, 0].type(torch.bool)]) + \
            #         img2mae(ret['dynamicness_map'][..., None], 1 - batch_mask)
            #     loss_dict['mask_loss'] = mask_loss
            #     if i_img < decay_iteration * 1000:
            #         loss += args['mask_loss_lambda'] * loss_dict['mask_loss']

            # [Include] Sparsity loss.
            if ('weights_d' in ret and 'weights_d_b' in ret and 'weights_d_f' in ret and 'blending' in ret):
                sparse_loss = \
                    entropy(ret['weights_d']) + \
                    entropy(ret['blending']) + \
                    entropy(ret['weights_d_b']) + \
                    entropy(ret['weights_d_f'])
                # entropy(ret['weights_d_b_b']) + \
                # entropy(ret['weights_d_f_f']) + \
                loss_dict['sparse_loss'] = sparse_loss
                loss += args['sparse_loss_lambda'] * loss_dict['sparse_loss']

            # # Depth constraint
            # # Depth in NDC space equals to negative disparity in Euclidean space.
            # if ('depth_map_d' in ret):
            #     depth_loss = compute_depth_loss(
            #         ret['depth_map_d'], -batch_invdepth)
            #     loss_dict['depth_loss'] = depth_loss
            #     loss += args['depth_loss_lambda'] * \
            #         Temp * loss_dict['depth_loss']

            # # FIXME: Order loss
            # # Not sure if the mask indices and the other indices correspond
            # if ('depth_map_s' in ret and 'depth_map_d' in ret):
            #     print("batch_mask: {}".format(batch_mask.shape))
            #     # print("batch_mask_s: {}".format(
            #     #     batch_mask[0, :ret['depth_map_s'].shape[0], 0].shape))
            #     # print("batch_mask_d: {}".format(
            #     #     batch_mask[0, -ret['depth_map_d'].shape[0]:, 0].shape))
            #     # print("batch_mask_d: {}".format(
            #     #     ret['depth_map_s'][batch_mask[0, :ret['depth_map_s'].shape[0], 0].type(torch.bool)].shape))
            #     # print("batch_mask_d: {}".format(
            #     #     ret['depth_map_d'][batch_mask[0, -ret['depth_map_d'].shape[0]:, 0].type(torch.bool)].shape))
            #     # print("ret['depth_map_d']: {}".format(
            #     #     ret['depth_map_d'].shape))
            #     # print("ret['depth_map_s']: {}".format(
            #     #     ret['depth_map_s'].shape))
            #     order_loss = torch.mean(torch.square(ret['depth_map_s'][batch_mask.type(torch.bool)] -
            #                                          ret['depth_map_d'][batch_mask.type(torch.bool)]))
            #     loss_dict['order_loss'] = order_loss
            #     loss += args['order_loss_lambda'] * loss_dict['order_loss']

            # # TODO: FIX
            # sf_smooth_loss = compute_sf_smooth_loss(ret['raw_pts_b'],
            #                                         ret['raw_pts'],
            #                                         ret['raw_pts_b_b'],
            #                                         H, W, focal) + \
            #     compute_sf_smooth_loss(ret['raw_pts_f'],
            #                            ret['raw_pts_f_f'],
            #                            ret['raw_pts'],
            #                            H, W, focal)
            # loss_dict['sf_smooth_loss'] = sf_smooth_loss
            # loss += args['smooth_loss_lambda'] * loss_dict['sf_smooth_loss']

            # [Include] Chain loss
            if chain_5frames:
                if ('rgb_map_d_b_b' in ret and 'rgb_map_d_f_f' in ret):
                    img_d_b_b_loss = img2mse(
                        ret['rgb_map_d_b_b'], gt_rgb[:, :ret['rgb_map_d_b_b'].shape[0], :])
                    loss_dict['img_d_b_b_loss'] = img_d_b_b_loss
                    loss += args['dynamic_loss_lambda'] * \
                        loss_dict['img_d_b_b_loss']

                    img_d_f_f_loss = img2mse(
                        ret['rgb_map_d_f_f'], gt_rgb[:, :ret['rgb_map_d_f_f'].shape[0], :])
                    loss_dict['img_d_f_f_loss'] = img_d_f_f_loss
                    loss += args['dynamic_loss_lambda'] * \
                        loss_dict['img_d_f_f_loss']

            # print("\n{}\n".format(loss_dict))

            # Write the losses to tensorboard
            for key in loss_dict:
                val = loss_dict[key]
                if ("psnr" in key):
                    self.writer.add_scalar(
                        "psnr/"+key, val, self.global_step)
                else:
                    self.writer.add_scalar(
                        "loss/"+key, val, self.global_step)

        else:
            # [B, N, 3] --> [B, N]
            # FIXME: uncomment the line below if necessary
            loss = self.criterion(pred_rgb, gt_rgb).mean(-1)
            # print("pred_rgb: {}".format(pred_rgb.shape))
            # print("gt_rgb: {}".format(gt_rgb.shape))
            # print("loss: {}".format(loss))
            # print("loss: {}".format(loss.min()))
            # print("loss: {}".format(loss.max()))
            # print("loss: {}".format(loss.isinf()))
            # print("loss: {}".format(loss.isinf().sum()))

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

        # print(loss_dict)

        loss = loss.mean()

        # FIXME: NEW!!!
        # deform regularization
        if 'deform' in ret and ret['deform'] is not None:
            loss = loss + 1e-3 * ret['deform'].abs().mean()

        return pred_rgb, gt_rgb, loss

    def eval_step(self, data):

        data = self.render_prereqs(data)

        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        time = data['time']  # [B, 1]
        images = data['images']  # [B, H, W, 3/4]
        B, H, W, C = images.shape

        self.opt.inds_s = data['inds_s']
        self.opt.inds_d = data['inds_d']

        if self.opt.color_space == 'linear':
            images[..., : 3] = srgb_to_linear(images[..., : 3])

        # eval with fixed background color
        bg_color = 1
        if C == 4:
            gt_rgb = images[..., : 3] * images[..., 3:] + \
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

        if ('inds_s' in data and 'inds_d' in data):
            self.opt.inds_s = data['inds_s']
            self.opt.inds_d = data['inds_d']
        else:
            self.opt.inds_s = 0
            self.opt.inds_d = 0

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

        rays = get_rays(pose, intrinsics, rH, rW, None, -1)

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
