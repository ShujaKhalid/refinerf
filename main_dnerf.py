from tkinter import W
import torch
import argparse

from dnerf.provider import NeRFDataset
from dnerf.gui import NeRFGUI
from dnerf.utils import *

from functools import partial
from loss import huber_loss

# torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-O', action='store_true',
                        help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)

    # =================================================================================
    # training options
    parser.add_argument('--iters', type=int, default=100000,
                        help="training iters")
    parser.add_argument('--lr', type=float, default=1e-2,  # 1e-2
                        help="initial learning rate")
    parser.add_argument('--lr_net', type=float, default=1e-3,  # 1e-3
                        help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--num_rays', type=int, default=1024,
                        help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true',
                        help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=256,  # sk_debug: used to be 1024
                        help="max num steps sampled per ray (only valid when using --cuda_ray)")
    # parser.add_argument('--dynamic_iters', type=str, default="[(204,312), (480,600), (2400, 3000)]",  # 2400 iters
    # parser.add_argument('--dynamic_iters', type=str, default="[(480, 960), (1200, 1440), (2400, 3600), (6000, 7200), (9600, 10800), (14400, 18000), (21600, 24000)]",  # 2400 iters
    # parser.add_argument('--dynamic_iters', type=str, default="{'d1': (2400, 3600), 'b1': (3600, 4800), 'd3': (6000, 7200), 'b3': (10800, 14400), 'd2': (15600, 16800)}",  # 2400 iters # BOOOO
    # parser.add_argument('--dynamic_iters', type=str, default="{'d2': (1200, 6000), 'd3': (7200, 8400), 'd4': (9600, 10800)}",  # 2400 iters
    # parser.add_argument('--dynamic_iters', type=str, default="{'d1': (1200, 2400), 'd2': (3600, 100000)}",  # 2400 iters
    parser.add_argument('--dynamic_iters', type=str, default="{'d1': (4800, 100000)}",  # 2400 iters
                        # parser.add_argument('--dynamic_iters', type=str, default="{'d1': (0, 12000)}",  # 24000 iters
                        help="intervals to train the dynamic model for")
    parser.add_argument('--update_extra_interval', type=int, default=12000000,  # TODO: used to be 100
                        help="iter interval to update extra status (only valid when using --cuda_ray)")
    # =================================================================================

    parser.add_argument('--num_steps', type=int, default=128,
                        help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0,
                        help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096,
                        help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")

    # network backbone options
    parser.add_argument('--fp16', action='store_true',
                        help="use amp mixed precision training")
    parser.add_argument('--basis', action='store_true',
                        help="[experimental] use temporal basis instead of deformation to model dynamic scene (check Fourier PlenOctree and NeuVV)")
    # parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    # parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")

    # dataset options
    parser.add_argument('--color_space', type=str, default='srgb',
                        help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', action='store_true',
                        help="preload all data into GPU, accelerate training but use more GPU memory")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=1,  # FIXME: 2
                        help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=0.33,
                        help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*',
                        default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1/128,
                        help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.2,
                        help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10,
                        help="threshold for density grid to be occupied")
    parser.add_argument('--bg_radius', type=float, default=-1,
                        help="if positive, use a background model at sphere(bg_radius)")

    # GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1920, help="GUI width")
    parser.add_argument('--H', type=int, default=1080, help="GUI height")
    parser.add_argument('--radius', type=float, default=5,
                        help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50,
                        help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=64,
                        help="GUI rendering max sample per pixel")

    # experimental
    parser.add_argument('--error_map', action='store_true',
                        help="use error map to sample rays")
    parser.add_argument('--clip_text', type=str, default='',
                        help="text input for CLIP guidance")
    parser.add_argument('--rand_pose', type=int, default=-1,
                        help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")

    opt = parser.parse_args()

    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.preload = True

    # opt.cuda_ray = False

    if opt.basis:
        assert opt.cuda_ray, "Non-cuda-ray mode is temporarily broken with temporal basis mode"
        from dnerf.network_basis import NeRFNetwork
    else:
        from dnerf.network import NeRFNetwork
        # from dnerf.network_camera import CameraNetwork
        from dnerf.network_fxfy import LearnFocal
        from dnerf.network_pose import LearnPose

    print(opt)

    seed_everything(opt.seed)

    model = NeRFNetwork(
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
        h=opt.H,
        w=opt.W,
        num_cams=24
    )
    model_fxfy = LearnFocal(H=270, W=480).cuda()  # FIXME
    model_pose = LearnPose(num_cams=24).cuda()  # FIXME
    # model_fxfy = LearnFocal(H=800, W=800).cuda()  # FIXME
    # model_pose = LearnPose(num_cams=150).cuda()  # FIXME
    # model_camera = CameraNetwork(opt.H, opt.W, num_cams=24)  # FIXME
    # send to provider for predicting int/ext camera params
    #opt.model_camera = model_camera

    # print(model_camera)

    criterion = torch.nn.MSELoss(reduction='none')
    # criterion = partial(huber_loss, reduction='none')
    # criterion = torch.nn.HuberLoss(reduction='none', beta=0.1) # only available after torch 1.10 ?

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opt.test:

        trainer = Trainer('ngp', opt, model, model_fxfy, model_pose, device=device, workspace=opt.workspace,
                          criterion=criterion, fp16=opt.fp16, metrics=[PSNRMeter()], use_checkpoint=opt.ckpt)

        # Update opt here with new
        # camera_params...

        if opt.gui:
            gui = NeRFGUI(opt, trainer)
            gui.render()

        else:
            test_loader = NeRFDataset(
                opt, device=device, type='test').dataloader()

            if test_loader.has_gt:
                # blender has gt, so evaluate it.
                trainer.evaluate(test_loader)
            else:
                # colmap doesn't have gt, so just test.
                trainer.test(test_loader)

            # trainer.save_mesh(resolution=256, threshold=10)

    else:

        def optimizer_model(model, state): return torch.optim.Adam(model.get_params(
            opt.lr, opt.lr_net, svd=state), betas=(0.9, 0.99), eps=1e-15)

        # def optimizer_cam_model(model_camera): return torch.optim.Adam(
        #     model_camera.parameters(), betas=(0.9, 0.99), eps=1e-15)

        train_loader = NeRFDataset(
            opt, device=device, type='train').dataloader()

        # decay to 0.1 * init_lr at last iter step
        def scheduler(optimizer): return optim.lr_scheduler.LambdaLR(
            optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

        trainer = Trainer('ngp', opt, model, model_fxfy, model_pose, device=device, workspace=opt.workspace, optimizer_model=optimizer_model, criterion=criterion, ema_decay=None,
                          fp16=opt.fp16, lr_scheduler=scheduler, scheduler_update_every_step=True, metrics=[PSNRMeter()], use_checkpoint=opt.ckpt, eval_interval=25)

        if opt.gui:
            gui = NeRFGUI(opt, trainer, train_loader)
            gui.render()

        else:
            valid_loader = NeRFDataset(
                opt, device=device, type='val', downscale=1).dataloader()

            max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
            trainer.train(train_loader, valid_loader, max_epoch)

            # # also test
            # test_loader = NeRFDataset(
            #     opt, device=device, type='test').dataloader()

            # if test_loader.has_gt:
            #     # blender has gt, so evaluate it.
            #     trainer.evaluate(test_loader)
            # else:
            #     # colmap doesn't have gt, so just test.
            #     trainer.test(test_loader)

            # trainer.save_mesh(resolution=256, threshold=10)
