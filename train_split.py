#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import numpy as np

import torch
import torchvision
import json
import wandb
import time
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
# from lpipsPyTorch import lpips
import lpips
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import prefilter_voxel, render, renderWithScore
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.encodings import get_binary_vxl_size
from scene.cameras import CameraManager, UnlimitedVRAMCameraManager
import faiss
from AnchorScoreTracker import AnchorScoreTracker

# torch.set_num_threads(32)
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

# from lpipsPyTorch import lpips

bit2MB_scale = 8 * 1024 * 1024
run_codec = True

def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = pathlib.Path(__file__).parent.resolve()


    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))

    print('Backup Finished!')


def training(args_param, dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, wandb=None, logger=None, ply_path=None):
    first_iter = 0

    is_synthetic_nerf = os.path.exists(os.path.join(dataset.source_path, "transforms_train.json"))
    gaussians = GaussianModel(
        dataset.feat_dim,
        dataset.n_offsets,
        dataset.voxel_size,
        dataset.update_depth,
        dataset.update_init_factor,
        dataset.update_hierachy_factor,
        dataset.use_feat_bank,
        n_features_per_level=args_param.n_features,
        log2_hashmap_size=args_param.log2,
        log2_hashmap_size_2D=args_param.log2_2D,
        is_synthetic_nerf=is_synthetic_nerf,
    )
    scene = Scene(dataset, gaussians, ply_path=ply_path)
    anchorScoreTracker = AnchorScoreTracker()

    gaussians.update_anchor_bound()

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # NEW
    cameraManager = CameraManager(scene.getTrainCameras(), 60)
    initial_num_anchors = gaussians.get_anchor.shape[0]

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    torch.cuda.synchronize(); t_start = time.time()
    log_time_sub = 0
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        viewpoint_cam = cameraManager.getNextCamera()

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)
        retain_grad = (iteration < opt.update_until and iteration >= 0)
        if iteration > 600 and iteration <= args.segIter:
            render_pkg = renderWithScore(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad)
            # accumulate gaussian -> anchor
            gauss_scores = render_pkg["gaussian_scores"].detach().view(-1)     # [N_gauss_kept]
            anchor_map   = render_pkg["anchor_map"]                   # [N_gauss_kept]
            anchorScoreTracker.update(gauss_scores, anchor_map, initial_num_anchors)
        else:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad, step=iteration)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad, step=iteration)

        image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]

        bit_per_param = render_pkg["bit_per_param"]
        bit_per_feat_param = render_pkg["bit_per_feat_param"]
        bit_per_scaling_param = render_pkg["bit_per_scaling_param"]
        bit_per_offsets_param = render_pkg["bit_per_offsets_param"]

        if iteration % 1000 == 0 and bit_per_param is not None:

            ttl_size_feat_MB = bit_per_feat_param.item() * gaussians.get_anchor.shape[0] * gaussians.feat_dim / bit2MB_scale
            ttl_size_scaling_MB = bit_per_scaling_param.item() * gaussians.get_anchor.shape[0] * 6 / bit2MB_scale
            ttl_size_offsets_MB = bit_per_offsets_param.item() * gaussians.get_anchor.shape[0] * 3 * gaussians.n_offsets / bit2MB_scale
            ttl_size_MB = ttl_size_feat_MB + ttl_size_scaling_MB + ttl_size_offsets_MB

            logger.info("\n----------------------------------------------------------------------------------------")
            logger.info("\n-----[ITER {}] bits info: bit_per_feat_param={}, anchor_num={}, ttl_size_feat_MB={}-----".format(iteration, bit_per_feat_param.item(), gaussians.get_anchor.shape[0], ttl_size_feat_MB))
            logger.info("\n-----[ITER {}] bits info: bit_per_scaling_param={}, anchor_num={}, ttl_size_scaling_MB={}-----".format(iteration, bit_per_scaling_param.item(), gaussians.get_anchor.shape[0], ttl_size_scaling_MB))
            logger.info("\n-----[ITER {}] bits info: bit_per_offsets_param={}, anchor_num={}, ttl_size_offsets_MB={}-----".format(iteration, bit_per_offsets_param.item(), gaussians.get_anchor.shape[0], ttl_size_offsets_MB))
            logger.info("\n-----[ITER {}] bits info: bit_per_param={}, anchor_num={}, ttl_size_MB={}-----".format(iteration, bit_per_param.item(), gaussians.get_anchor.shape[0], ttl_size_MB))
            with torch.no_grad():
                binary_grid_masks_anchor = gaussians.get_mask_anchor.float()
                mask_1_rate, mask_size_bit, mask_size_MB, mask_numel = get_binary_vxl_size(binary_grid_masks_anchor + 0.0)  # [0, 1] -> [-1, 1]
            logger.info("\n-----[ITER {}] bits info: 1_rate_mask={}, mask_numel={}, mask_size_MB={}-----".format(iteration, mask_1_rate, mask_numel, mask_size_MB))

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        ssim_loss = (1.0 - ssim(image, gt_image))
        scaling_reg = scaling.prod(dim=1).mean()
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01*scaling_reg

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 100 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(100)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            torch.cuda.synchronize(); t_start_log = time.time()
            #training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), wandb, logger, args_param.model_path)
            if (iteration in saving_iterations):
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                #scene.save(iteration)
                scene.save_separate(iteration)
            torch.cuda.synchronize(); t_end_log = time.time()
            t_log = t_end_log - t_start_log
            log_time_sub += t_log
            
            if iteration == args.segIter:
                cameraManager.toggleMaskLoadingOff()
                anchor_scores = anchorScoreTracker.get_scores()

                # 1. Set objectid = 1 if score > ..., else remain 0
                objectid = torch.zeros_like(anchor_scores, dtype=torch.int8)
                objectid[anchor_scores > args.segReq] = 1
                n_init = int(objectid.sum().item())
                print(f"[seg] anchors above threshold: {n_init}/{len(objectid)}")
                
                xyz_all = gaussians.get_anchor.detach().cpu().numpy().astype('float32')
                objid_np = objectid.cpu().numpy()

                # 2. Expand objectid=1 via FAISS
                print("expanding objectid 1 via radius search")
                radius = args.segSpread * scene.cameras_extent
                src_mask = (objid_np == 1)
                tgt_mask = (objid_np == 0)

                src_xyz = xyz_all[src_mask]
                tgt_xyz = xyz_all[tgt_mask]
                tgt_indices = np.where(tgt_mask)[0]

                index = faiss.IndexFlatL2(3)
                index.add(tgt_xyz)
                LIMS, D, I = index.range_search(src_xyz, radius**2)
                global_ids = tgt_indices[I.astype(np.int64)]
                unique_ids = np.unique(global_ids)
                objectid[unique_ids] = 1  # promote these to objectid 1

                print(f"[seg] FAISS promoted anchors: {len(unique_ids)}")
                n_final = int(objectid.sum().item())
                print(f"[seg] total anchors after expansion: {n_final}/{len(objectid)} "
                    f"(+{n_final - n_init} added via FAISS)")

                # 3. Update model
                gaussians._object_id = objectid.cuda()
                # Per-anchor object ids (int32)
                obj_ids = gaussians._object_id  # [N]
                unique_ids = torch.unique(obj_ids)
                print(f"unique object ids: {unique_ids.tolist()}")

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            if (iteration in checkpoint_iterations):
                logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    torch.cuda.synchronize(); t_end = time.time()
    logger.info("\n Total Training time: {}".format(t_end-t_start-log_time_sub))

    return gaussians.x_bound_min, gaussians.x_bound_max, scene, gaussians

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, object_id=None):
    # folder suffix if filtering by object id
    suffix = f"ours_{iteration}" if object_id is None else f"ours_{iteration}_obj{int(object_id)}"
    render_path = os.path.join(model_path, name, suffix, "renders")
    error_path  = os.path.join(model_path, name, suffix, "errors")
    gts_path    = os.path.join(model_path, name, suffix, "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(error_path,  exist_ok=True)
    makedirs(gts_path,    exist_ok=True)

    t_list, visible_count_list, name_list, per_view_dict, psnr_list = [], [], [], {}, []
    for idx, view in enumerate(tqdm(views, desc=f"Rendering progress{'' if object_id is None else f' (obj{int(object_id)})'}")):
        view.load()
        torch.cuda.synchronize(); t_start = time.time()

        # 1) standard prefilter for frustum/size
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)  # [N_anchors] bool (CUDA)

        # 2) AND with object mask if requested
        if object_id is not None:
            # gaussians._object_id set in training (dtype int8 / int, device CUDA)
            obj_mask = (gaussians._object_id.view(-1) == int(object_id))
            voxel_visible_mask = voxel_visible_mask & obj_mask

        # 3) render only the selected anchors
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)

        torch.cuda.synchronize(); t_end = time.time()
        t_list.append(t_end - t_start)

        # outputs
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = (render_pkg["radii"] > 0).sum()
        visible_count_list.append(visible_count)

        gt = view.original_image[0:3, :, :]
        gt_image = torch.clamp(view.original_image.to("cuda"), 0.0, 1.0)
        render_image = torch.clamp(rendering.to("cuda"), 0.0, 1.0)
        psnr_view = psnr(render_image, gt_image).mean().double()
        psnr_list.append(psnr_view)

        errormap = (rendering - gt).abs()

        fname = f"{idx:05d}.png"
        name_list.append(fname)
        torchvision.utils.save_image(rendering, os.path.join(render_path, fname))
        torchvision.utils.save_image(errormap, os.path.join(error_path,  fname))
        torchvision.utils.save_image(gt,        os.path.join(gts_path,    fname))
        per_view_dict[fname] = visible_count.item()
        view.unload()

    with open(os.path.join(model_path, name, suffix, "per_view_count.json"), 'w') as fp:
        json.dump(per_view_dict, fp, indent=True)

    if len(psnr_list) > 0:
        print(f'{suffix} testing_float_psnr=:', sum(psnr_list) / len(psnr_list))

    return t_list, visible_count_list

def render_sets(args_param, dataset: ModelParams, iteration: int, pipeline: PipelineParams,
                scene, gaussians, wandb=None, tb_writer=None, dataset_name=None,
                logger=None, x_bound_min=None, x_bound_max=None):
    with torch.no_grad():
        gaussians.eval()
        if x_bound_min is not None:
            gaussians.x_bound_min = x_bound_min
            gaussians.x_bound_max = x_bound_max

        bg_color  = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # list which object IDs actually exist (e.g., [0,1])
        if hasattr(gaussians, "_object_id") and gaussians._object_id is not None:
            uniq = torch.unique(gaussians._object_id).tolist()
            uniq = [int(u) for u in uniq]
        else:
            uniq = [0]  # fallback: single object

        # test cameras come from the scene you already built during training
        test_views = scene.getTestCameras()
        all_fps = {}

        for oid in uniq:
            t_list, _ = render_set(dataset.model_path, "test", scene.loaded_iter, test_views,
                                   gaussians, pipeline, background, object_id=oid)
            if len(t_list) > 5:
                fps = 1.0 / torch.tensor(t_list[5:]).mean()
                all_fps[f"obj{oid}"] = fps.item()
                logger and logger.info(f'Test FPS (obj{oid}): \033[1;35m{fps.item():.5f}\033[0m')

        # you can return counts for the last call, or aggregate if you prefer
        return None


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO)
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu", type=str, default = '-1')
    parser.add_argument("--log2", type=int, default = 13)
    parser.add_argument("--log2_2D", type=int, default = 15)
    parser.add_argument("--n_features", type=int, default = 4)
    parser.add_argument("--lmbda", type=float, default = 0.001)
    parser.add_argument("--memMB", type=int, default = None)
    parser.add_argument("--segIter", type=int, default = 500)
    parser.add_argument("--segReq", type=float, default = 0.1)
    parser.add_argument("--segSpread", type=float, default = 0.1)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # enable logging

    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)
    logger.info(f'args: {args}')

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")
        logger.info(f'using GPU {args.gpu}')

    dataset = args.source_path.split('/')[-1]
    exp_name = args.model_path.split('/')[-2]

    logger.info("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    args.port = np.random.randint(10000, 20000)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # training
    x_bound_min, x_bound_max, scene, gaussians = training(args, lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, None, logger)

    # All done
    logger.info("\nTraining complete.")

    # rendering
    logger.info(f'\nStarting Rendering~')
    visible_count = render_sets(args, lp.extract(args), -1, pp.extract(args), scene, gaussians, wandb=None, logger=logger, x_bound_min=x_bound_min, x_bound_max=x_bound_max)
    logger.info("\nRendering complete.")

