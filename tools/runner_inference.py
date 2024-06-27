import cv2
import time
import torch
import numpy as np
from tools import builder
from utils.logger import *
from utils.load import load, pc_norm
from utils import misc, dist_utils
from utils.transforms import get_transforms
from modules.voxelization import voxel_to_point


def inference(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)

    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)
    base_model.eval()

    start_time = time.time()
    if args.text:
        text_query = args.query
        predict_points = base_model.text_condition_generation(text_query)
        print(text_query)
    elif args.img:
        img_path = args.img_path
        img = cv2.imread(img_path)
        img = get_transforms()['test'](img)
        img = img.unsqueeze(0).cuda()
        predict_points = base_model.image_condition_generation(img)
    else:
        raise NotImplementedError

    end_time = time.time()
    print('running time: ', end_time - start_time)
    np.save('generated_points.npy', predict_points.cpu().numpy())
    print('Successfully save generation data to generated_points.npy')


def editing(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    text_query = args.query
    pts_path = args.pts_path

    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)
    base_model.zero_grad()
    base_model.eval()

    ori_points = load(pts_path)
    ori_points = pc_norm(ori_points)
    ori_points = torch.from_numpy(ori_points).unsqueeze(0).cuda()
    text_features = base_model.text_encoder(text_query)
    un_text_features = base_model.text_encoder("")
    _, origin_tokens, _, _ = base_model.vqgan.encode(ori_points)
    edited_points = base_model(text_features, un_text_features, origin_tokens)

    edited_data = {
        'ori_points': ori_points.cpu().numpy(),
        'edited_points': edited_points.cpu().numpy()
    }
    np.save('edited_data.npy', edited_data)
    print('Successfully save editing data to edited_data.npy')


def partial(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    text_query = args.query
    pts_path = args.pts_path

    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)
    base_model.zero_grad()
    base_model.eval()

    ori_points = load(pts_path)
    ori_points = pc_norm(ori_points)
    partial_points = ori_points[ori_points[:, 1] < 0.0]

    ori_points = torch.from_numpy(ori_points).unsqueeze(0).cuda()
    partial_points = torch.from_numpy(partial_points).unsqueeze(0).cuda()

    text_features = base_model.text_encoder(text_query)
    un_text_features = base_model.text_encoder("")
    _, origin_tokens, _, _ = base_model.vqgan.encode(ori_points)
    _, C, R, _, _ = origin_tokens.shape

    origin_tokens = origin_tokens.reshape(1, C, R * R * R)
    origin_tokens = origin_tokens.transpose(1, 2)

    num_voxel = R * R * R
    num_mask = int(num_voxel * 0.5)
    mask = np.hstack([
        np.zeros(num_voxel - num_mask),
        np.ones(num_mask),
    ])
    mask = torch.from_numpy(mask).to(torch.bool).reshape(R, R, R)
    mask = mask.transpose(0, 1)
    mask = mask.reshape(1, R * R * R)
    origin_tokens[mask] = base_model.voxel_generator.mask_token
    origin_tokens = origin_tokens.transpose(1, 2)
    origin_tokens = origin_tokens.reshape(1, C, R, R, R)

    edited_points = base_model(text_features, un_text_features, origin_tokens)[0]
    edited_points = edited_points[edited_points[:, 1] > 0.0]
    edited_points = torch.cat([edited_points, partial_points[0]], dim=0)

    edited_data = {
        'ori_points': ori_points[0].cpu().numpy(),
        'partial_points': partial_points[0].cpu().numpy(),
        'edited_points': edited_points.cpu().numpy()
    }
    np.save('partial_data.npy', edited_data)
    print('Successfully save partial generation data to partial_data.npy')


def upsample(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    pts_path = args.pts_path

    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)
    base_model.zero_grad()
    base_model.eval()

    ori_points = load(pts_path)
    ori_points = pc_norm(ori_points)
    ori_points = torch.from_numpy(ori_points).unsqueeze(0).cuda()
    if "init_num" in config.keys():
        ori_points = misc.fps(ori_points.float(), config.init_num)

    with torch.no_grad():
        upsample_points = base_model.upsample(ori_points)

    upsample_data = {
        'ori_points': ori_points.cpu().numpy(),
        'upsample_points': upsample_points.cpu().numpy()
    }
    np.save('upsample_data.npy', upsample_data)
    print('Successfully save upsampled data to upsample_data.npy')


def smooth(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.val)

    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)
    base_model.zero_grad()
    base_model.eval()

    gt_points = []
    grid_points = []
    smo_points = []
    with torch.no_grad():
        for taxonomy_ids, model_ids, data, _, _, _ in test_dataloader:
            points = data.cuda()
            B = points.shape[0]
            voxels = base_model.grid_smoother.voxelization(points)

            for i in range(B):
                grid_point = voxel_to_point(voxels[i])
                smo_point = base_model.grid_smoother.inference(grid_point.unsqueeze(dim=0))[0]
                N = smo_point.shape[0]
                gt_point = misc.fps(points[i].unsqueeze(dim=0), N)[0]

                gt_points.append(gt_point.cpu().numpy())
                grid_points.append(grid_point.cpu().numpy())
                smo_points.append(smo_point.cpu().numpy())

    smooth_data = {
        'gt_points': gt_points,
        'grid_points': grid_points,
        'smo_points': smo_points,
    }
    np.save('smoother.npy', smooth_data)
    print_log(f'[Grid Sommther] Saving grid points and smoothed points to smoother.npy')
