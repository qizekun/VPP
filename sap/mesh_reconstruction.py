import os
import json
import argparse
import torch

from models.pointnet2_with_pcld_condition import PointNet2CloudCondition
from models.dpsr_utils.dpsr import DPSR
from models.npz_dataset import GeneralNpzDataset
from models.dpsr_evaluation import visualize_per_rank
from models.json_reader import restore_string_to_list_in_a_dict


def main(ckpt, dataset_path, save_dir, split_points_to_normals, rank=0, num_gpus=1, label_number=None):
    if split_points_to_normals:
        ext_vis_dataset = GeneralNpzDataset(dataset_path, scale=trainset_config['scale'],
                                            noise_magnitude=trainset_config["augmentation"]["noise_magnitude"],
                                            rank=rank, world_size=num_gpus, data_key='points',
                                            data_key_split_names=['points', 'normals'],
                                            data_key_split_dims=[0, 3, 6])
    else:
        ext_vis_dataset = GeneralNpzDataset(dataset_path, scale=trainset_config['scale'],
                                            noise_magnitude=trainset_config["augmentation"]["noise_magnitude"],
                                            rank=rank, world_size=num_gpus, data_key='points')
    ext_vis_loader = torch.utils.data.DataLoader(ext_vis_dataset,
                                                 batch_size=int(trainset_config['eval_batch_size'] / num_gpus),
                                                 shuffle=False, num_workers=trainset_config['num_workers'])

    net = PointNet2CloudCondition(pointnet_config)
    checkpoint = torch.load(ckpt, map_location='cpu')
    net.load_state_dict(checkpoint['model_state_dict'])
    net.cuda()
    net.eval()

    dpsr = DPSR(res=(dpsr_config['grid_res'], dpsr_config['grid_res'], dpsr_config['grid_res']),
                sig=dpsr_config['psr_sigma']).cuda()

    visualize_per_rank(net, dpsr, ext_vis_loader, pointnet_config, dpsr_config, trainset_config,
                       train_config['dataset'],
                       save_dir, 0, 0, scale=trainset_config['scale'], rank=rank, world_size=num_gpus,
                       use_autoencoder=False, autoencoder=None,
                       noise_magnitude=0, sample_points_from_mesh=True, explicit_normalize=True,
                       label_number=label_number, return_original_scale=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        default='../configs/shapenet_psr_configs'
                                '/config_refine_and_upsample_standard_attention_noise_0.025.json',
                        help='JSON file for configuration')
    parser.add_argument('--ckpt', type=str,
                        default='../exps/exp_shapenet_psr_generation/T1000_betaT0'
                                '.02_shapenet_dpsr_upsample_10_noise_0.025/checkpoint/pointnet_ckpt_748019.pkl',
                        help='the checkpoint to use')
    parser.add_argument('--dataset_path', type=str,
                        default='shapenet_psr_generated_data_2048_pts_epoch_2200_iter_195799.npz',
                        help='The npz file that contains the point clouds that we want to reconstruct mesh for. It '
                             'should contain keys: points of shape (B, npoints, 3), the point clouds will be '
                             'explicitly normalized; normals of shape (B, npoints, 3) if the SAP model need to use '
                             'normals for recontruction; label of shape (B,) that specify the category of each point '
                             'cloud, if the npz file do not contain the key label, we will assume that all point '
                             'clouds in the file belong to the label specified by the argument label_number.')
    parser.add_argument('--save_dir', type=str, default='dpsr_reconstruct_mesh', help='the directory to save meshes')
    parser.add_argument('--split_points_to_normals', action='store_true',
                        help='whether to split points in the npz file to points and normals')
    parser.add_argument('--label_number', type=int, default=-1,
                        help='the label of pcds stored in the npz file, it it only valid when label is not a key in '
                             'the npz file')
    args = parser.parse_args()

    # Parse configs. Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    config = restore_string_to_list_in_a_dict(config)

    global train_config
    train_config = config["train_config"]  # training parameters
    global pointnet_config
    pointnet_config = config["pointnet_config"]  # to define pointnet

    global trainset_config
    if train_config['dataset'] == 'mvp_dataset':
        trainset_config = config["mvp_dataset_config"]
    elif train_config['dataset'] == 'shapenet_psr_dataset':
        trainset_config = config['shapenet_psr_dataset_config']
    else:
        raise Exception('%s dataset is not supported' % train_config['dataset'])

    global dpsr_config
    dpsr_config = config['dpsr_config']

    save_dir = args.save_dir
    file_name = os.path.split(args.dataset_path)[1]
    file_name = os.path.splitext(file_name)[0]
    save_dir = os.path.join(save_dir, file_name)

    main(args.ckpt, args.dataset_path, save_dir, args.split_points_to_normals, rank=0, num_gpus=1,
         label_number=args.label_number)
