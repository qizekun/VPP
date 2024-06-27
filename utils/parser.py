import os
import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        help='yaml config file')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)
    # bn
    parser.add_argument(
        '--sync_bn',
        action='store_true',
        default=False,
        help='whether to use sync bn')
    # seed 
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    # some args
    parser.add_argument('--exp_name', type=str, default=None, help='experiment name')
    parser.add_argument('--start_ckpts', type=str, default=None, help='reload used ckpt path')
    parser.add_argument('--ckpts', type=str, default=None, help='test used ckpt path')
    parser.add_argument('--val_freq', type=int, default=1, help='test freq')
    parser.add_argument('--pts_path', type=str, default=None, help='input point clouds path')
    parser.add_argument('--img_path', type=str, default=None, help='input image path')
    parser.add_argument('--query', type=str, default=None, help='text condition query')
    parser.add_argument('--text', action='store_true', default=False, help='text condition generation')
    parser.add_argument('--img', action='store_true', default=False, help='image condition generation')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='autoresume training (interrupted by accident)')
    parser.add_argument(
        '--test',
        action='store_true',
        default=False,
        help='test mode for certain ckpt')
    parser.add_argument(
        '--vqgan',
        action='store_true',
        default=False,
        help='training vqgan')
    parser.add_argument(
        '--inference',
        action='store_true',
        default=False,
        help='inference model')
    parser.add_argument(
        '--upsample',
        action='store_true',
        default=False,
        help='upsample model')
    parser.add_argument(
        '--point',
        action='store_true',
        default=False,
        help='train the point upsampler')
    parser.add_argument(
        '--smooth',
        action='store_true',
        default=False,
        help='train the grid smoother')
    parser.add_argument(
        '--voxel',
        action='store_true',
        default=False,
        help='train the voxel generator')
    parser.add_argument(
        '--edit',
        action='store_true',
        default=False,
        help='edit the inputs point with text prompts')
    parser.add_argument(
        '--partial',
        action='store_true',
        default=False,
        help='edit the partial inputs point with text prompts')
    parser.add_argument(
        '--finetune_model',
        action='store_true',
        default=False,
        help='finetune modelnet with pretrained weight')
    parser.add_argument(
        '--scratch_model',
        action='store_true',
        default=False,
        help='training modelnet from scratch')

    args = parser.parse_args()

    if args.test and args.resume:
        raise ValueError(
            '--test and --resume cannot be both activate')

    if args.resume and args.start_ckpts is not None:
        raise ValueError(
            '--resume and --start_ckpts cannot be both activate')

    if args.finetune_model and args.ckpts is None:
        print(
            'training from scratch')

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.exp_name is not None:
        if args.test:
            args.exp_name += '_test'
        args.experiment_path = os.path.join('./experiments', Path(args.config).stem, Path(args.config).parent.stem,
                                            args.exp_name)
        args.tfboard_path = os.path.join('./experiments', Path(args.config).stem, Path(args.config).parent.stem, 'TFBoard',
                                         args.exp_name)
        create_experiment_dir(args)
    args.log_name = Path(args.config).stem
    return args


def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)
    if not os.path.exists(args.tfboard_path):
        os.makedirs(args.tfboard_path)
        print('Create TFBoard path successfully at %s' % args.tfboard_path)
