from tools import train_point_upsampler as train_point_upsampler
from tools import train_voxel_generator as train_voxel_generator
from tools import train_smoother as train_smoother
from tools import test_run_net as test_net
from tools import finetune_run_net as finetune
from tools import inference_run_net as inference
from tools import upsample_run_net as upsample
from tools import smooth_run_net as smooth
from tools import partial_run_net as partial
from tools import train_vqgan as vggan
from tools import points_edit as edit
from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *
import time
import os
import torch
from tensorboardX import SummaryWriter


def main():
    # args
    args = parser.get_args()
    # CUDA
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        args.distributed = False
        args.world_size = 1
    else:
        args.distributed = True
        dist_utils.init_dist(args.launcher)
        # re-set gpu_ids with distributed training mode
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size
    # logger
    if args.exp_name:
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    else:
        log_file = None
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    # define the tensorboard writer
    if args.local_rank == 0 and args.exp_name:
        train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
        val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))
    else:
        train_writer = None
        val_writer = None
    # config
    config = get_config(args, logger=logger)
    # batch size
    set_batch_size(args, config)
    # log
    log_args_to_file(args, 'args', logger=logger)
    log_config_to_file(config, 'config', logger=logger)
    logger.info(f'Distributed training: {args.distributed}')
    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed + args.local_rank,
                             deterministic=args.deterministic)  # seed + rank, for augmentation
    if args.distributed:
        assert args.local_rank == torch.distributed.get_rank()

    # run
    if args.vqgan:
        vggan(args, config)
    elif args.inference:
        inference(args, config)
    elif args.upsample:
        upsample(args, config)
    elif args.point:
        train_point_upsampler(args, config, train_writer, val_writer)
    elif args.voxel:
        train_voxel_generator(args, config, train_writer, val_writer)
    elif args.smooth:
        if args.test:
            smooth(args, config, train_writer, val_writer)
        else:
            train_smoother(args, config, train_writer, val_writer)
    elif args.edit:
        edit(args, config)
    elif args.partial:
        partial(args, config)
    else:
        if args.test:
            test_net(args, config)
        elif args.finetune_model or args.scratch_model:
            finetune(args, config, train_writer, val_writer)


if __name__ == '__main__':
    main()
