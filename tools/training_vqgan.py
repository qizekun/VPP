import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tools import builder
from utils import misc, dist_utils
import numpy as np
from utils.logger import *
from utils.AverageMeter import AverageMeter
from torchvision import transforms
from datasets import data_transforms
from modules.voxelization import voxel_to_point

train_transforms = transforms.Compose(
    [
        # data_transforms.PointcloudScale(),
        # data_transforms.PointcloudRotate(),
        # data_transforms.PointcloudRotatePerturbation(),
        # data_transforms.PointcloudTranslate(),
        # data_transforms.PointcloudJitter(),
        # data_transforms.PointcloudRandomInputDropout(),
        # data_transforms.PointcloudScaleAndTranslate(),
    ]
)


class Loss_Metric:
    def __init__(self, loss=torch.inf):
        if type(loss).__name__ == 'dict':
            self.loss = loss['loss']
        else:
            self.loss = loss

    def better_than(self, other):
        if self.loss < other.loss:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['loss'] = self.loss
        return _dict


def test(args, config, test_dataloader, generator, logger):
    ckpt_path = args.ckpts
    generator.load_model_from_ckpt(ckpt_path)
    generator.eval()
    gt_points = []
    rec_points = []
    with torch.no_grad():
        for taxonomy_ids, model_ids, data, _, _, _ in test_dataloader:
            points = data.cuda()
            gt_voxel, rec_voxel, _ = generator(points)

            B = gt_voxel.shape[0]
            for i in range(B):
                gt_points.append(voxel_to_point(voxel=gt_voxel[i]).cpu().numpy())
                rec_points.append(voxel_to_point(voxel=rec_voxel[i]).cpu().numpy())

    voxel_data = {
        'gt_voxel': gt_points,
        'rec_voxel': rec_points,
    }
    np.save(os.path.join(args.experiment_path, 'vqgan.npy'), voxel_data)
    print_log(f'[VQGAN] Saving ground truth voxel and reconstruction voxel to '
              f'{os.path.join(args.experiment_path, "vqgan.npy")}', logger=logger)


def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    # config.dataset.train.others.whole = True
    train_sampler, train_dataloader = builder.dataset_builder(args, config.dataset.train)

    # build model
    # config.discriminator.with_color = config.model.with_color = config.with_color
    config.model.with_color = config.with_color
    generator = builder.model_builder(config.model)
    discriminator = builder.model_builder(config.discriminator)
    if args.use_gpu:
        generator.to(args.local_rank)
        discriminator.to(args.local_rank)

    if args.test:
        (_, test_dataloader) = builder.dataset_builder(args, config.dataset.val) if config.dataset.get(
            'val') else (None, None)
        return test(args, config, test_dataloader, generator, logger)

    # parameter setting
    start_epoch = 0
    best_metrics = Loss_Metric(torch.inf)
    metrics = Loss_Metric(torch.inf)

    if args.distributed:
        # Sync BN
        if args.sync_bn:
            generator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(generator)
            discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        generator = nn.parallel.DistributedDataParallel(generator,
                                                        device_ids=[args.local_rank % torch.cuda.device_count()],
                                                        broadcast_buffers=False,
                                                        find_unused_parameters=True)
        discriminator = nn.parallel.DistributedDataParallel(discriminator,
                                                            device_ids=[args.local_rank % torch.cuda.device_count()],
                                                            broadcast_buffers=False,
                                                            find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        generator = nn.DataParallel(generator).cuda()
        discriminator = nn.DataParallel(discriminator).cuda()

    # optimizer & scheduler
    g_optimizer, g_scheduler = builder.build_opti_sche(generator, config)
    d_optimizer, d_scheduler = builder.build_opti_sche(discriminator, config)

    # trainval
    # training
    generator.zero_grad()
    discriminator.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        generator.train()
        discriminator.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['Loss'])

        num_iter = 0

        n_batches = len(train_dataloader)
        for idx, (data, _, _) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx

            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train.others.npoints
            dataset_name = config.dataset.train._base_.NAME
            points = data.cuda()

            assert points.size(1) == npoints
            points = train_transforms(points)

            # for p in discriminator.parameters():
            #     p.data.clamp_(-0.01, 0.01)

            voxel, decoded_voxel, q_loss = generator(points)
            coord = voxel[:, :1]
            decoded_coord = decoded_voxel[:, :1]

            disc_real = discriminator(coord)
            disc_fake = discriminator(decoded_coord)

            disc_factor = generator.module.adopt_weight(config.discriminator.factor, epoch,
                                                        threshold=config.discriminator.start_epoch)

            if config.model.with_color:
                occupy = coord > 0.5
                decoded_voxel[:, 1:] = decoded_voxel[:, 1:] * occupy.detach()

            rec_loss = torch.abs(voxel - decoded_voxel).mean()
            occ_loss = torch.abs(coord.mean(dim=(1, 2, 3, 4)) - decoded_coord.mean(dim=(1, 2, 3, 4))).mean()
            perceptual_rec_loss = rec_loss + occ_loss

            g_loss = -torch.mean(disc_fake)
            lam = generator.module.calculate_lambda(perceptual_rec_loss, g_loss)
            g_loss = disc_factor * lam * g_loss

            vq_loss = perceptual_rec_loss + q_loss + g_loss

            d_loss_real = torch.mean(F.relu(1. - disc_real))
            d_loss_fake = torch.mean(F.relu(1. + disc_fake))
            d_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)

            g_optimizer.zero_grad()
            vq_loss.backward(retain_graph=True)

            d_optimizer.zero_grad()
            d_loss.backward()

            g_optimizer.step()
            d_optimizer.step()

            print(g_loss, d_loss, rec_loss, q_loss)

            loss = rec_loss + occ_loss

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                losses.update([loss.item() * 1000])
            else:
                losses.update([loss.item() * 1000])

            if args.distributed:
                torch.cuda.synchronize()

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', g_optimizer.param_groups[0]['lr'], n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                          (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                           ['%.4f' % l for l in losses.val()], g_optimizer.param_groups[0]['lr']), logger=logger)

        g_scheduler.step(epoch)
        d_scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss_1', losses.avg(0), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
                  (epoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],
                   g_optimizer.param_groups[0]['lr']), logger=logger)

        builder.save_checkpoint(generator, g_optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger=logger)
        if epoch % args.val_freq == 0 and test_dataloader is not None:
            # Validate the current model
            metrics = validate(generator, test_dataloader, epoch, val_writer, args, config, best_metrics,
                               logger=logger)
            better = metrics.better_than(best_metrics)
            # Save ckeckpoints
            if better:
                best_metrics = metrics
                builder.save_checkpoint(generator, g_optimizer, epoch, metrics, best_metrics, 'ckpt-best',
                                        args, logger=logger)
                print_log(
                    "--------------------------------------------------------------------------------------------",
                    logger=logger)
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()


def validate(base_model, test_dataloader, epoch, val_writer, args, config, best_metrics, logger=None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger=logger)
    base_model.eval()  # set model to eval mode

    losses = AverageMeter(['Loss'])
    with torch.no_grad():
        for idx, (data, _, _) in enumerate(test_dataloader):
            points = data.cuda()

            voxel, decoded_voxel, q_loss = base_model(points)
            coord = voxel[:, :1]
            decoded_coord = decoded_voxel[:, :1]

            occ_loss = torch.abs(coord.mean(dim=(1, 2, 3, 4)) - decoded_coord.mean(dim=(1, 2, 3, 4))).mean()
            if config.model.with_color:
                occupy = coord > 0.5
                decoded_voxel[:, 1:] = decoded_voxel[:, 1:] * occupy.detach()

            rec_loss = torch.abs(voxel - decoded_voxel).mean()
            loss = rec_loss + occ_loss

            losses.update([loss.item() * 1000])
        print_log('[Validation] EPOCH: %d  loss = %.4f, best = %.4f' %
                  (epoch, losses.avg(0), min(best_metrics.loss, losses.avg(0))), logger=logger)

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/LOSS', loss, epoch)

    return Loss_Metric(losses.avg(0))
