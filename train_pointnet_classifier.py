import os
import time
import torch
import torch.nn as nn
from evals.classifier.classifier import PointNetClassifier, Acc_Metric, pc_norm
from datasets import build_dataset_from_cfg
from tools import builder
from utils import misc
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.config import *
import argparse


def training_classifier(args):
    logger = get_logger("classifier")
    config = cfg_from_yaml_file(args.cfg)

    ckpt_path = os.path.join("experiments/classifier/", args.exp_name)
    os.makedirs(ckpt_path, exist_ok=True)

    bs = config.total_bs
    with_color = config.with_color

    train_dataset = build_dataset_from_cfg(config.dataset.train._base_, config.dataset.train.others)
    test_dataset = build_dataset_from_cfg(config.dataset.val._base_, config.dataset.val.others)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True,
                                                   drop_last=True, num_workers=8)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=bs * 2, shuffle=False,
                                                  drop_last=False, num_workers=8)
    base_model = PointNetClassifier(normal_channel=with_color)
    base_model = nn.DataParallel(base_model).cuda()
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)

    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['loss', 'acc'])
        num_iter = 0
        base_model.train()  # set model to training mode

        npoints = config.npoints
        for idx, (taxonomy_ids, model_ids, data, _, _, label) in enumerate(train_dataloader):
            num_iter += 1
            data_time.update(time.time() - batch_start_time)
            points = data.cuda()
            label = label.cuda()
            points = misc.fps(points, npoints)
            points = pc_norm(points)

            ret = base_model(points)
            loss, acc = base_model.module.get_loss_acc(ret, label)
            loss.backward()

            # forward
            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            losses.update([loss.item(), acc.item()])
            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
                  (epoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],
                   optimizer.param_groups[0]['lr']), logger=logger)

        # Validate the current model
        metrics = validate(base_model, test_dataloader, best_metrics, epoch, config, logger=logger)

        better = metrics.better_than(best_metrics)
        # Save ckeckpoints
        if better:
            best_metrics = metrics
            torch.save(base_model.state_dict(), os.path.join(ckpt_path, 'best_model.pth'))
        if epoch % 10 == 0:
            torch.save(base_model.state_dict(), os.path.join(ckpt_path, 'epoch_%d.pth' % epoch))


def validate(base_model, test_dataloader, best_metrics, epoch, config, logger=None):
    base_model.eval()  # set model to eval mode

    test_pred = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data, _, _, label) in enumerate(test_dataloader):
            points = data.cuda()
            label = label.cuda()
            points = misc.fps(points, npoints)
            points = pc_norm(points)

            logits = base_model(points)
            target = label.view(-1)

            pred = logits.argmax(-1).view(-1)
            test_pred.append(pred.detach())
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[Validation] EPOCH: %d  acc = %.4f best_acc = %.4f' % (epoch, acc, max(acc, best_metrics.acc)),
                  logger=logger)

    return Acc_Metric(acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfgs/classifier.yaml', help='config file')
    parser.add_argument('--exp_name', type=str, default=None, help='experiment name')
    args = parser.parse_args()
    training_classifier(args)
