import torch
import argparse
import numpy as np
from utils import misc
from utils.config import *
from utils.load import load, pc_norm
from datasets import build_dataset_from_cfg
from evals.classifier.classifier import PointNetClassifier, pc_norm


def retrieval(pts, cls_config, k=10, with_color=False):

    npoints = cls_config.npoints
    pts = torch.from_numpy(pts).cuda()
    if len(pts.shape) == 2:
        pts = pts.unsqueeze(0)
    retrival_points = misc.fps(pts, npoints)
    retrival_points = pc_norm(retrival_points)

    dataset = build_dataset_from_cfg(cls_config.dataset.train._base_, cls_config.dataset.train.others)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cls_config.total_bs, shuffle=False,
                                             drop_last=False, num_workers=8)

    classifier = PointNetClassifier(normal_channel=with_color)
    classifier.load_model_from_ckpt(cls_config.ckpt_path)
    classifier.cuda()
    classifier.eval()

    with torch.no_grad():
        retrival_features = classifier.features_and_preds(retrival_points)[0]

        gt_features = []
        dataset_pts = []
        for idx, (data, _, _) in enumerate(dataloader):
            gt_points = data.cuda()
            gt_points = misc.fps(gt_points, npoints)
            gt_points = pc_norm(gt_points)

            features, _ = classifier.features_and_preds(gt_points)
            gt_features.append(features)
            dataset_pts.append(data)

        gt_features = torch.cat(gt_features, dim=0)
        dataset_pts = torch.cat(dataset_pts, dim=0)

        sim = torch.cosine_similarity(retrival_features, gt_features, dim=1)
        sim = sim.cpu().numpy()
        idx = np.argsort(sim)[::-1][:k]
        result = []
        for i in idx:
            result.append(dataset_pts[i].cpu().numpy())
        result = np.array(result)
        np.save("retrieval.npy", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pts_path", type=str, default="table3.npy")
    args = parser.parse_args()
    pts_path = args.pts_path
    pts = load(pts_path)
    with_color = False
    cls_config = cfg_from_yaml_file("cfgs/classifier.yaml")
    retrieval(pts, cls_config, with_color=with_color, k=10)
