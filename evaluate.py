import torch
from tools import builder
from utils.config import *
from evals.fid_is import compute_statistics, compute_inception_score
from utils.logger import *
from utils import misc
import numpy as np
from datasets import build_dataset_from_cfg
from evals.classifier.text_queries import generate_all_queries
from evals.classifier.classifier import PointNetClassifier, pc_norm


def evalulate(inf_config, cls_config, with_color=False, multiple=1):

    text_queries, text_labels = generate_all_queries(prefix="a")

    # build model
    base_model = builder.model_builder(inf_config.model)
    base_model.cuda()
    base_model.eval()

    npoints = cls_config.npoints
    test_dataset = build_dataset_from_cfg(cls_config.dataset.val._base_, cls_config.dataset.val.others)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cls_config.total_bs, shuffle=False,
                                                  drop_last=False, num_workers=8)

    classifier = PointNetClassifier(normal_channel=with_color)
    classifier.load_model_from_ckpt(cls_config.ckpt_path)
    classifier.cuda()
    classifier.eval()

    with torch.no_grad():

        gen_features = []
        gen_predictions = []
        gen_labels = []
        for i in range(multiple):
            gen_points = base_model.text_condition_generation(text_queries)
            gen_points = misc.fps(gen_points, npoints)
            gen_points = pc_norm(gen_points)
            gen_feature, gen_prediction = classifier.features_and_preds(gen_points)
            gen_features.append(gen_feature)
            gen_predictions.append(gen_prediction)
            gen_labels += text_labels

        gen_features = torch.cat(gen_features, dim=0)
        gen_predictions = torch.cat(gen_predictions, dim=0)
        gen_labels = torch.tensor(gen_labels, dtype=torch.long, device=gen_points.device)
        _, acc = classifier.get_loss_acc(gen_predictions, gen_labels)

        gt_features = []
        for idx, (data, _, _) in enumerate(test_dataloader):
            gt_points = data.cuda()
            gt_points = misc.fps(gt_points, npoints)
            gt_points = pc_norm(gt_points)

            features, _ = classifier.features_and_preds(gt_points)
            gt_features.append(features)
        gt_features = torch.cat(gt_features, dim=0)

    stats_1 = compute_statistics(gen_features.cpu().numpy())
    stats_2 = compute_statistics(gt_features.cpu().numpy())
    p_fid = stats_1.frechet_distance(stats_2)
    p_is = compute_inception_score(gen_predictions.cpu().numpy())

    return acc, p_fid, p_is


def eval_from_npy(path, cls_config, with_color=False):

    gen_data = np.load(path, allow_pickle=True).item()
    gen_labels = gen_data["labels"]
    gen_points = torch.zeros(234, 1024, 3, device="cuda")
    for i in range(234):
        gen_points[i] = torch.tensor(gen_data["points"][i], device="cuda")

    bs = 32
    npoints = cls_config.npoints
    test_dataset = build_dataset_from_cfg(cls_config.dataset.val._base_, cls_config.dataset.val.others)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=bs * 2, shuffle=False,
                                                  drop_last=False, num_workers=8)

    classifier = PointNetClassifier(normal_channel=with_color)
    classifier.load_model_from_ckpt(cls_config.ckpt_path)
    classifier.cuda()
    classifier.eval()

    with torch.no_grad():
        gen_features = []
        gen_predictions = []
        gen_points = pc_norm(gen_points)
        gen_feature, gen_prediction = classifier.features_and_preds(gen_points)
        gen_features.append(gen_feature)
        gen_predictions.append(gen_prediction)

        gen_features = torch.cat(gen_features, dim=0)
        gen_predictions = torch.cat(gen_predictions, dim=0)
        gen_labels = torch.tensor(gen_labels, dtype=torch.long, device=gen_points.device)

        _, acc = classifier.get_loss_acc(gen_predictions, gen_labels)
        gt_features = []
        for idx, (taxonomy_ids, model_ids, data, _, _, _) in enumerate(test_dataloader):
            gt_points = data.cuda()
            gt_points = misc.fps(gt_points, npoints)
            gt_points = pc_norm(gt_points)
            features, _ = classifier.features_and_preds(gt_points)
            gt_features.append(features)
        gt_features = torch.cat(gt_features, dim=0)

    stats_1 = compute_statistics(gen_features.cpu().numpy())
    stats_2 = compute_statistics(gt_features.cpu().numpy())
    p_fid = stats_1.frechet_distance(stats_2)
    p_is = compute_inception_score(gen_predictions.cpu().numpy())

    return acc, p_fid, p_is


if __name__ == "__main__":
    logger = get_logger("classifier")
    with_color = False
    inf_config = cfg_from_yaml_file("cfgs/inference.yaml")
    cls_config = cfg_from_yaml_file("cfgs/classifier.yaml")
    acc, p_fid, p_is = evalulate(inf_config, cls_config, with_color=with_color)
    print_log(f"Acc: {acc} P-FID: {p_fid} P-IS: {p_is}", logger=logger)
