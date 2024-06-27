import torch
import torch.nn as nn
from utils.logger import *
from evals.classifier.pointnet2_cls_ssg import get_model


def pc_norm(pc):
    centroid = torch.mean(pc, dim=1, keepdim=True)
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc ** 2, dim=-1)))
    pc = pc / m
    return pc


class PointNetClassifier(nn.Module):
    def __init__(self, num_class=13, normal_channel=False, width_mult=1, **kwargs):
        super().__init__()
        self.model = get_model(num_class=num_class, normal_channel=normal_channel, width_mult=width_mult)
        self.loss_ce = nn.CrossEntropyLoss()

    def load_model_from_ckpt(self, ckpt_path):
        base_ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = {k.replace("module.", ""): v for k, v in base_ckpt.items()}
        self.load_state_dict(state_dict, strict=True)

        print_log(f'[MaskDream] Successful Loading the ckpt', logger='PointNetClassifier')

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def forward(self, pts):
        logits, _ = self.model(pts.transpose(1, 2))
        return logits

    def features_and_preds(self, pts):
        self.model.eval()
        logits, _, features = self.model(pts.transpose(1, 2), features=True)
        output_features = features
        output_predictions = logits.exp()

        return output_features, output_predictions


class Acc_Metric:
    def __init__(self, acc=0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        elif type(acc).__name__ == 'Acc_Metric':
            self.acc = acc.acc
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict
