import torch
import torch.nn as nn
from torch.autograd import Function
import modules.functional as F
import numpy as np
from skimage.measure import marching_cubes

__all__ = ['Voxelization', 'voxel_to_point', 'BinaryFunction']


class Voxelization(nn.Module):
    def __init__(self, config, normalize=True, eps=0):
        super().__init__()
        self.r = int(config.get('resolution', 24))
        self.normalize = normalize
        self.eps = eps

    def forward(self, pts):
        B, N, C = pts.shape
        if C == 6:
            coords = pts[:, :, :3].transpose(1, 2)
            features = pts[:, :, 3:].transpose(1, 2)
            features = torch.cat([torch.ones(B, 1, N, device=coords.device), features], dim=1)
        elif C == 3:
            coords = pts.transpose(1, 2)
            features = torch.ones(B, 1, N, device=coords.device)
        else:
            raise ValueError("Channel Error!")

        coords = coords.detach()
        norm_coords = coords - coords.mean(2, keepdim=True)
        if self.normalize:
            norm_coords = norm_coords / (
                        norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + self.eps) + 0.5
        else:
            norm_coords = (norm_coords + 1) / 2.0
        norm_coords = torch.clamp(norm_coords * self.r, 0, self.r - 1)
        vox_coords = torch.round(norm_coords).to(torch.int32)
        return F.avg_voxelize(features, vox_coords, self.r)

    def extra_repr(self):
        return 'resolution={}{}'.format(self.r, ', normalized eps = {}'.format(self.eps) if self.normalize else '')


# def voxel_to_point(voxel):
#     C, R, _, _ = voxel.shape
#     device = voxel.device
#     voxel = voxel.detach().cpu().numpy()
#     if C == 1:
#         pts, _, _, _ = marching_cubes(voxel[0], level=0.5)
#     elif C == 4:
#         coords, _, _, _ = marching_cubes(voxel[0], level=0.5)
#         x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
#         x, y, z = x.astype(int), y.astype(int), z.astype(int)
#         color = voxel[1:, x, y, z]
#         pts = torch.cat([coords, color.transpose(1, 0)], dim=1)
#     else:
#         print("The shape of voxel is {}".format(voxel.shape))
#         raise NotImplementedError
#     pts = pc_norm(pts)
#     pts = torch.from_numpy(pts.copy()).to(device)
#
#     return pts


def voxel_to_point(voxel):
    C, R, _, _ = voxel.shape
    if C == 1:
        indices = torch.nonzero(voxel[0] > 0.5)
        pts = indices / R * 2 - 1
    elif C == 4:
        indices = torch.nonzero(voxel[0] > 0.5)
        index = indices[:, 0] * R * R + indices[:, 1] * R + indices[:, 2]
        values = voxel.view(C, -1).transpose(0, 1)[index]
        values = values[:, 1:]
        indices = indices / R * 2 - 1
        pts = torch.cat([indices, values], dim=1)
    else:
        print("The shape of voxel is {}".format(voxel.shape))
        raise NotImplementedError
    return pts


def pc_norm(pc):
    """ pc: NxC, return NxC """
    centroid = np.mean(pc[:, :3], axis=0)
    pc[:, :3] = pc[:, :3] - centroid
    m = np.max(np.sqrt(np.sum(pc[:, :3] ** 2, axis=1)))
    pc[:, :3] = pc[:, :3] / m
    return pc


class BinaryFunction(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        ones = torch.ones_like(x, requires_grad=True, device=x.device)
        zeros = torch.zeros_like(x, requires_grad=True, device=x.device)
        output = torch.where(x >= 0.5, ones, zeros)
        return output

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        input_grad = x - 0.5
        return input_grad * grad
