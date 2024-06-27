import torch.nn as nn
from modules.se import SE3d
import torch.nn.functional as F

__all__ = ['VoxelConv', 'ResVoxelConv', 'VoxelDeConv']


class VoxelConv(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=True, with_se=False, bias=True):
        super().__init__()
        stride = 2 if pooling else 1
        voxel_layers = [
            nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=bias),
            nn.BatchNorm3d(out_channels),
         ]
        if with_se:
            voxel_layers.append(SE3d(out_channels))
        self.voxel_layers = nn.Sequential(*voxel_layers)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        voxel_features = self.voxel_layers(x)
        voxel_features = self.relu(voxel_features)
        return voxel_features


class ResVoxelConv(nn.Module):
    def __init__(self, channels, with_se=False):
        super().__init__()
        voxel_layers = [
            nn.Conv3d(channels, channels, 3, stride=1, padding=1),
            nn.BatchNorm3d(channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(channels, channels, 3, stride=1, padding=1),
            nn.BatchNorm3d(channels),
         ]
        if with_se:
            voxel_layers.append(SE3d(channels))
        self.voxel_layers = nn.Sequential(*voxel_layers)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        voxel_features = self.voxel_layers(x) + x
        voxel_features = self.relu(voxel_features)
        return voxel_features


class VoxelDeConv(nn.Module):
    def __init__(self, in_channels, out_channels, scale, with_se=False):
        super(VoxelDeConv, self).__init__()
        self.scale = scale
        voxel_layers = [
            nn.Conv3d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
        ]
        if with_se:
            voxel_layers.append(SE3d(out_channels))
        self.voxel_layers = nn.Sequential(*voxel_layers)
        self.relu = nn.LeakyReLU(0.2, True)

        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale)
        voxel_features = self.voxel_layers(x) + self.shortcut(x)
        voxel_features = self.relu(voxel_features)
        return voxel_features
