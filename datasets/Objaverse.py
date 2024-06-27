import os
import cv2
import json
import torch
import random
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS
from utils.logger import *
from utils.transforms import get_transforms


@DATASETS.register_module()
class Objaverse(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.img_path = config.IMG_PATH
        self.text_path = config.TEXT_PATH
        self.npoints = config.N_POINTS
        self.ratio = config.ratio
        assert self.npoints in [1024, 2048, 8192, 50000]

        if 'lvis' in self.pc_path:
            data_list_file = os.path.join(self.data_root, 'lvis.json')
            index_list = json.load(open(data_list_file))
            self.extension = 'npy'
        else:
            data_list_file = os.path.join(self.data_root, 'common_ids.txt')
            mata_data = open(data_list_file).readlines()
            index_list = [x.strip() for x in mata_data]
            self.extension = 'npz'
        self.index_list = index_list[: int(len(index_list) * self.ratio)]

        self.sample_points_num = config.npoints
        assert self.sample_points_num <= self.npoints

        self.text_description_dict = json.load(open(self.text_path))

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger='Objaverse')
        print_log(f'[DATASET] Open file {data_list_file}', logger='Objaverse')
        print_log(f'[DATASET] load ratio is {self.ratio}', logger='Objaverse')
        print_log(f'[DATASET] {len(self.index_list)} instances were loaded', logger='Objaverse')

        self.permutation = np.arange(self.npoints)

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc

    def __getitem__(self, idx):
        index = self.index_list[idx]
        pc_path = os.path.join(self.pc_path, f'{index}/{index}_{self.npoints}.{self.extension}')
        pc = IO.get(pc_path).astype(np.float32)
        pc = self.random_sample(pc, self.sample_points_num)
        pc = self.pc_norm(pc)
        pc = torch.from_numpy(pc).float()

        img_index = f'{index}/{str(random.randint(0, 11)).zfill(3)}.png'
        img_path = os.path.join(self.img_path, img_index)
        img = cv2.imread(img_path)
        img = get_transforms()['train'](img)

        text_key = f'/export/einstein-vision/3d_vision/objaverse/render_images_split_100/{img_index}'
        text = random.choice(self.text_description_dict[text_key])

        return pc, img, text

    def __len__(self):
        return len(self.index_list)
