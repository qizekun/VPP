import os
import cv2
import json
import torch
import random
import numpy as np

from .io import IO
import pandas as pd
from utils.logger import *
from .build import DATASETS
import torch.utils.data as data
from utils.transforms import get_transforms


@DATASETS.register_module()
class ShapeNet(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.img_path = config.IMG_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS
        self.multi_view = config.MULTI_VIEW

        self.index_list = {}
        self.text_list = {}
        for index, row in pd.read_json(config.CATEGORY_PATH).iterrows():
            self.index_list["0" + str(row['catalogue'])] = index
            self.text_list["0" + str(row['catalogue'])] = row['describe']

        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        test_data_list_file = os.path.join(self.data_root, 'test.txt')

        self.text_query_file = os.path.join(config.TEXT_PATH)
        assert os.path.exists(self.text_query_file), "text query file not found"
        with open(self.text_query_file, 'r') as f:
            self.text_query = json.load(f)

        self.sample_points_num = config.npoints
        self.whole = config.get('whole')

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger='ShapeNet')
        print_log(f'[DATASET] Open file {self.data_list_file}', logger='ShapeNet')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        if self.whole:
            with open(test_data_list_file, 'r') as f:
                test_lines = f.readlines()
            print_log(f'[DATASET] Open file {test_data_list_file}', logger='ShapeNet')
            lines = test_lines + lines
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger='ShapeNet')

        self.permutation = np.arange(self.npoints)

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc[:, :3], axis=0)
        pc[:, :3] = pc[:, :3] - centroid
        m = np.max(np.sqrt(np.sum(pc[:, :3] ** 2, axis=1)))
        pc[:, :3] = pc[:, :3] / m
        return pc

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc

    def __getitem__(self, idx):
        sample = self.file_list[idx]

        pc = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)
        if self.multi_view:
            path = sample['taxonomy_id'] + '/' + sample['file_path'][9:-4] + '-' + str(random.randint(0, 11)) + '.png'
            img = cv2.imread(os.path.join(self.img_path, path))
        else:
            img = cv2.imread(os.path.join(self.img_path, sample['file_path'].replace(".npy", ".png")))
        img = get_transforms()['train'](img)

        pc = self.random_sample(pc, self.sample_points_num)
        pc = self.pc_norm(pc)
        pc = torch.from_numpy(pc).float()
        # text = self.text_query[sample['file_path'].split('.')[0]]
        text = self.text_list[sample['taxonomy_id']]
        if text[0] in ['a', 'e', 'i', 'o', 'u']:
            text = 'an ' + text
        else:
            text = 'a ' + text

        return pc, img, text

    def __len__(self):
        return len(self.file_list)
