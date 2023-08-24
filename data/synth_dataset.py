from pathlib import Path
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
import random
import pickle
import torch
import sys

import os
from glob import glob
from utils.config import cfg
from torch.utils.data import Dataset
from utils.build_graphs import build_graphs
from torchvision import transforms

# dataset_dir =
# anno_path = cfg.VOC2011.KPT_ANNO_DIR
# img_path = cfg.VOC2011.ROOT_DIR + 'JPEGImages'
# ori_anno_path = cfg.VOC2011.ROOT_DIR + 'Annotations'
# set_path = cfg.VOC2011.SET_SPLIT
# cache_path = cfg.CACHE_PATH


def get_matching_matrix(matching_vec):

    n = matching_vec.shape[0]
    matching_mat = torch.zeros((n, n))
    for i in range(n):
        matching_mat[i][matching_vec[i]] = 1

    return matching_mat

class SynthDataset(Dataset):

    def __init__(self, dir, dst_size):
        super(SynthDataset, self).__init__()
        self.dir = dir
        self.dst_size = dst_size
        self.gt_file_list = glob(os.path.join(dir, '*_gt.txt'))
        self.len = len(self.gt_file_list)

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        P1_gt = np.loadtxt(os.path.join(self.dir, f'pair_{i:04d}_kpts0.txt'))
        P2_gt = np.loadtxt(os.path.join(self.dir, f'pair_{i:04d}_kpts1.txt'))
        gt = np.loadtxt(os.path.join(self.dir, f'pair_{i:04d}_gt.txt')).astype(int)
        with Image.open(os.path.join(self.dir, f'pair_{i:04d}_img0.png')) as img:
            img0 = img.resize(self.dst_size, resample=Image.BICUBIC)

        with Image.open(os.path.join(self.dir, f'pair_{i:04d}_img1.png')) as img:
            img1 = img.resize(self.dst_size, resample=Image.BICUBIC)

        perm_mat = get_matching_matrix(gt)
        n1_gt, n2_gt = P1_gt.shape[0], P2_gt.shape[0]

        G1_gt, H1_gt, e1_gt, edge_indices1, edge_feat1 = build_graphs(P1_gt, n1_gt, stg=cfg.PAIR.GT_GRAPH_CONSTRUCT)
        if cfg.PAIR.REF_GRAPH_CONSTRUCT == 'same':
            G2_gt = perm_mat.transpose().dot(G1_gt)
            H2_gt = perm_mat.transpose().dot(H1_gt)
            e2_gt = e1_gt
        else:
            G2_gt, H2_gt, e2_gt, edge_indices2, edge_feat2 = build_graphs(P2_gt, n2_gt,
                                                                          stg=cfg.PAIR.REF_GRAPH_CONSTRUCT)

        ret_dict = {'Ps': [torch.Tensor(x) for x in [P1_gt, P2_gt]],
                    'ns': [torch.tensor(x) for x in [n1_gt, n2_gt]],
                    'es': [torch.tensor(x) for x in [e1_gt, e2_gt]],
                    'gt_perm_mat': perm_mat,
                    'Gs': [torch.Tensor(x) for x in [G1_gt, G2_gt]],
                    'Hs': [torch.Tensor(x) for x in [H1_gt, H2_gt]],
                    "edge_src": [torch.Tensor(x) for x in [edge_indices1]],
                    "edge_tgt": [torch.Tensor(x) for x in [edge_indices2]],
                    "edge_feat1": [torch.Tensor(x) for x in [edge_feat1]],
                    "edge_feat2": [torch.Tensor(x) for x in [edge_feat2]]
                    }

        imgs = [img0, img1]
        if imgs[0] is not None:
            trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(cfg.NORM_MEANS, cfg.NORM_STD)
            ])
            imgs = [trans(img) for img in imgs]
            ret_dict['images'] = imgs

        return ret_dict

if __name__ == '__main__':
    dataset = SynthDataset('/Ship03/Sources/FeatureMatching/FastQuadMatching/synth_graphs/affine', (256, 256))
    a = dataset.get_pair()
    pass
