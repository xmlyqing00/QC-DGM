import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from utils.sh_layer import sh
from utils.sm_layer import sm
from utils.feature_align import feature_align
from QCDGM.gconv import Gconv_layer
from QCDGM.affinity_layer import Affinity2
from utils.hungarian import hungarian
import torch.nn.functional as F
from utils.config import cfg
import utils.backbone
CNN = eval('utils.backbone.{}'.format(cfg.BACKBONE))

import quad_sinkhorn

class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()
        self.sh_layer = sh(max_iter=cfg.QCDGM.BS_ITER_NUM, epsilon=cfg.QCDGM.BS_EPSILON)
        self.sm_layer = sm(alpha=cfg.QCDGM.SCALE_ALPHA)
        self.l2norm = nn.LocalResponseNorm(cfg.QCDGM.FEATURE_CHANNEL * 2, alpha=cfg.QCDGM.FEATURE_CHANNEL * 2, beta=0.5, k=0)
        self.gnn_layer = cfg.QCDGM.GNN_LAYER
        self.softmax = nn.Softmax(dim=-1)

        self.quad_sinkhorn_flag = False

        self.gnn_layer0 = Gconv_layer(cfg.QCDGM.FEATURE_CHANNEL * 2 + 2, cfg.QCDGM.GNN_FEAT + 2)
        self.affinity_layer0 = Affinity2(cfg.QCDGM.GNN_FEAT + 2)

        self.gnn_layer1 = Gconv_layer(cfg.QCDGM.GNN_FEAT * 2 + 4, cfg.QCDGM.GNN_FEAT + 2)
        self.affinity_layer1 = Affinity2(cfg.QCDGM.GNN_FEAT + 2)

        self.w_edge = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w_pos = nn.Parameter(torch.ones(1), requires_grad=True)


    def forward(self, src, tgt, P_src, P_tgt, G_src, G_tgt, H_src, H_tgt, ns_src, ns_tgt, K_G, K_H, edge_src, edge_tgt, edge_feat1, edge_feat2, perm_mat, type='img'):
        if type == 'img' or type == 'image':
            # extract feature. src shape: bs, 3, 256, 256
            src_node = self.node_layers(src)  # bs, 512, 32, 32
            src_edge = self.edge_layers(src_node)  # bs, 512, 16, 16
            tgt_node = self.node_layers(tgt)
            tgt_edge = self.edge_layers(tgt_node)

            # feature normalization
            src_node = self.l2norm(src_node)
            src_edge = self.l2norm(src_edge)
            tgt_node = self.l2norm(tgt_node)
            tgt_edge = self.l2norm(tgt_edge)

            # arrange features
            U_src = feature_align(src_node, P_src, ns_src, cfg.PAIR.RESCALE)
            F_src = feature_align(src_edge, P_src, ns_src, cfg.PAIR.RESCALE)
            U_tgt = feature_align(tgt_node, P_tgt, ns_tgt, cfg.PAIR.RESCALE)
            F_tgt = feature_align(tgt_edge, P_tgt, ns_tgt, cfg.PAIR.RESCALE)
        elif type == 'feat' or type == 'feature':
            U_src = src[:, :src.shape[1] // 2, :]
            F_src = src[:, src.shape[1] // 2:, :]
            U_tgt = tgt[:, :tgt.shape[1] // 2, :]
            F_tgt = tgt[:, tgt.shape[1] // 2:, :]
        else:
            raise ValueError('unknown type string {}'.format(type))

        A_src = torch.bmm(G_src, H_src.transpose(1, 2))
        A_tgt = torch.bmm(G_tgt, H_tgt.transpose(1, 2))
        P1_src = torch.zeros_like(P_src)
        P2_tgt = torch.zeros_like(P_tgt)
        for k in range(P_src.shape[0]):
            for i in range(P_src.shape[1]):
               for j in range(P_tgt.shape[2]):
                  if torch.norm(P_src[k, i, :]) == 0:
                      P1_src[k, i, j] = 0
                      P2_tgt[k, i, j] = 0
                  else:
                      P1_src[k, i, j] = P_src[k, i, j]/torch.norm(P_src[k, i, :])
                      P2_tgt[k, i, j] = P_tgt[k, i, j]/torch.norm(P_tgt[k, i, :])     
        
        ## Node embedding with unary geometric prior
        emb1, emb2 = torch.cat((U_src, F_src, P1_src.transpose(1,2)), dim=1).transpose(1, 2), torch.cat((U_tgt, F_tgt, P2_tgt.transpose(1,2)), dim=1).transpose(1, 2)

        emb1, emb2 = self.gnn_layer0([A_src, emb1], [A_tgt, emb2])
        s = self.affinity_layer0(emb1, emb2)

        emb1_normed = emb1 / torch.norm(emb1, dim=2, keepdim=True)
        emb2_normed = emb2 / torch.norm(emb2, dim=2, keepdim=True)

        AA_src = torch.einsum('nlc, nsc -> nls', emb1_normed, emb1_normed)
        BB_tgt = torch.einsum('nlc, nsc -> nls', emb2_normed, emb2_normed)
        # AA_src = torch.mul(torch.exp(AA), A_src)
        # BB_tgt = torch.mul(torch.exp(BB), A_tgt)
        # AA_src = AA
        # BB_tgt = BB

        ## Normalization in evaluation
        # if self.training == False:
        #     for b in range(s.shape[0]):
        #         s[b, :, :] = s[b, :, :].clone() / torch.max(s[b, :, :].clone())
        #
        s = self.sm_layer(s, ns_src, ns_tgt)
        # s = self.sh_layer(s, ns_src, ns_tgt)

        emb1_new = torch.cat((emb1, torch.bmm(s, emb2)), dim=-1)
        emb2_new = torch.cat((emb2, torch.bmm(s.transpose(1, 2), emb1)), dim=-1)
        emb1, emb2 = self.gnn_layer1([AA_src, emb1_new], [BB_tgt, emb2_new])
        s_vertex = self.affinity_layer1(emb1, emb2)

        emb1_normed = torch.matmul(emb1 / torch.norm(emb1, dim=2, keepdim=True), self.affinity_layer1.A)
        emb2_normed = torch.matmul(emb2 / torch.norm(emb2, dim=2, keepdim=True), self.affinity_layer1.A)

        s_edge = torch.einsum('nlc, nsc -> nls', emb1_normed.sum(dim=2, keepdims=True),
                              emb2_normed.sum(dim=2, keepdims=True))
        s_edge /= emb2_normed.shape[2]

        img_size = torch.tensor(src.shape[-2:], dtype=torch.int, device=src.device)
        tpos0 = quad_sinkhorn.encode_positions(P_src, img_size)
        tpos1 = quad_sinkhorn.encode_positions(P_src, img_size)

        s_pos = torch.einsum('nlc, nsc -> nls', tpos0.sum(dim=2, keepdims=True), tpos1.sum(dim=2, keepdims=True))

        scores = s_vertex + self.w_edge * s_edge + self.w_pos * s_pos
        # scores = s + edge_s
        z = quad_sinkhorn.quad_matching(scores, (ns_src, ns_tgt), iters=5)

        ## Normalization in evaluation

        # if self.training == False:
        #     for b in range(s.shape[0]):
        #         s[b, :, :] = s[b, :, :].clone() / torch.max(s[b, :, :].clone())
        #
        z = self.sm_layer(z, ns_src, ns_tgt)
        # s = self.sh_layer(s, ns_src, ns_tgt)

        return z, None, None, None, None, None, None

