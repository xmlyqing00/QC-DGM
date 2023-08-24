import torch
import time
from datetime import datetime
from pathlib import Path
from utils.qc_opt import qc_opt
from utils.hungarian import hungarian
from data.data_loader import GMDataset, get_dataloader
from utils.evaluation_metric import matching_accuracy, f1_score
from parallel import DataParallel
from utils.model_sl import load_model
import torch.nn.functional as F
from utils.config import cfg
import numpy as np

from data.synth_dataset import SynthDataset

import os
import cv2
import quad_sinkhorn


def plot_matching(img0, img1, kpts0, kpts1, matching, gt=None, linecolor=[(0, 200, 0), (0, 0, 200)]):

    assert img0.shape == img1.shape

    h, w = img0.shape[:2]
    gap = 20
    offset = w + gap

    canvas = np.ones((h, w * 2 + gap, 3), dtype=np.uint8) * 128
    canvas[:h, :w] = img0
    canvas[:h, offset:offset + w] = img1
    canvas[0] = 0
    canvas[-1] = 0
    canvas[:, 0] = 0
    canvas[:, -1] = 0
    canvas[:, w] = 0
    canvas[:, offset] = 0

    kpts0 = np.round(kpts0).astype(int)
    kpts1 = np.round(kpts1).astype(int)

    for i in range(kpts0.shape[0]):
        kpt_id = matching[i]
        if kpt_id == gt[i]:
            cv2.line(canvas, tuple(kpts0[i]), (kpts1[kpt_id][0] + offset, kpts1[kpt_id][1]), linecolor[0])
        else:
            cv2.line(canvas, tuple(kpts0[i]), (kpts1[kpt_id][0] + offset, kpts1[kpt_id][1]), linecolor[1])

    return canvas




def eval_model(model, dataloader, eval_epoch=None, verbose=False, quad_sinkhorn_flag=False):
    print('Start evaluation...')

    device = next(model.parameters()).device

    if eval_epoch is not None:
        model_path = str(Path(cfg.OUTPUT_PATH) / 'params' / 'params_{:04}.pt'.format(eval_epoch))
        print('Loading model parameters from {}'.format(model_path))
        load_model(model, model_path)

    # was_training = model.training
    model.eval()

    # ds = dataloader.dataset
    # classes = ds.classes
    # cls_cache = ds.clss
    classes = ['synth']

    lap_solver = hungarian

    accs = torch.zeros(len(classes)).cuda()
    # f1s = torch.zeros(len(classes)).cuda()
    # pcs = torch.zeros(len(classes)).cuda()
    # rcl = torch.zeros(len(classes)).cuda()

    total_num = 0
    pred_time = []

    edge_opt = 2
    plot_flag = True
    norm_mean = torch.tensor(cfg.NORM_MEANS).reshape(1, 3, 1, 1)
    norm_std = torch.tensor(cfg.NORM_STD).reshape(1, 3, 1, 1)
    if quad_sinkhorn_flag:
        out_dir = os.path.join(subset_name, f'quad_sinkhorn_{edge_opt}')
    else:
        out_dir = os.path.join(subset_name, f'{cfg.MODEL_NAME}_{edge_opt}')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for i, cls in enumerate(classes):
        if verbose:
            print('Evaluating class {}: {}/{}'.format(cls, i, len(classes)))

        running_since = time.time()
        cls_total_num = 0

        # ds.clss = cls
        acc_match_num = torch.zeros(1).cuda()
        acc_total_num = torch.zeros(1).cuda()
        iter_num = 0


        for inputs in dataloader:
            if 'images' in inputs:
                data1, data2 = [_.cuda() for _ in inputs['images']]
                inp_type = 'img'
            elif 'features' in inputs:
                data1, data2 = [_.cuda() for _ in inputs['features']]
                inp_type = 'feat'
            else:
                raise ValueError('no valid data key (\'images\' or \'features\') found from dataloader!')
            P1_gt, P2_gt = [_.cuda() for _ in inputs['Ps']]
            n1_gt, n2_gt = [_.cuda() for _ in inputs['ns']]
            # e1_gt, e2_gt = [_.cuda() for _ in inputs['es']]
            G1_gt, G2_gt = [_.cuda() for _ in inputs['Gs']]
            H1_gt, H2_gt = [_.cuda() for _ in inputs['Hs']]
            KG, KH = [_.cuda() for _ in inputs['Ks']]
            edge_src = [_.cuda() for _ in inputs['edge_src']]
            edge_tgt = [_.cuda() for _ in inputs['edge_tgt']]
            edge_feat1 = [_.cuda() for _ in inputs['edge_feat1']]
            edge_feat2 = [_.cuda() for _ in inputs['edge_feat2']]
            perm_mat = inputs['gt_perm_mat'].cuda()

            # print(torch.dist(edge_map0_c, torch.bmm(edge_map0_r, edge_map0_r.transpose(1, 2))))
            # print(torch.dist(edge_map1_c, torch.bmm(edge_map1_r, edge_map1_r.transpose(1, 2))))

            # print(perm_mat[0])

            batch_num = data1.size(0)
            cls_total_num += batch_num

            if edge_opt > 0:
                with torch.set_grad_enabled(False):
                    if quad_sinkhorn_flag:
                        if edge_opt == 1:
                            d0 = torch.bmm(G1_gt, H1_gt.transpose(1, 2))
                            d1 = torch.bmm(G2_gt, H2_gt.transpose(1, 2))
                        else:
                            img_size = torch.tensor(data1.shape[-2:]).reshape(1, 1, 2).to(device=P1_gt.device)
                            P1_gt = P1_gt / img_size
                            P2_gt = P2_gt / img_size
                            d0 = torch.cdist(P1_gt, P1_gt)
                            d1 = torch.cdist(P2_gt, P2_gt)
                        max_val0, _ = d0.abs().sum(2).max(dim=1)
                        max_val1, _ = d1.abs().sum(2).max(dim=1)
                        diag_val = torch.max(max_val0, max_val1)
                        d0_c, h0 = quad_sinkhorn.decompose_sym_mat(d0, diag_val)
                        d1_c, h1 = quad_sinkhorn.decompose_sym_mat(d1, diag_val)

                        scores = torch.zeros_like(d0)
                        s_pred = quad_sinkhorn.matching(scores, h0, h1, (n1_gt, n2_gt), edge_w=1, eps=1, iters=5)
                        s_pred_perm = lap_solver(s_pred, n1_gt, n2_gt)

                    else:
                        if edge_opt == 1:
                            A_src = torch.bmm(G1_gt, H1_gt.transpose(1, 2))
                            A_tgt = torch.bmm(G2_gt, H2_gt.transpose(1, 2))
                        else:
                            img_size = torch.tensor(data1.shape[-2:]).reshape(1, 1, 2).to(device=P1_gt.device)
                            P1_gt = P1_gt / img_size
                            P2_gt = P2_gt / img_size
                            A_src = torch.cdist(P1_gt, P1_gt)
                            A_tgt = torch.cdist(P2_gt, P2_gt)

                        Xnew = torch.zeros_like(A_src)
                        s_pred = torch.zeros_like(A_src)
                        lb = 1

                        for miter in range(10):
                            X = qc_opt(A_src, A_tgt, s_pred, Xnew, lb)
                            Xnew = lap_solver(X, n1_gt, n2_gt)
                        s_pred_perm = lap_solver(Xnew, n1_gt, n2_gt)
            else:
                with torch.set_grad_enabled(False):
                    s_pred, U_src, F_src, U_tgt, F_tgt, AA, BB = \
                        model(data1, data2, P1_gt, P2_gt, G1_gt, G2_gt, H1_gt, H2_gt, n1_gt, n2_gt, KG, KH,  edge_src, edge_tgt, edge_feat1, edge_feat2, perm_mat, inp_type)

                    A_src = torch.bmm(G1_gt, H1_gt.transpose(1, 2))
                    A_tgt = torch.bmm(G2_gt, H2_gt.transpose(1, 2))

                    lb = 0.1
                    if quad_sinkhorn_flag:
                        max_val0, _ = A_src.abs().sum(2).max(dim=1)
                        max_val1, _ = A_tgt.abs().sum(2).max(dim=1)
                        diag_val = torch.max(max_val0, max_val1)
                        A_c, A_r = quad_sinkhorn.decompose_sym_mat(A_src, diag_val)
                        B_c, B_r = quad_sinkhorn.decompose_sym_mat(A_tgt, diag_val)
                        #
                        s_pred = quad_sinkhorn.matching(s_pred, A_r, B_r, (n1_gt, n2_gt), edge_w=10, eps=10, iters=5)
                        # edge_s = torch.bmm(A_r.sum(dim=2, keepdims=True), B_r.sum(dim=2, keepdims=True).transpose(1, 2))
                        # scores = s_pred + 10 * edge_s.abs()
                        # s_pred = quad_sinkhorn.log_sinkhorn(scores, False, None, 5)
                        s_pred_perm = lap_solver(s_pred, n1_gt, n2_gt)
                    else:
                        Xnew = lap_solver(s_pred, n1_gt, n2_gt)
                        for miter in range(10):
                           X = qc_opt(A_src, A_tgt, s_pred, Xnew, lb)
                           Xnew = lap_solver(X, n1_gt, n2_gt)
                        s_pred_perm = lap_solver(Xnew, n1_gt, n2_gt)
                        #
                        # s_pred_perm = lap_solver(s_pred, n1_gt, n2_gt)

            _, _acc_match_num, _acc_total_num = matching_accuracy(s_pred_perm, perm_mat, n1_gt)
            acc_match_num += _acc_match_num
            acc_total_num += _acc_total_num

            iter_num += 1
            if plot_flag:
                imgs0 = ((inputs['images'][0] * norm_std + norm_mean).numpy() * 255).astype(np.uint8).transpose(
                    [0, 2, 3, 1])[..., ::-1]
                imgs1 = ((inputs['images'][1] * norm_std + norm_mean).numpy() * 255).astype(np.uint8).transpose(
                    [0, 2, 3, 1])[..., ::-1]
                kpts0 = inputs['Ps'][0].cpu().numpy()
                kpts1 = inputs['Ps'][1].cpu().numpy()
                matching_vec = torch.argmax(s_pred_perm, dim=2)
                gt_vec = torch.argmax(perm_mat, dim=2)
                for j in range(batch_num):
                    idx = (iter_num - 1) * batch_num + j

                    canvas = plot_matching(imgs0[j], imgs1[j], kpts0[j], kpts1[j], matching_vec[j], gt_vec[j])
                    save_path = os.path.join(out_dir, f'{idx:04d}.png')
                    cv2.imwrite(save_path, canvas)

                print(acc_match_num / acc_total_num, acc_total_num, _acc_match_num, _acc_total_num)


            # print('s_pred')
            # print(s_pred[0])
            # print('XNew')
            # print(Xnew[0])
            # print('gt')
            # print(perm_mat[0])
            # print('Xnew', torch.argmax(Xnew[0], dim=1))
            # print('GT', torch.argmax(perm_mat[0], dim=1))
            # print('\n===================\n')

            #
            # if verbose:
            #     print(acc_match_num / acc_total_num, acc_total_num, _acc_match_num, _acc_total_num)
            #     running_speed = total_num / (time.time() - running_since)
            #     print('fps', running_speed)

        pred_time.append((time.time() - running_since) / cls_total_num)
        total_num += cls_total_num
        accs[i] = acc_match_num / acc_total_num
        if verbose:
            print(f'Class {cls}\t\t acc = {accs[i]:.4f}\t\t time = {pred_time[-1]:.4f}')

    # print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Total number', total_num, f'avg time = {np.array(pred_time).mean():.4f}')

    # model.train(mode=was_training)
    # ds.clss = cls_cache
    print('Matching accuracy')
    for cls, single_acc in zip(classes, accs):
       print('{} = {:.4f}'.format(cls, single_acc))
    print('average = {:.4f}'.format(torch.mean(accs)))
    
    
    return accs


if __name__ == '__main__':
    from utils.dup_stdout_manager import DupStdoutFileManager
    from utils.parse_args import parse_args
    from utils.print_easydict import print_easydict

    args = parse_args('Deep learning of graph matching evaluation code.')

    import importlib
    mod = importlib.import_module(cfg.MODULE)
    Net = mod.Net

    torch.set_printoptions(precision=4, threshold=2000, linewidth=130, sci_mode=False)

    torch.manual_seed(cfg.RANDOM_SEED)

    subset_name = 'affine_1.0'
    image_dataset = SynthDataset(f'/Ship03/Sources/FeatureMatching/FastQuadMatching/synth_graphs/{subset_name}', (500, 500))
    bs = args.bs
    print('arg.bs', args.bs)
    if args.local:
        bs = cfg.BATCH_SIZE
    dataloader = get_dataloader(image_dataset, bs=16)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net()
    model = model.cuda()
    model.quad_sinkhorn_flag = args.quad_sinkhorn

    if args.local:
        model = DataParallel(model, device_ids=[0])
    else:
        model = DataParallel(model, device_ids=range(torch.cuda.device_count()))

    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)
    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    with DupStdoutFileManager(str(Path(cfg.OUTPUT_PATH) / ('eval_log_' + now_time + '.log'))) as _:
        print_easydict(cfg)
        pcks = eval_model(model, dataloader,
                          eval_epoch=cfg.EVAL.EPOCH if cfg.EVAL.EPOCH != 0 else None,
                          verbose=True, quad_sinkhorn_flag=args.quad_sinkhorn)
