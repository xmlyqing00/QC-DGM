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


import quad_sinkhorn

def eval_model(model, dataloader, eval_epoch=None, verbose=False, quad_sinkhorn_flag=False):
    print('Start evaluation...')
    since = time.time()

    device = next(model.parameters()).device

    if eval_epoch is not None:
        model_path = str(Path(cfg.OUTPUT_PATH) / 'params' / 'params_{:04}.pt'.format(eval_epoch))
        print('Loading model parameters from {}'.format(model_path))
        load_model(model, model_path)

    was_training = model.training
    model.eval()

    ds = dataloader.dataset
    classes = ds.classes
    cls_cache = ds.clss

    lap_solver = hungarian

    accs = torch.zeros(len(classes)).cuda()
    f1s = torch.zeros(len(classes)).cuda()
    pcs = torch.zeros(len(classes)).cuda()
    rcl = torch.zeros(len(classes)).cuda()

    total_num = 0

    for i, cls in enumerate(classes):
        if verbose:
            print('Evaluating class {}: {}/{}'.format(cls, i, len(classes)))

        running_since = time.time()
        iter_num = 0

        ds.clss = cls
        acc_match_num = torch.zeros(1).cuda()
        acc_total_num = torch.zeros(1).cuda()
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
            e1_gt, e2_gt = [_.cuda() for _ in inputs['es']]
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
            total_num += batch_num

            iter_num = iter_num + 1

            with torch.set_grad_enabled(False):
                s_pred, U_src, F_src, U_tgt, F_tgt, AA, BB = \
                    model(data1, data2, P1_gt, P2_gt, G1_gt, G2_gt, H1_gt, H2_gt, n1_gt, n2_gt, KG, KH,  edge_src, edge_tgt, edge_feat1, edge_feat2, perm_mat, inp_type)

            # print(P1_gt.shape)
            # dist1 = torch.cdist(P1_gt, P1_gt)
            # dist2 = torch.cdist(P2_gt, P2_gt)

            # print('dist1', dist1[0])
            #
            # ea = U_src
            # eb = F_src
            # q = torch.bmm(ea, torch.bmm(perm_mat, eb))
            # print('q', q[0])
            # # q2 = torch.bmm(torch.bmm(perm_mat, eb), perm_mat.transpose(1, 2))
            # q1 = torch.bmm(perm_mat, eb)
            # q2 = torch.bmm(q1, perm_mat.transpose(1, 2))
            # d = q2 - ea
            # print(perm_mat[0])
            # print('d', d[0])
            # print('q2', q2[0])
            #
            # print('eb', eb[0])
            # print('ea', ea[0])
            # # print(torch.exp(z)[0])
            # # print(v_scores[0])
            # print('\n=========\n')

            A_src = torch.bmm(G1_gt, H1_gt.transpose(1, 2))
            A_tgt = torch.bmm(G2_gt, H2_gt.transpose(1, 2))

            lb = 0.1
            if quad_sinkhorn_flag:
                A_c, A_r = quad_sinkhorn.decompose_sym_mat(A_src)
                B_c, B_r = quad_sinkhorn.decompose_sym_mat(A_tgt)

                edge_s = torch.bmm(A_r.sum(dim=2, keepdims=True), B_r.sum(dim=2, keepdims=True).transpose(1, 2))
                scores = s_pred + lb * edge_s.abs()
                s_pred = quad_sinkhorn.log_sinkhorn(scores, False, None, 5)
                Xnew = lap_solver(s_pred, n1_gt, n2_gt)
                s_pred_perm = Xnew
            else:

                for miter in range(10):
                   X = qc_opt(A_src, A_tgt, s_pred, Xnew, lb)
                   Xnew = lap_solver(X, n1_gt, n2_gt)
                s_pred_perm = lap_solver(Xnew, n1_gt, n2_gt)

            _, _acc_match_num, _acc_total_num = matching_accuracy(s_pred_perm, perm_mat, n1_gt)
            acc_match_num += _acc_match_num
            acc_total_num += _acc_total_num

            # print('s_pred')
            # print(s_pred[0])
            # print('XNew')
            # print(Xnew[0])
            # print('gt')
            # print(perm_mat[0])
            # print('Xnew', torch.argmax(Xnew[0], dim=1))
            # print('GT', torch.argmax(perm_mat[0], dim=1))
            # print('\n===================\n')

            print(acc_match_num / acc_total_num, acc_total_num,  _acc_match_num, _acc_total_num)

            #
            # if iter_num % cfg.STATISTIC_STEP == 0 and verbose:
            #     running_speed = cfg.STATISTIC_STEP * batch_num / (time.time() - running_since)
            #     print(f'Total numbers {cfg.STATISTIC_STEP * batch_num}, total time {time.time() - running_since}')
            #     print('Class {:<8} Iteration {:<4} {:>4.2f}sample/s'.format(cls, iter_num, running_speed))
            #     running_since = time.time()

        accs[i] = acc_match_num / acc_total_num
        if verbose:
            print('Class {} acc = {:.4f}'.format(cls, accs[i]))

    time_elapsed = time.time() - since
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Total number', total_num, 'total seconds', time_elapsed)

    model.train(mode=was_training)
    ds.clss = cls_cache
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

    image_dataset = GMDataset(cfg.DATASET_FULL_NAME,
                              sets='test',
                              length=cfg.EVAL.SAMPLES,
                              obj_resize=cfg.PAIR.RESCALE)
    bs = 512
    if args.local:
        bs = cfg.BATCH_SIZE
    dataloader = get_dataloader(image_dataset, bs=bs)

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
        classes = dataloader.dataset.classes
        pcks = eval_model(model, dataloader,
                          eval_epoch=cfg.EVAL.EPOCH if cfg.EVAL.EPOCH != 0 else None,
                          verbose=True, quad_sinkhorn_flag=args.quad_sinkhorn)
