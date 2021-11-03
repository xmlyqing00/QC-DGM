import torch
import numpy as np



def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int, eps=0.1):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    # print(u, v)
    # print('z', Z)
    # print('logmu', log_mu)
    # print('lognu', log_nu)
    for _ in range(iters):
        u = eps * (log_mu - torch.logsumexp((Z + v.unsqueeze(1)) / eps, dim=2))
        v = eps * (log_nu - torch.logsumexp((Z + u.unsqueeze(2)) / eps, dim=1))
        # print(u, v)
    return (Z + u.unsqueeze(2) + v.unsqueeze(1)) / eps


def log_sinkhorn(scores: torch.Tensor, dustbin_flag: bool, dustinbin_alpha: torch.Tensor,  iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape

    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    if dustbin_flag:
        # one = scores.new_tensor(1)
        # print('Score', scores.shape, iters)
        # print('One', one.shape)
        # ms, ns = (m*one).to(scores), (n*one).to(scores)

        # print('alpha0', alpha)
        bins0 = dustinbin_alpha.expand(b, m, 1)
        bins1 = dustinbin_alpha.expand(b, 1, n)
        alpha = dustinbin_alpha.expand(b, 1, 1)

        # print('alpha', alpha, bins0.shape, bins1.shape)
        couplings = torch.cat([torch.cat([scores, bins0], -1),
                               torch.cat([bins1, alpha], -1)], 1)

        norm = - (ms + ns).log()
        log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
        log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
        log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

        z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
        z = z - norm  # multiply probabilities by M+N

    else:
        norm = - np.log(m + n)
        log_mu = norm * torch.ones(m, device=scores.device)
        log_nu = norm * torch.ones(n, device=scores.device)
        log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

        z = log_sinkhorn_iterations(scores, log_mu, log_nu, iters)
        z = z - norm  # multiply probabilities by M+N

    return z


def encode_positions(kpts_pos: torch.Tensor, img_size: torch.Tensor):

    # kpts_pos in shape (bs, n, 2)
    tpos = torch.zeros(kpts_pos.shape[:2] + (4,), device=kpts_pos.device)
    kpts_pos = kpts_pos / (img_size / np.pi * 2)
    tpos[:, :, 0] = torch.sin(kpts_pos[:, :, 0])
    tpos[:, :, 1] = torch.cos(kpts_pos[:, :, 0])
    tpos[:, :, 2] = torch.sin(kpts_pos[:, :, 1])
    tpos[:, :, 3] = torch.cos(kpts_pos[:, :, 1])

    return tpos


def quad_matching(scores: torch.Tensor, kptsn: tuple = None, dustbin_flag: bool = False, dustbin_alpha: torch.Tensor = None, iters: int = 3):

    # print(kptsn)
    for i in range(len(kptsn[0])):
        scores[i, kptsn[0][i]:, :] = 0
        scores[i, :, kptsn[1][i]:] = 0
        for j in range(kptsn[0][i], scores.shape[1]):
            scores[i, j, j] = 1
        # print(kptsn[0][i], scores.shape[1])

    # print('filterred', scores[0])

    # z = log_sinkhorn(scores, dustbin_flag, dustbin_alpha, iters)
    z = log_sinkhorn(scores, False, None, iters)
    z = torch.exp(z)  # could be deleted

    # print("scores", scores)
    # print("z", z)
    # print(z[0])
    # print(torch.exp(z)[0])
    # print(v_scores[0])
    # print('\n=========\n')

    # print('\n=========\n')

    return z

