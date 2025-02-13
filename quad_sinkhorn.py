import torch
import numpy as np


def sqrt_semi_definite_mat(matrix):
    """Compute the square root of a positive definite matrix."""
    """Url: https://github.com/pytorch/pytorch/issues/25481"""
    _, s, v = matrix.svd()
    good = s > s.max(-1, True).values * s.size(-1) * torch.finfo(s.dtype).eps
    components = good.sum(-1)
    common = components.max()
    unbalanced = common != components.min()
    if common < s.size(-1):
        s = s[..., :common]
        v = v[..., :common]
        if unbalanced:
            good = good[..., :common]
    if unbalanced:
        s = s.where(good, torch.zeros((), device=s.device, dtype=s.dtype))

    return (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)
    # return v * s.sqrt().unsqueeze(-2)


def decompose_sym_mat(mat: torch.Tensor, diag_val: torch.Tensor = None):

    diag_tensor = torch.ones(mat.shape[:2], device=mat.device)
    if diag_val is not None:
        diag_tensor = diag_tensor * diag_val.unsqueeze(1)
    else:
        max_val, _ = mat.abs().sum(2).max(dim=1)
        diag_tensor = diag_tensor * (max_val + 1).unsqueeze(1)
    mat_c = mat + torch.diag_embed(diag_tensor)

    r = sqrt_semi_definite_mat(mat_c)

    return mat_c, r


def log_sinkhorn_iterations2(Z, log_mu, log_nu, iters: int, eps=1):
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


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int, eps=1):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
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
        z = z[:, :-1, :-1]

    else:
        norm = - np.log(m + n)
        log_mu = norm * torch.ones(m, device=scores.device)
        log_nu = norm * torch.ones(n, device=scores.device)
        log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

        z = log_sinkhorn_iterations(scores, log_mu, log_nu, iters)
        z = z - norm  # multiply probabilities by M+N

    z = torch.exp(z)  # could be deleted

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

    # z = log_sinkhorn(scores, False, None, iters)
    z = log_sinkhorn(scores, True, torch.ones(1, device=scores.device) * 0.1, iters)

    # print("scores", scores)
    # print("z", z)
    # print(z[0])
    # print(torch.exp(z)[0])
    # print(v_scores[0])
    # print('\n=========\n')

    # print('\n=========\n')

    return z



def matching(node_s: torch.Tensor, h0: torch.Tensor, h1: torch.Tensor, kptsn: tuple = None,
             edge_w: float = 1, eps: float = 1, iters: int = 5):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = node_s.shape

    if kptsn:
        for i in range(len(kptsn[0])):
            node_s[i, kptsn[0][i]:, :] = 0
            node_s[i, :, kptsn[1][i]:] = 0
            for j in range(kptsn[0][i], n):
                node_s[i, j, j] = 1

    norm = np.log(n)
    log_mu = norm * torch.ones(m, device=node_s.device)
    log_nu = norm * torch.ones(n, device=node_s.device)
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    # print('logmu', log_mu)
    # print('lognu', log_nu)

    edge_const = torch.bmm(h0.sum(dim=2, keepdims=True), h1.sum(dim=2, keepdims=True).transpose(1, 2))

    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)

    for _ in range(iters):
        z = torch.zeros_like(node_s)

        p = torch.exp((z + u.unsqueeze(2) + v.unsqueeze(1)) / eps - norm)
        pn = torch.sign(torch.bmm(torch.bmm(h0.transpose(1, 2), p), h1))
        if ((pn < 0).nonzero().shape[0] > 0):
            # print('pn', pn)
            for k in range(n):
                for l in range(m):
                    z = z + pn[:, k, l].reshape(-1, 1, 1) * torch.bmm(h0[:, :, k:k+1], h1[:, :, l:l+1].transpose(1, 2))
            # print('new_z', z)
            # print('old_z', torch.bmm(h0.sum(dim=2, keepdims=True), h1.sum(dim=2, keepdims=True).transpose(1, 2)))
            z = node_s + edge_w * z
        else:
            z = node_s + edge_w * edge_const

        u = eps * (log_mu - torch.logsumexp((z + v.unsqueeze(1)) / eps, dim=2))
        v = eps * (log_nu - torch.logsumexp((z + u.unsqueeze(2)) / eps, dim=1))
        # print(u, v)

    p = torch.exp((z + u.unsqueeze(2) + v.unsqueeze(1)) / eps - norm)
    return p

