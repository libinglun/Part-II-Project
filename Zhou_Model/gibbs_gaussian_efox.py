## This file is the code for direct-assignment sampler with with 1d Gaussian observation with unknown mean and known var for S-HDP-HMM and HDP-HMM
import sys

import numpy as np
from numpy.random import choice, normal, dirichlet, beta, gamma, multinomial, exponential, binomial, uniform
import scipy.stats as ss
import scipy.special as ssp
from munkres import Munkres, print_matrix

eps = 1e-6


def transform(concentration, stick_ratio):
    rho0 = concentration * stick_ratio
    alpha0 = concentration - rho0
    return rho0, alpha0


def sample_one_step_ahead(zt, yt, n_mat, ysum, ycnt, beta_vec, beta_new, alpha0, gamma0, sigma0, mu0, sigma0_pri, rho0,
                          K):
    T = len(zt)

    ########### last time point ##########
    for t in range(1, T):
        j = zt[t - 1]

        ## conpute posterior distributions
        tmp_vec = np.arange(K)
        zt_dist = (alpha0 * beta_vec + n_mat[j] + rho0 * (j == tmp_vec)) / (alpha0 + n_mat[j].sum() + rho0)
        knew_dist = alpha0 * beta_new / (alpha0 + n_mat[j].sum() + rho0)

        ## compute y marginal likelihood
        varn = 1 / (1 / (sigma0_pri ** 2) + ycnt / (sigma0 ** 2))
        mun = ((mu0 / (sigma0_pri ** 2)) + (ysum / (sigma0 ** 2))) * varn

        yt_dist = ss.norm.pdf(yt[t], mun, np.sqrt((sigma0 ** 2) + varn))
        yt_knew_dist = ss.norm.pdf(yt[t], mu0, np.sqrt((sigma0 ** 2) + (sigma0_pri ** 2)))

        ## construct z's posterior by cases
        post_cases = np.hstack((zt_dist * yt_dist, knew_dist * yt_knew_dist))
        post_cases = post_cases / (post_cases.sum())

        ## sample zt
        zt[t] = np.where(multinomial(1, post_cases))[0][0]

        ## update beta_vec, n_mat when having a new state
        if zt[t] == K:
            b = beta(1, gamma0, size=1)
            beta_vec = np.hstack((beta_vec, b * beta_new))
            beta_new = (1 - b) * beta_new
            n_mat = np.hstack((n_mat, np.zeros((K, 1))))
            n_mat = np.vstack((n_mat, np.zeros((1, K + 1))))
            ysum = np.hstack((ysum, 0))
            ycnt = np.hstack((ycnt, 0))
            K += 1

        ## update n_mat
        n_mat[j, zt[t]] += 1
        ysum[zt[t]] += yt[t]
        ycnt[zt[t]] += 1
    return zt, n_mat, ysum, ycnt, beta_vec, beta_new, K


## initialization
def init_gibbs(rho0, alpha0, gamma0, sigma0, mu0, sigma0_pri, T, yt):
    K = 1
    zt = np.zeros(T, dtype='int')
    beta_vec = dirichlet(np.array([1, gamma0]), size=1)[0]
    beta_new = beta_vec[-1]
    beta_vec = beta_vec[:-1]
    n_mat = np.array([[0]])  # t = 0 count as wt=0, don't need to infer wt
    ysum = np.array([yt[0]])
    ycnt = np.array([1])

    zt, n_mat, ysum, ycnt, beta_vec, beta_new, K = sample_one_step_ahead(zt, yt, n_mat, ysum, ycnt, beta_vec, beta_new,
                                                                         alpha0, gamma0, sigma0, mu0, sigma0_pri, rho0,
                                                                         K)
    return rho0, alpha0, gamma0, sigma0, mu0, sigma0_pri, K, zt, beta_vec, beta_new, n_mat, ysum, ycnt


def init_gibbs_full_bayesian(alpha0_a_pri, alpha0_b_pri, gamma0_a_pri, gamma0_b_pri, c_pri, d_pri, sigma0, mu0,
                             sigma0_pri, T, yt):
    K = 1
    zt = np.zeros(T, dtype='int')

    concentration = gamma(alpha0_a_pri, 1 / alpha0_b_pri)
    gamma0 = gamma(gamma0_a_pri, 1 / gamma0_b_pri)
    stick_ratio = beta(c_pri, d_pri)
    rho0, alpha0 = transform(concentration, stick_ratio)

    beta_vec = dirichlet(np.array([1, gamma0]), size=1)[0]
    beta_new = beta_vec[-1]
    beta_vec = beta_vec[:-1]

    n_mat = np.array([[0]])  # t = 0 count as wt=0, don't need to infer wt, t=T will add one
    ysum = np.array([yt[0]])
    ycnt = np.array([1])

    zt, n_mat, ysum, ycnt, beta_vec, beta_new, K = sample_one_step_ahead(zt, yt, n_mat, ysum, ycnt, beta_vec, beta_new,
                                                                         alpha0, gamma0, sigma0, mu0, sigma0_pri, rho0,
                                                                         K)

    return rho0, alpha0, gamma0, sigma0, mu0, sigma0_pri, K, zt, beta_vec, beta_new, n_mat, ysum, ycnt


def init_gibbs_full_bayesian_regular(alpha0_a_pri, alpha0_b_pri, gamma0_a_pri, gamma0_b_pri, sigma0, mu0, sigma0_pri, T,
                                     yt):
    K = 1
    zt = np.zeros(T, dtype='int')

    alpha0 = gamma(alpha0_a_pri, 1 / alpha0_b_pri)
    gamma0 = gamma(gamma0_a_pri, 1 / gamma0_b_pri)

    beta_vec = dirichlet(np.array([1, gamma0]), size=1)[0]
    beta_new = beta_vec[-1]
    beta_vec = beta_vec[:-1]

    n_mat = np.array([[0]])  # t = 0 count as wt=0, don't need to infer wt, t=T will add one
    ysum = np.array([yt[0]])
    ycnt = np.array([1])

    zt, n_mat, ysum, ycnt, beta_vec, beta_new, K = sample_one_step_ahead(zt, yt, n_mat, ysum, ycnt, beta_vec, beta_new,
                                                                         alpha0, gamma0, sigma0, mu0, sigma0_pri, 0, K)

    return alpha0, gamma0, sigma0, mu0, sigma0_pri, K, zt, beta_vec, beta_new, n_mat, ysum, ycnt


def sample_last(zt, yt, n_mat, ysum, ycnt, beta_vec, beta_new, alpha0, gamma0, sigma0, mu0, sigma0_pri, rho0, K):
    T = len(zt)

    ########### last time point ##########
    t = T - 1
    j = zt[t - 1]
    n_mat[j, zt[t]] -= 1
    ysum[zt[t]] -= yt[t]
    ycnt[zt[t]] -= 1

    ## conpute posterior distributions
    tmp_vec = np.arange(K)
    zt_dist = (alpha0 * beta_vec + n_mat[j] + rho0 * (j == tmp_vec)) / (alpha0 + n_mat[j].sum() + rho0)
    knew_dist = alpha0 * beta_new / (alpha0 + n_mat[j].sum() + rho0)

    ## compute y marginal likelihood
    varn = 1 / (1 / (sigma0_pri ** 2) + ycnt / (sigma0 ** 2))
    mun = ((mu0 / (sigma0_pri ** 2)) + (ysum / (sigma0 ** 2))) * varn

    yt_dist = ss.norm.pdf(yt[t], mun, np.sqrt((sigma0 ** 2) + varn))
    yt_knew_dist = ss.norm.pdf(yt[t], mu0, np.sqrt((sigma0 ** 2) + (sigma0_pri ** 2)))

    ## construct z's posterior by cases
    post_cases = np.hstack((zt_dist * yt_dist, knew_dist * yt_knew_dist))
    post_cases = post_cases / (post_cases.sum())

    ## sample zt
    zt[t] = np.where(multinomial(1, post_cases))[0][0]

    ## update beta_vec, n_mat when having a new state
    if zt[t] == K:
        b = beta(1, gamma0, size=1)
        beta_vec = np.hstack((beta_vec, b * beta_new))
        beta_new = (1 - b) * beta_new
        n_mat = np.hstack((n_mat, np.zeros((K, 1))))
        n_mat = np.vstack((n_mat, np.zeros((1, K + 1))))
        ysum = np.hstack((ysum, 0))
        ycnt = np.hstack((ycnt, 0))
        K += 1

    ## update n_mat
    n_mat[j, zt[t]] += 1
    ysum[zt[t]] += yt[t]
    ycnt[zt[t]] += 1
    return zt, n_mat, ysum, ycnt, beta_vec, beta_new, K


## input zt, wt, n_mat, beta_vec, beta_new, kappa_vec, kappa_new, alpha0, gamma0, sigma0, K
def sample_zw(zt, yt, n_mat, ysum, ycnt, beta_vec, beta_new, alpha0, gamma0, sigma0, mu0, sigma0_pri, rho0, K):
    T = len(zt)

    for t in range(1, T - 1):
        # print(wt[t],wt[t+1],n_mat);
        j = zt[t - 1]
        l = zt[t + 1]
        n_mat[j, zt[t]] -= 1
        n_mat[zt[t], l] -= 1
        ysum[zt[t]] -= yt[t]
        ycnt[zt[t]] -= 1

        ## conpute posterior distributions
        tmp_vec = np.arange(K)
        zt_dist = (alpha0 * beta_vec + n_mat[j] + rho0 * (j == tmp_vec)) / (alpha0 + n_mat[j].sum() + rho0)
        ztplus1_dist = (alpha0 * beta_vec[l] + n_mat[:, l] + rho0 * (l == tmp_vec) + (j == l) * (j == tmp_vec)) / (
                    alpha0 + n_mat.sum(axis=1) + rho0 + (j == tmp_vec))

        knew_dist = (alpha0 ** 2) * beta_vec[l] * beta_new / ((alpha0 + rho0) * (alpha0 + n_mat[j].sum() + rho0))

        ## compute y marginal likelihood
        varn = 1 / (1 / (sigma0_pri ** 2) + ycnt / (sigma0 ** 2))
        mun = ((mu0 / (sigma0_pri ** 2)) + (ysum / (sigma0 ** 2))) * varn

        yt_dist = ss.norm.pdf(yt[t], mun, np.sqrt((sigma0 ** 2) + varn))
        yt_knew_dist = ss.norm.pdf(yt[t], mu0, np.sqrt((sigma0 ** 2) + (sigma0_pri ** 2)))

        ## construct z's posterior by cases
        post_cases = np.hstack((zt_dist * ztplus1_dist * yt_dist, knew_dist * yt_knew_dist))
        post_cases = post_cases / (post_cases.sum())

        ## sample zt
        zt[t] = np.where(multinomial(1, post_cases))[0][0]

        ## update beta_vec, kappa_vec, n_mat when having a new state
        if zt[t] == K:
            b = beta(1, gamma0, size=1)
            beta_vec = np.hstack((beta_vec, b * beta_new))
            beta_new = (1 - b) * beta_new
            n_mat = np.hstack((n_mat, np.zeros((K, 1))))
            n_mat = np.vstack((n_mat, np.zeros((1, K + 1))))
            ysum = np.hstack((ysum, 0))
            ycnt = np.hstack((ycnt, 0))
            K += 1

        ## update n_mat
        n_mat[j, zt[t]] += 1
        n_mat[zt[t], l] += 1
        ysum[zt[t]] += yt[t]
        ycnt[zt[t]] += 1

    ## update last time point
    zt, n_mat, ysum, ycnt, beta_vec, beta_new, K = sample_last(zt, yt, n_mat, ysum, ycnt, beta_vec, beta_new, alpha0,
                                                               gamma0, sigma0, mu0, sigma0_pri, rho0, K)

    return zt, n_mat, ysum, ycnt, beta_vec, beta_new, K


## decrement K update n_mat, zt, beta_vec, K
def decre_K(zt, n_mat, ysum, ycnt, beta_vec):
    rem_ind = np.unique(zt)

    d = {k: v for v, k in enumerate(sorted(rem_ind))}
    zt = np.array([d[x] for x in zt])

    n_mat = n_mat[rem_ind][:, rem_ind]
    ysum = ysum[rem_ind]
    ycnt = ycnt[rem_ind]
    beta_vec = beta_vec[rem_ind]
    K = len(rem_ind)

    return zt, n_mat, ysum, ycnt, beta_vec, K


## input n_mat, beta_vec, alpha0
def sample_m(n_mat, beta_vec, alpha0, rho0, K):
    m_mat = np.zeros((K, K))

    for j in range(K):
        for k in range(K):
            if n_mat[j, k] == 0:
                m_mat[j, k] = 0
            else:
                x_vec = binomial(1, (alpha0 * beta_vec[k] + rho0 * (j == k)) / (
                            np.arange(n_mat[j, k]) + alpha0 * beta_vec[k] + rho0 * (j == k)))
                x_vec = np.array(x_vec).reshape(-1)
                m_mat[j, k] = sum(x_vec)

    return m_mat


def sample_w(K, m_mat, beta_vec, alpha0, rho0):
    w_vec = np.zeros(K)
    m_mat_bar = m_mat.copy()
    stick_ratio = rho0 / (rho0 + alpha0)
    for j in range(K):
        w_vec[j] = binomial(m_mat[j, j], stick_ratio / (stick_ratio + beta_vec[j] * (1 - stick_ratio)))
        m_mat_bar[j, j] = m_mat[j, j] - w_vec[j]

    ## for the first time point
    m_mat_bar[0, 0] += 1
    m_mat[0, 0] += 1
    return w_vec, m_mat, m_mat_bar


## input m_mat, gamma0
def sample_beta(m_mat_bar, gamma0):
    beta_vec = dirichlet(np.hstack((m_mat_bar.sum(axis=0), gamma0)), size=1)[0]
    beta_new = beta_vec[-1]
    beta_vec = beta_vec[:-1]

    ## return beta_vec, beta_new
    return beta_vec, beta_new


## sample hyperparams alpha0+rho0, gamma0, stick_ratio=rho0/(alpha0+rho0)

def sample_concentration(m_mat, n_mat, alpha0, rho0, alpha0_a_pri, alpha0_b_pri):
    r_vec = []
    tmp = n_mat.sum(axis=1)
    concentration = alpha0 + rho0

    for val in tmp:
        if val > 0:
            r_vec.append(beta(concentration + 1, val))
    r_vec = np.array(r_vec)

    s_vec = binomial(1, n_mat.sum(axis=1) / (n_mat.sum(axis=1) + concentration))

    concentration = gamma(alpha0_a_pri + (m_mat.sum()) - 1 - sum(s_vec), 1 / (alpha0_b_pri - sum(np.log(r_vec + eps))))

    return concentration  # , r_vec, s_vec


def sample_gamma(K, m_mat_bar, gamma0, gamma0_a_pri, gamma0_b_pri):
    eta = beta(gamma0 + 1, m_mat_bar.sum())

    pi_m = (gamma0_a_pri + K - 1) / (gamma0_a_pri + K - 1 + m_mat_bar.sum() * (gamma0_b_pri - np.log(eta + eps)))
    indicator = binomial(1, pi_m)

    if indicator:
        gamma0 = gamma(gamma0_a_pri + K, 1 / (gamma0_b_pri - np.log(eta + eps)))
    else:
        gamma0 = gamma(gamma0_a_pri + K - 1, 1 / (gamma0_b_pri - np.log(eta + eps)))

    return gamma0  # , eta


def sample_stick_ratio(w_vec, m_mat, c_pri, d_pri):
    stick_ratio = beta(w_vec.sum() + c_pri, m_mat.sum() - 1 - w_vec.sum() + d_pri)
    return stick_ratio


## sample pi|z, alpha, beta, rho0
def sample_pi_efox(K, alpha0, beta_vec, beta_new, n_mat, rho0):
    pi_mat = np.zeros((K + 1, K + 1))
    for j in range(K):
        prob_vec = np.hstack((alpha0 * beta_vec + n_mat[j], alpha0 * beta_new))
        prob_vec[j] += rho0
        prob_vec[prob_vec < 0.01] = 0.01  ## clip step
        pi_mat[j] = dirichlet(prob_vec, size=1)[0]
    prob_vec = np.hstack((alpha0 * beta_vec, alpha0 * beta_new + rho0))
    prob_vec[prob_vec < 0.01] = 0.01  ## clip step
    pi_mat[-1] = dirichlet(prob_vec, size=1)[0]
    return pi_mat


## compute log marginal likelihood p(y|pi,alpha,beta,sigma,z,w,kappa,yold)
# forward - backward algorithm
# p(y1, ... yt, zt=k)
# p(y1, ... yt, yt+1, zt+1=j) = sum_k(p(y1, ... yt, zt=k)*pi(k,j))*yt+1

def compute_log_marginal_lik_gaussian(K, yt, zt, prob_mat, mu0, sigma0, sigma0_pri, ysum, ycnt):
    ## if zt is -1, then yt is a brand new sequence starting with state 0
    ## if zt is not -1, then it's the state of time point before the first time point of yt

    T = len(yt)
    a_mat = np.zeros((T + 1, K + 1))
    c_vec = np.zeros(T)
    if zt != -1:
        a_mat[0, zt] = 1  # np.log(ss.norm.pdf(yt[0],0,sigma0));

    ## compute mu sigma posterior
    varn = 1 / (1 / (sigma0_pri ** 2) + ycnt / (sigma0 ** 2))
    mun = ((mu0 / (sigma0_pri ** 2)) + (ysum / (sigma0 ** 2))) * varn

    varn = np.hstack((np.sqrt((sigma0 ** 2) + varn), np.sqrt((sigma0 ** 2) + (sigma0_pri ** 2))))
    mun = np.hstack((mun, mu0))

    for t in range(T):
        if t == 0 and zt == -1:
            j = 0
            a_mat[t + 1, j] = ss.norm.pdf(yt[t], mun[j], varn[j])
        else:
            for j in range(K + 1):
                a_mat[t + 1, j] = sum(a_mat[t, :] * prob_mat[:, j]) * ss.norm.pdf(yt[t], mun[j], varn[j])
        c_vec[t] = sum(a_mat[t + 1, :])
        a_mat[t + 1, :] /= c_vec[t]

    log_marginal_lik = sum(np.log(c_vec))
    return a_mat, log_marginal_lik

def compute_cost(zt, zt_real):
    cost_mat = []  #np.zeros((len(np.unique(zt_real)), len(np.unique(zt))));
    K_use = max(len(np.unique(zt_real)), len(np.unique(zt)))
    for ii in range(K_use):  ## real
        cost_mat.append([])
        for jj in range(K_use):
            cost_mat[ii].append((np.abs((zt_real==ii)*1 - (zt==jj)*1)).sum())
    #print_matrix(cost_mat);

    m = Munkres()
    indexes = m.compute(cost_mat)

    total = 0
    for row, column in indexes:
        value = cost_mat[row][column]
        total += value
        #print(f'({row}, {column}) -> {value}')
    #print(f'total cost: {total}')
    return total, indexes
