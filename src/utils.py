import numpy as np
from munkres import Munkres, print_matrix

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

def euclidean_distance(A, B):
    assert A.shape == B.shape, "two matrices should have the same shape"
    return np.sqrt(np.sum((A - B) ** 2))

def difference(A, B):
    assert len(A) == len(B), "two hidden states set should have the same length"
    miss_sum = 0
    tot_num = 0
    for i in range(len(A)):
        assert len(A[i]) == len(B[i])
        miss_sum += np.sum(np.array(A[i]) != np.array(B[i]))
        tot_num += len(A[i])
    return miss_sum, tot_num

def kl_divergence(P, Q):
    assert P.shape == Q.shape, "two matrices should have the same shape"
    mask = (P != 0) & (Q != 0)
    filtered_P = P[mask]
    filtered_Q = Q[mask]
    return np.sum(filtered_P * np.log(filtered_P / filtered_Q))