import numpy as np
from munkres import Munkres, print_matrix
from collections import Counter


def compute_cost(zt, zt_real):
    cost_mat = []  # np.zeros((len(np.unique(zt_real)), len(np.unique(zt))));
    K_use = max(len(np.unique(zt_real)), len(np.unique(zt)))
    for ii in range(K_use):  ## real
        cost_mat.append([])
        for jj in range(K_use):
            cost_mat[ii].append((np.abs((zt_real == ii) * 1 - (zt == jj) * 1)).sum())
    # print_matrix(cost_mat);

    m = Munkres()
    indexes = m.compute(cost_mat)

    total = 0
    for row, column in indexes:
        value = cost_mat[row][column]
        total += value
        # print(f'({row}, {column}) -> {value}')
    # print(f'total cost: {total}')
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


def viterbi(observations, num_states, transition_prob, emission_prob):
    V = np.zeros((num_states, len(observations)))
    path = {}
    # first column of V is the transition prob from state 0
    V[:, 0] = transition_prob[0, :]

    for t in range(1, len(observations)):
        for s in range(1, num_states):
            prob = V[:, t - 1] * transition_prob[:, s] * emission_prob[s - 1, observations[t]]
            V[s, t] = np.max(prob)
            path[s, t] = np.argmax(prob)

    optimal_path = []
    last_state = np.argmax(V[:, -1])
    optimal_path.append(last_state)

    for t in range(len(observations) - 1, 1, -1):
        last_state = path[last_state, t]
        optimal_path.insert(0, last_state)

    optimal_path.insert(0, 0)

    return optimal_path


def set_print_options():
    np.set_printoptions(suppress=True, precision=4)
    np.set_printoptions(linewidth=180)
    np.set_printoptions(formatter={'int': '{:5d}'.format})

def flatten(lists):
    return [element for l in lists for element in l]


def calculate_entropy(cluster):
    """Calculate the entropy of a clustering."""
    total_points = len(cluster)
    if total_points == 0:
        return 0
    label_counts = Counter(cluster)
    probabilities = [count / total_points for count in label_counts.values()]
    entropy = -sum(p * np.log2(p) for p in probabilities)
    return entropy

def calculate_mutual_information(U, V):
    """Calculate the mutual information between two clusterings."""
    total_points = len(U)
    mutual_info = 0
    U_labels, V_labels = set(U), set(V)
    for u in U_labels:
        for v in V_labels:
            intersection_size = sum(1 for i in range(total_points) if U[i] == u and V[i] == v)
            if intersection_size == 0:
                continue
            p_u = sum(1 for x in U if x == u) / total_points
            p_v = sum(1 for x in V if x == v) / total_points
            p_uv = intersection_size / total_points
            mutual_info += p_uv * np.log2(p_uv / (p_u * p_v))
    return mutual_info

def calculate_variation_of_information(U, V):
    """Calculate the variation of information between two clusterings."""
    entropy_U = calculate_entropy(U)
    entropy_V = calculate_entropy(V)
    mutual_information = calculate_mutual_information(U, V)
    return entropy_U + entropy_V - 2 * mutual_information

