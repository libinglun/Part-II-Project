from hdp_hmm import HDPHMM
import numpy as np
import tqdm
from direct_assign_gibbs_pos import DirectAssignmentPOS
import argparse
import time
import re

seed_vec = [111, 222, 333, 444, 555, 666, 777, 888, 999, 1000]
seed = 0  # random seed
np.random.seed(seed_vec[seed])  # fix randomness

np.set_printoptions(suppress=True, precision=4)
np.set_printoptions(linewidth=180)
np.set_printoptions(formatter={'int': '{:5d}'.format})

file_path = "../data/"

num_states = 10
num_observations = 100

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--iter', type=int, default=20)
    args.add_argument('--path', type=str)
    return args.parse_args()


def euclidean_distance(A, B):
    return np.sqrt(np.sum((A - B) ** 2))


def kl_divergence(P, Q):
    mask = (P != 0) & (Q != 0)
    filtered_P = P[mask]
    filtered_Q = Q[mask]
    return np.sum(filtered_P * np.log(filtered_P / filtered_Q))

loaded_npz = np.load("../data/hmm_synthetic_dataset.npz", allow_pickle=True)
observations = list(loaded_npz['observation'])
real_hidden_states = list(loaded_npz['real_hidden'])
noisy_hidden_states = list(loaded_npz['noisy_hidden'])
real_trans_dist = np.vstack(loaded_npz['real_trans'])
noisy_level = loaded_npz['noisy_level']

size = 50000

real_trans_count = np.zeros((num_states, num_states), dtype='int')
noisy_trans_count = np.zeros((num_states, num_states), dtype='int')
emis_count = np.zeros((num_observations, num_states), dtype='int')

for i in range(size):
    for t in range(len(observations[i])):
        emis_count[observations[i][t], noisy_hidden_states[i][t]] += 1
        if t > 0:
            noisy_trans_count[noisy_hidden_states[i][t - 1], noisy_hidden_states[i][t]] += 1
            real_trans_count[real_hidden_states[i][t - 1], real_hidden_states[i][t]] += 1
print("real trans count: \n", real_trans_count)
print("noisy trans count: \n", noisy_trans_count)
print(np.sum(noisy_trans_count))
print(euclidean_distance(real_trans_count, noisy_trans_count))


def train_sampler(iters, prev_iters, hidden_states, observations, transition_count, emission_count, real_transition_count, real_trans_dist, noisy_level, alpha=None, gamma=None, beta=None, K=10, mode='train'):
    model = HDPHMM(alpha, gamma, beta)
    sampler = DirectAssignmentPOS(model, observations, num_observations)
    sampler.K = K
    sampler.transition_count = transition_count.copy()
    sampler.token_state_matrix = emission_count.copy()
    sampler.hidden_states = hidden_states.copy()
    if mode == 'train':
        for i in range(9):
            sampler.model.update_beta_with_new_state()

    iterations = iters
    for iter in tqdm.tqdm(range(iterations), desc="training sampler:"):
        for index in range(len(observations)):
            for t in range(1, sampler.seq_length[index] - 1):
                sampler.sample_hidden_states_on_last_next_state(index, t)
            sampler.sample_hidden_states_on_last_state(index, sampler.seq_length[index] - 1)

        sampler.update_K()
        # print("hidden states after update K:", sampler.hidden_states[:5])
        print("new K: ", sampler.K)
        sampler.sample_m()
        sampler.sample_beta()
        sampler.sample_alpha()
        sampler.sample_gamma()
        # print(f"iteration {iter} has transition counts {sampler.transition_count.sum()} in total: \n {sampler.transition_count}")
        count_distance = euclidean_distance(sampler.transition_count[:10, :10], real_transition_count)
        print(f"Distance between sampled and real transition counts is {count_distance}")

        sampler.sample_transition_distribution()
        trans_distance = euclidean_distance(sampler.pi_mat[:10, :10], real_trans_dist)
        trans_KL_divergence = kl_divergence(sampler.pi_mat[:10, :10], real_trans_dist)
        print(f"Distance between sampled and real transition distribution is {trans_distance}")
        print(f"KL Divergence between sampled and real transition distribution is {trans_KL_divergence}")


        if iter == iterations - 1:
            observation_object = np.array(observations, dtype=object)
            hidden_states_object = np.array(sampler.hidden_states, dtype=object)
            beta = np.hstack((sampler.model.beta_vec, sampler.model.beta_new.reshape(1,)))
            timestamp = time.strftime("%m%d_%H%M%S", time.gmtime(time.time()))
            np.savez(file_path + f"noise-{noisy_level}_iter-{iterations+prev_iters}_timestamp-{timestamp}_result.npz", observation=observation_object, K=sampler.K,
                     hidden_state=hidden_states_object, trans_count=sampler.transition_count, emis_count=sampler.token_state_matrix,
                     alpha=sampler.model.alpha, gamma=sampler.model.gamma, beta=beta)


if __name__ == "__main__":
    args = parse_args()
    iters = args.iter
    if args.mode == 'train':
        train_sampler(iters, noisy_hidden_states, observations, noisy_trans_count, emis_count, real_trans_count, real_trans_dist, noisy_level)
    if args.mode == 'resume':
        if args.path is None:
            raise ValueError("Please specify the path of stored params!")
        match = re.search(r'iter-(\d+)', args.path)
        prev_iters = int(match.group(1))
        load_path = file_path + args.path + ".npz"
        loaded_model = np.load(load_path, allow_pickle=True)
        observations = list(loaded_model['observation'])
        hidden_states = list(loaded_model['hidden_state'])
        K = int(loaded_model['K'])
        trans_count = np.array(loaded_model['trans_count'])
        emis_count = np.vstack(loaded_model['emis_count'])
        alpha = float(loaded_model['alpha'])
        gamma = float(loaded_model['gamma'])
        beta = np.array(loaded_model['beta'])

        train_sampler(iters, prev_iters, hidden_states, observations, trans_count, emis_count, real_trans_count, real_trans_dist, noisy_level, alpha, gamma, beta, K, 'resume')
