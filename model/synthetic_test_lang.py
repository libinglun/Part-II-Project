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
save_path = "../result/"

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--iter', type=int, default=20)
    args.add_argument('--path', type=str)
    return args.parse_args()


def euclidean_distance(A, B):
    assert A.shape == B.shape, "two matrices should have the same shape"
    return np.sqrt(np.sum((A - B) ** 2))

def difference(A, B):
    assert A.shape == B.shape, "two matrices should have the same shape"
    return np.sum(A != B)

def kl_divergence(P, Q):
    assert P.shape == Q.shape, "two matrices should have the same shape"
    mask = (P != 0) & (Q != 0)
    filtered_P = P[mask]
    filtered_Q = Q[mask]
    return np.sum(filtered_P * np.log(filtered_P / filtered_Q))

dataset_path = "../data/PennTreebank_synthetic_dataset(noise-0.8).npz"
loaded_npz = np.load(dataset_path, allow_pickle=True)
num_states = int(loaded_npz['num_states'])
num_observations = int(loaded_npz['num_obs'])
observations = list(loaded_npz['observation'])
real_hidden_states = list(loaded_npz['real_hidden_universal'])
noisy_hidden_states = list(loaded_npz['noisy_hidden_universal'])
noisy_level = loaded_npz['noisy_level']

size = len(observations)

real_trans_count = np.zeros((num_states, num_states), dtype='int')
noisy_trans_count = np.zeros((num_states, num_states), dtype='int')
emis_count = np.zeros((num_observations, num_states), dtype='int')

for i in range(size):
    for t in range(1, len(observations[i])):            # starts from 1 to bypass -1 at the beginning
        emis_count[observations[i][t], noisy_hidden_states[i][t]] += 1

        noisy_trans_count[noisy_hidden_states[i][t - 1], noisy_hidden_states[i][t]] += 1
        real_trans_count[real_hidden_states[i][t - 1], real_hidden_states[i][t]] += 1
print("real trans count: \n", real_trans_count)
print("noisy trans count: \n", noisy_trans_count)
print(np.sum(noisy_trans_count))
print(euclidean_distance(real_trans_count, noisy_trans_count))


def train_sampler(iters, prev_iters, hidden_states, observations, transition_count, emission_count, real_transition_count, noisy_level, alpha=None, gamma=None, beta=None, K=num_states, mode='train'):
    model = HDPHMM(alpha, gamma, beta)
    sampler = DirectAssignmentPOS(model, observations, num_observations)
    sampler.K = K
    sampler.transition_count = transition_count.copy()
    sampler.emission_count = emission_count.copy()
    sampler.hidden_states = hidden_states.copy()
    if mode == 'train':
        for i in range(num_states - 1):
            sampler.model.update_beta_with_new_state()

    kl_divergence_result = []
    K_result = []
    alpha_result = []
    gamma_result = []
    best_distance = 1e9
    best_alpha, best_gamma = None, None
    best_beta = None
    sampled_trans_dist = None
    sampled_emis_dist = None

    iterations = iters
    for iter in tqdm.tqdm(range(iterations), desc="training sampler:"):
        for index in range(len(observations)):
            for t in range(1, sampler.seq_length[index] - 1):
                sampler.sample_hidden_states_on_last_next_state(index, t)
            sampler.sample_hidden_states_on_last_state(index, sampler.seq_length[index] - 1)

        sampler.update_K()
        print("new K: ", sampler.K)
        K_result.append(sampler.K)

        sampler.sample_m()
        sampler.sample_beta()
        sampler.sample_alpha()
        alpha_result.append(alpha)
        sampler.sample_gamma()
        gamma_result.append(gamma)
        # print(f"iteration {iter} has transition counts {sampler.transition_count.sum()} in total: \n {sampler.transition_count}")
        count_distance = euclidean_distance(sampler.transition_count[:num_states, :num_states], real_transition_count)
        trans_KL_divergence = kl_divergence(sampler.transition_count[:num_states, :num_states], real_transition_count)
        print(f"Distance between sampled and real transition counts is {count_distance}")
        print(f"KL Divergence between sampled and real transition counts is {trans_KL_divergence}")

        mis_states = difference(sampler.hidden_states, np.array(real_hidden_states))
        print(f"Total number of wrong hidden states is {mis_states}")

        kl_divergence_result.append((count_distance, trans_KL_divergence, mis_states))
        print(sampler.transition_count)

        if count_distance < best_distance:
            best_distance = count_distance
            sampled_trans_dist = sampler.sample_transition_distribution()
            sampled_emis_dist = sampler.calculate_emission_distribution()
            beta = np.hstack((sampler.model.beta_vec, sampler.model.beta_new.reshape(1, )))
            best_alpha = sampler.model.alpha
            best_beta = beta
            best_gamma = sampler.model.gamma


        if iter == iterations - 1:
            observation_object = np.array(observations, dtype=object)
            hidden_states_object = np.array(sampler.hidden_states, dtype=object)
            beta = np.hstack((sampler.model.beta_vec, sampler.model.beta_new.reshape(1,)))
            timestamp = time.strftime("%m%d_%H%M%S", time.gmtime(time.time()))
            np.savez(save_path + f"ptb-noise-{noisy_level}_iter-{iterations+prev_iters}_timestamp-{timestamp}_state.npz",
                     observation=observation_object, K=sampler.K,
                     hidden_state=hidden_states_object, trans_count=sampler.transition_count, emis_count=sampler.emission_count,
                     alpha=sampler.model.alpha, gamma=sampler.model.gamma, beta=beta)
            np.savez(
                save_path + f"ptb-noise-{noisy_level}_iter-{iterations+prev_iters}_timestamp-{timestamp}_result.npz",
                trans_dist=sampled_trans_dist, emis_dist=sampled_emis_dist, K=K_result,
                alpha=best_alpha, gamma=best_gamma, beta=best_beta, result=np.array(kl_divergence_result),
                     hyperparam_alpha=alpha_result, hyperparam_gamma=gamma_result)


if __name__ == "__main__":
    args = parse_args()
    iters = args.iter
    if args.mode == 'train':
        train_sampler(iters, 0, noisy_hidden_states, observations, noisy_trans_count, emis_count, real_trans_count, noisy_level)
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

        train_sampler(iters, prev_iters, hidden_states, observations, trans_count, emis_count, real_trans_count, noisy_level, alpha, gamma, beta, K, 'resume')
