import numpy as np
import tqdm
import argparse
import time
import re

from model import HDPHMM, DirectAssignmentPOS
from utils.utils import euclidean_distance, difference, kl_divergence

seed_vec = [111, 222, 333, 444, 555, 666, 777, 888, 999, 1000]
seed = 0  # random seed
np.random.seed(seed_vec[seed])  # fix randomness

np.set_printoptions(suppress=True, precision=4)
np.set_printoptions(linewidth=180)
np.set_printoptions(formatter={'int': '{:5d}'.format})

file_path = "../../data/"
save_path = "../../result/"

num_states = 10
num_observations = 5000
size = 50000
noisy_level = 0.5

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--iter', type=int, default=20)
    args.add_argument('--path', type=str)
    return args.parse_args()

dataset_path = f"../data/hmm_synthetic_dataset(noise-{noisy_level}_state-{num_states}_obs-{num_observations}_size-{size}).npz"
loaded_npz = np.load(dataset_path, allow_pickle=True)
observations = list(loaded_npz['observation'])
real_hidden_states = list(loaded_npz['real_hidden'])
noisy_hidden_states = list(loaded_npz['noisy_hidden'])
real_trans_dist = np.vstack(loaded_npz['real_trans'])
# noisy_level = loaded_npz['noisy_level']

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
total_count = np.sum(noisy_trans_count)
print(total_count)
print(euclidean_distance(real_trans_count, noisy_trans_count))

kl_divergence_result = []
K_result = []
alpha_result = []
gamma_result = []


def train_sampler(iters, prev_iters, hidden_states, observations, transition_count, emission_count, real_transition_count, real_trans_dist, noisy_level, alpha=None, gamma=None, beta=None, K=num_states, mode='train'):
    model = HDPHMM(alpha, gamma, beta)
    sampler = DirectAssignmentPOS(model, observations, num_observations, hidden_states=hidden_states, emission_count=emission_count, transition_count=transition_count, K=K)
    if mode == 'train':
        for i in range(num_states - 1):
            sampler.model.update_beta_with_new_state()

    best_score = 100
    sampled_trans_dist = None
    sampled_emis_dist = None

    iterations = iters
    for iter in tqdm.tqdm(range(iterations), desc="training model"):
        for index in range(len(observations)):
            for t in range(1, sampler.seq_length[index] - 1):
                sampler.sample_hidden_states_on_last_next_state(index, t)
            sampler.sample_hidden_states_on_last_state(index, sampler.seq_length[index] - 1)

        sampler.update_K()
        # print("hidden states after update K:", model.hidden_states[:5])
        print("new K: ", sampler.K)
        K_result.append(sampler.K)

        sampler.sample_m()
        sampler.sample_beta()
        sampler.sample_alpha()
        alpha_result.append(sampler.model.alpha)
        sampler.sample_gamma()
        gamma_result.append(sampler.model.gamma)
        # print(f"iteration {iter} has transition counts {model.transition_count.sum()} in total: \n {model.transition_count}")
        count_distance = euclidean_distance(sampler.transition_count[:num_states, :num_states], real_transition_count)
        print(f"Distance between sampled and real transition counts is {count_distance}")

        trans_dist = sampler.sample_transition_distribution()
        trans_distance = euclidean_distance(trans_dist[:num_states, :num_states], real_trans_dist)
        trans_KL_divergence = kl_divergence(trans_dist[:num_states, :num_states], real_trans_dist)
        print(f"Distance between sampled and real transition distribution is {trans_distance}")
        print(f"KL Divergence between sampled and real transition distribution is {trans_KL_divergence}")

        mis_states = difference(sampler.hidden_states, real_hidden_states)
        print(f"The rate of missing states is:  {round(mis_states / total_count * 100, 3)}%")

        kl_divergence_result.append((count_distance, trans_KL_divergence, mis_states))

        if trans_KL_divergence < best_score:
            best_score = trans_KL_divergence
            sampled_trans_dist = trans_dist
            sampled_emis_dist = sampler.calculate_emission_distribution()

        if iter == iterations - 1:
            observation_object = np.array(observations, dtype=object)
            hidden_states_object = np.array(sampler.hidden_states, dtype=object)
            beta = np.hstack((sampler.model.beta_vec, sampler.model.beta_new.reshape(1,)))
            timestamp = time.strftime("%m%d_%H%M%S", time.gmtime(time.time()))
            np.savez(save_path + f"noise-{noisy_level}_iter-{iterations+prev_iters}_state-{num_states}_obs-{num_observations}_size-{size}_timestamp-{timestamp}_state.npz",
                     observation=observation_object, K=sampler.K,
                     hidden_state=hidden_states_object, trans_count=sampler.transition_count, emis_count=sampler.emission_count,
                     alpha=sampler.model.alpha, gamma=sampler.model.gamma, beta=beta)
            np.savez(
                save_path + f"noise-{noisy_level}_iter-{iterations + prev_iters}_state-{num_states}_obs-{num_observations}_size-{size}_timestamp-{timestamp}_result.npz",
                trans_dist=sampled_trans_dist, emis_dist=sampled_emis_dist, K=K_result, result=np.array(kl_divergence_result),
                hyperparam_alpha=alpha_result, hyperparam_gamma=gamma_result)


if __name__ == "__main__":
    args = parse_args()
    iters = args.iter
    if args.mode == 'train':
        train_sampler(iters, 0, noisy_hidden_states, observations, noisy_trans_count, emis_count, real_trans_count, real_trans_dist, noisy_level)
    if args.mode == 'resume':
        if args.path is None:
            raise ValueError("Please specify the path of stored params!")
        match = re.search(r'iter-(\d+)', args.path)
        prev_iters = int(match.group(1))
        load_path = save_path + args.path + ".npz"
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
