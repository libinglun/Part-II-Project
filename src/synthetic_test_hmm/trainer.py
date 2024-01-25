import tqdm
import time
import numpy as np

from utils import euclidean_distance, kl_divergence, difference
from const import SAVE_PATH, NOISE_LEVEL, NUM_STATES, NUM_OBS, SIZE

def train_sampler(sampler, iters, dataset, prev_iters=0):

    best_score = 100
    sampled_trans_dist = None
    sampled_emis_dist = None
    kl_divergence_result = []
    K_result = []
    alpha_result = []
    gamma_result = []

    iterations = iters
    for iter in tqdm.tqdm(range(iterations), desc="training model"):
        for index in range(len(dataset.observations)):
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
        count_distance = euclidean_distance(sampler.transition_count[:NUM_STATES, :NUM_STATES], dataset.real_trans_count)
        print(f"Distance between sampled and real transition counts is {count_distance}")

        trans_dist = sampler.sample_transition_distribution()
        trans_distance = euclidean_distance(trans_dist[:NUM_STATES, :NUM_STATES], dataset.real_trans_dist)
        trans_KL_divergence = kl_divergence(trans_dist[:NUM_STATES, :NUM_STATES], dataset.real_trans_dist)
        print(f"Distance between sampled and real transition distribution is {trans_distance}")
        print(f"KL Divergence between sampled and real transition distribution is {trans_KL_divergence}")

        mis_states, total_states = difference(sampler.hidden_states, dataset.real_hidden_states)
        print(mis_states, total_states)
        print(f"The rate of missing states is:  {round(mis_states / total_states * 100, 3)}%")

        print(sampler.hidden_states[:3])
        print(dataset.real_hidden_states[:3])

        kl_divergence_result.append((count_distance, trans_KL_divergence, mis_states))

        if trans_KL_divergence < best_score:
            best_score = trans_KL_divergence
            sampled_trans_dist = trans_dist
            sampled_emis_dist = sampler.calculate_emission_distribution()

        if iter == iterations - 1:
            observation_object = np.array(dataset.observations, dtype=object)
            hidden_states_object = np.array(sampler.hidden_states, dtype=object)
            beta = np.hstack((sampler.model.beta_vec, sampler.model.beta_new.reshape(1,)))
            timestamp = time.strftime("%m%d_%H%M%S", time.gmtime(time.time()))
            np.savez(SAVE_PATH + f"noise-{NOISE_LEVEL}_iter-{iterations + prev_iters}_state-{NUM_STATES}_obs-{NUM_OBS}_size-{SIZE}_timestamp-{timestamp}_state.npz",
                     observation=observation_object, K=sampler.K,
                     hidden_state=hidden_states_object, trans_count=sampler.transition_count, emis_count=sampler.emission_count,
                     alpha=sampler.model.alpha, gamma=sampler.model.gamma, beta=beta)
            np.savez(
                SAVE_PATH + f"noise-{NOISE_LEVEL}_iter-{iterations + prev_iters}_state-{NUM_STATES}_obs-{NUM_OBS}_size-{SIZE}_timestamp-{timestamp}_result.npz",
                trans_dist=sampled_trans_dist, emis_dist=sampled_emis_dist, K=K_result, result=np.array(kl_divergence_result),
                hyperparam_alpha=alpha_result, hyperparam_gamma=gamma_result)