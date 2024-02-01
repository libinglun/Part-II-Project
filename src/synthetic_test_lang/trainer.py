import tqdm
import time
import numpy as np

from ..utils.utils import euclidean_distance, kl_divergence, difference
from ..utils.const import SAVE_PATH

from ..logger import mylogger

def train_sampler(sampler, args, dataset, prev_iters=0):

    best_distance = 1e9
    sampled_trans_dist = None
    sampled_emis_dist = None
    kl_divergence_result = []
    K_result = []
    alpha_result = []
    gamma_result = []
    best_alpha, best_gamma = None, None
    best_beta = None

    iterations = args.iter
    for iteration in tqdm.tqdm(range(iterations), desc="training model:"):
        for index in range(dataset.size):
            for t in range(1, sampler.seq_length[index] - 1):
                sampler.sample_hidden_states_on_last_next_state(index, t)
            sampler.sample_hidden_states_on_last_state(index, sampler.seq_length[index] - 1)

        sampler.update_K()
        print("new K: ", sampler.K)
        K_result.append(sampler.K)

        sampler.sample_m()
        sampler.sample_beta()

        sampler.sample_alpha()
        alpha_result.append(sampler.model.alpha)
        sampler.sample_gamma()
        gamma_result.append(sampler.model.gamma)

        count_distance = euclidean_distance(sampler.transition_count[:dataset.num_states, :dataset.num_states], dataset.real_trans_count)
        trans_KL_divergence = kl_divergence(sampler.transition_count[:dataset.num_states, :dataset.num_states], dataset.real_trans_count)
        print(f"Distance between sampled and real transition counts is {count_distance}")
        print(f"KL Divergence between sampled and real transition counts is {trans_KL_divergence}")
        mylogger.info(f"Distance between sampled and real transition counts is {count_distance}")
        mylogger.info(f"KL Divergence between sampled and real transition counts is {trans_KL_divergence}")

        mis_states, total_states = difference(sampler.hidden_states, dataset.real_hidden_states)
        print(f"The rate of missing states is:  {round(mis_states / total_states * 100, 3)}%")
        mylogger.info(f"The rate of missing states is:  {round(mis_states / total_states * 100, 3)}%")

        kl_divergence_result.append((count_distance, trans_KL_divergence, mis_states))
        print(sampler.transition_count)
        mylogger.info(f"The new sampled transition count is:\n {sampler.transition_count}")

        if count_distance < best_distance:
            best_distance = count_distance
            sampled_trans_dist = sampler.sample_transition_distribution()
            sampled_emis_dist = sampler.calculate_emission_distribution()
            beta = np.hstack((sampler.model.beta_vec, sampler.model.beta_new.reshape(1, )))
            best_alpha = sampler.model.alpha
            best_beta = beta
            best_gamma = sampler.model.gamma

        if iteration == iterations - 1:
            observation_object = np.array(dataset.observations, dtype=object)
            hidden_states_object = np.array(sampler.hidden_states, dtype=object)
            beta = np.hstack((sampler.model.beta_vec, sampler.model.beta_new.reshape(1, )))
            timestamp = time.strftime("%m%d_%H%M%S", time.gmtime(time.time()))
            np.savez(
                SAVE_PATH + f"{args.name}-noise-{args.noise}_iter-{iterations + prev_iters}_timestamp-{timestamp}_state.npz",
                observation=observation_object, K=sampler.K,
                hidden_state=hidden_states_object, trans_count=sampler.transition_count,
                emis_count=sampler.emission_count,
                alpha=sampler.model.alpha, gamma=sampler.model.gamma, beta=beta)
            np.savez(
                SAVE_PATH + f"{args.name}-noise-{args.noise}_iter-{iterations + prev_iters}_timestamp-{timestamp}_result.npz",
                trans_dist=sampled_trans_dist, emis_dist=sampled_emis_dist, K=K_result,
                alpha=best_alpha, gamma=best_gamma, beta=best_beta, result=np.array(kl_divergence_result),
                hyperparam_alpha=alpha_result, hyperparam_gamma=gamma_result)