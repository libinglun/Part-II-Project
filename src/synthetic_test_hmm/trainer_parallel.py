import tqdm
import time
import numpy as np
import multiprocessing
import os

from ..utils.utils import euclidean_distance, kl_divergence, difference
from ..utils.const import SAVE_PATH

from ..logger import mylogger


def process_index(sampler, index):
    print(f"Processing index {index} in process ID: {os.getpid()}")
    for t in range(1, sampler.seq_length[index] - 1):
        sampler.sample_hidden_states_on_last_next_state(index, t)
    sampler.sample_hidden_states_on_last_state(index, sampler.seq_length[index] - 1)


def train_sampler_parallel(sampler, args, dataset, prev_iters=0):
    best_score = 100
    sampled_trans_dist = None
    sampled_emis_dist = None
    kl_divergence_result = []
    K_result = []
    alpha_result = []
    gamma_result = []

    iterations = args.iter
    pool = multiprocessing.Pool()
    num_cores = multiprocessing.cpu_count()
    print(f"Number of CPU cores: {num_cores}")

    for iteration in tqdm.tqdm(range(iterations), desc="training model:"):
        pool.map(process_index, [(sampler, index) for index in range(args.size)])

        sampler.update_K()
        mylogger.info(f"new K: {sampler.K}")
        K_result.append(sampler.K)

        sampler.sample_m()
        sampler.sample_beta()
        sampler.sample_alpha()
        alpha_result.append(sampler.model.alpha)
        sampler.sample_gamma()
        gamma_result.append(sampler.model.gamma)
        # print(f"iteration {iter} has transition counts {model.transition_count.sum()} in total: \n {model.transition_count}")
        count_distance = euclidean_distance(sampler.transition_count[:args.states, :args.states],
                                            dataset.real_trans_count)
        mylogger.info(f"Distance between sampled and real transition counts is {count_distance}")
        print(f"Distance between sampled and real transition counts is {count_distance}")

        trans_dist = sampler.sample_transition_distribution()
        trans_distance = euclidean_distance(trans_dist[:args.states, :args.states], dataset.real_trans_dist)
        trans_KL_divergence = kl_divergence(trans_dist[:args.states, :args.states], dataset.real_trans_dist)
        mylogger.info(f"Distance between sampled and real transition distribution is {trans_distance}")
        mylogger.info(f"KL Divergence between sampled and real transition distribution is {trans_KL_divergence}")
        print(f"Distance between sampled and real transition distribution is {trans_distance}")
        print(f"KL Divergence between sampled and real transition distribution is {trans_KL_divergence}")

        mis_states, total_states = difference(sampler.hidden_states, dataset.real_hidden_states)
        # print(mis_states, total_states)
        mylogger.info(f"The rate of missing states is:  {round(mis_states / total_states * 100, 3)}%")
        print(f"The rate of missing states is:  {round(mis_states / total_states * 100, 3)}%")

        # print(sampler.hidden_states[:3])
        # print(dataset.real_hidden_states[:3])

        kl_divergence_result.append((count_distance, trans_KL_divergence, mis_states))

        if trans_KL_divergence < best_score:
            best_score = trans_KL_divergence
            sampled_trans_dist = trans_dist
            sampled_emis_dist = sampler.calculate_emission_distribution()

        if iteration == iterations - 1:
            observation_object = np.array(dataset.observations, dtype=object)
            hidden_states_object = np.array(sampler.hidden_states, dtype=object)
            beta = np.hstack((sampler.model.beta_vec, sampler.model.beta_new.reshape(1, )))
            timestamp = time.strftime("%m%d_%H%M%S", time.gmtime(time.time()))
            np.savez(
                SAVE_PATH + f"noise-{args.noise}_iter-{iterations + prev_iters}_state-{args.states}_obs-{args.obs}_size-{args.size}_timestamp-{timestamp}_state.npz",
                observation=observation_object, K=sampler.K,
                hidden_state=hidden_states_object, trans_count=sampler.transition_count,
                emis_count=sampler.emission_count,
                alpha=sampler.model.alpha, gamma=sampler.model.gamma, beta=beta)
            np.savez(
                SAVE_PATH + f"noise-{args.noise}_iter-{iterations + prev_iters}_state-{args.states}_obs-{args.obs}_size-{args.size}_timestamp-{timestamp}_result.npz",
                trans_dist=sampled_trans_dist, emis_dist=sampled_emis_dist, K=K_result,
                result=np.array(kl_divergence_result),
                hyperparam_alpha=alpha_result, hyperparam_gamma=gamma_result)

    pool.close()
    pool.join()