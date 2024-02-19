import numpy as np
import tqdm

from ..utils.utils import compute_cost, set_print_options

from ..logger import mylogger


def train_sampler(sampler, args, dataset):
    set_print_options()
    iterations = args.iter
    # K_real = args.num_states

    # the hidden states are empty initially. Fill in hidden states for the first iteration only based on last state j
    for t in range(1, sampler.seq_length):
        sampler.sample_one_step_ahead(t)

    print(sampler.transition_count)
    print(sampler.emission_count[:10])

    # raise ValueError('break point')

    for iteration in range(iterations):
        for t in range(1, sampler.seq_length - 1):
            sampler.sample_hidden_states_on_last_next_state(t)
        sampler.sample_hidden_states_on_last_state(sampler.seq_length - 1)
        # print(sampler.transition_count)
        # print(sampler.emission_count[:10])
        print(f"New sampled K is: {sampler.K}")
        print(sampler.hidden_states[:20])
        sampler.update_K()
        sampler.sample_m()
        sampler.sample_beta()
        sampler.sample_alpha()
        sampler.sample_gamma()

        if iteration % 10 == 0:
            print(f"New updated K is: {sampler.K}")
            sampler.sample_transition_distribution()
            # calculate the log likelihood of test observation sequence based on the new sampled transition distribution and result of direct assignment sampling (every 10 iters)
            _, loglik = sampler.compute_log_marginal_likelihood(dataset.test_obs)
            print(f"The log likelihood of test observation is: {loglik}")
            mylogger.info(f"The log likelihood of test observation is: {loglik}")

            # loglik_test_sample.append(loglik)
            # hidden_states_sample.append(sampler.hidden_states.copy())
            # hyperparams_sample.append(np.array([sampler.model.alpha, sampler.model.gamma]))

            cost, indexes = compute_cost(sampler.hidden_states.copy(), dataset.train_hid)
            dic = dict((v, k) for k, v in indexes)
            tmp = np.array([dic[sampler.hidden_states[t]] for t in range(args.len)])
            zero_one_loss = np.sum(tmp != dataset.train_hid)
            print(f"Zero one loss rate is : {round(zero_one_loss / args.len * 100, 3)}%")
            mylogger.info(f"Zero one loss rate is : {round(zero_one_loss / args.len * 100, 3)}%")
