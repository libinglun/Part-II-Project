import numpy as np
import sys
from hdp_hmm import HDPHMM
from direction_assignment_gibbs import DirectAssignmentGibbs


seed_vec = [111, 222, 333, 444, 555, 666, 777, 888, 999, 1000]

seed = int((int(sys.argv[1]) - 1) % 10)  # random seed
np.random.seed(seed_vec[seed])  # fix randomness

file_name = "fix_8states_multinomial_same_trans_diff_stick"

train_data = np.load('../data/' + file_name + '.npz')
test_data = np.load('../data/test_' + file_name + '.npz')

real_hidden_states = train_data['zt']
real_observations = train_data['yt']

test_observations = test_data['yt']

loglik_test_sample = []
hidden_states_sample = []
hyperparams_sample = []

if __name__ == "__main__":
    iterations = 100
    model = HDPHMM()

    sampler = DirectAssignmentGibbs(model, real_observations)

    # the hidden states are empty initially. Fill in hidden states for the first iteration only based on last state j
    for t in range(1, sampler.seq_length):
        sampler.sample_hidden_states_on_last_state(t)

    for iteration in range(iterations):
        for t in range(1, sampler.seq_length - 1):
            sampler.sample_hidden_states_on_last_next_state(t)
        sampler.sample_hidden_states_on_last_state(sampler.seq_length - 1)
        sampler.update_K()
        sampler.sample_m()
        sampler.sample_beta()
        sampler.sample_alpha()
        sampler.sample_gamma()

        # sample transition distribution matrix based on the result of direct assignment sampling (every 10 iters)
        if iteration % 10 == 0:
            sampler.sample_transition_distribution()
            # calculate the log likelihood of test observation sequence based on the new sampled transition distribution and result of direct assignment sampling (every 10 iters)
            _, loglik = sampler.compute_log_marginal_likelihood(test_observations)
            # output a matrix a_mat, a_mat[i, j] represents the probability of state j at time stamp i
            loglik_test_sample.append(loglik)

            # save the result of sampled hidden states and hyperparameter (every 10 iters)
            hidden_states_sample.append(sampler.hidden_states.copy())
            hyperparams_sample.append(np.array([sampler.model.alpha, sampler.model.gamma]))
