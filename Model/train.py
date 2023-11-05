import numpy as np

from hdp_hmm import HDPHMM
from direction_assignment_gibbs import DirectAssignmentGibbs

loglik_test_sample = []
hidden_states_sample = []
hyperparams_sample = []

if __name__ == "__main__":
    iterations = 100
    model = HDPHMM(0, 0, 0, 0)
    observations = []
    test_observations = []
    sampler = DirectAssignmentGibbs(model, observations)

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
