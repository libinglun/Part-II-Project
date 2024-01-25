import numpy as np
import scipy.stats as stats
from model.direct_assign_gibbs_base import DirectAssignment


class DirectAssignmentGaussian(DirectAssignment):
    def __init__(self, model, observations):
        DirectAssignment.__init__(self, model, observations)
        self.observations = observations
        self.seq_length = len(self.observations)
        self.hidden_states = np.zeros(self.seq_length, dtype='int')

        # emission params
        # (ysum) data points sum at each state (cluster), for Gaussian, we would calculate the sum of all observations for Gaussian use
        self.observed_data = np.array([observations[0]])
        # (ycnt) number of data points at each state (cluster)，observed_count_each_state[i] == 3 represents state i occurs 3 times
        self.observed_count = np.array([1])

        # Gaussian params
        self.mu0 = np.mean(self.observations)
        self.sigma_prior = np.std(self.observations) # sigma0_pri
        self.sigma0 = 0.5 # sigma0, subject to change

    def emission_pdf(self):

        # compute y marginal likelihood
        varn = 1 / (1 / (self.sigma_prior ** 2) + self.observed_count / (self.sigma0 ** 2))
        mun = ((self.mu0 / (self.sigma_prior ** 2)) + (self.observed_data / (self.sigma0 ** 2))) * varn

        return (lambda x: stats.norm.pdf(x, mun, np.sqrt((self.sigma0 ** 2) + varn)),
                lambda x: stats.norm.pdf(x, self.mu0, np.sqrt((self.sigma0 ** 2) + (self.sigma_prior ** 2))))


    def sample_one_step_ahead(self, t):
        # j is the hidden state value, ranging from 0 to T - 1
        last_state = self.hidden_states[t - 1]

        # derive the current hidden state posterior over K states
        posterior = self.model.hidden_states_posterior_with_last_state(last_state, self.observations[t],
                                                                       self.transition_count, self.K,
                                                                       self.emission_pdf)

        # update the current hidden state by multinomial
        self.hidden_states[t] = np.where(np.random.multinomial(1, posterior))[0][0]

        # update beta_vec, n_mat when having a new state
        have_new_state = (self.hidden_states[t] == self.K)

        if have_new_state:
            self.model.update_beta_with_new_state()

            # Extend the transition matrix with the new state
            # a new column of zero is being added to the right side of n_mat
            self.transition_count = np.hstack((self.transition_count, np.zeros((self.K, 1))))
            # a new row of zero is being added to the bottom of n_mat
            self.transition_count = np.vstack((self.transition_count, np.zeros((1, self.K + 1))))
            # both ysum and ycnt is a 1D array, just append 0 at the end of array
            self.observed_data = np.hstack((self.observed_data, 0))
            self.observed_count = np.hstack((self.observed_count, 0))

            # Add a new state
            self.K += 1

        self.transition_count[last_state, self.hidden_states[t]] += 1
        self.observed_data[self.hidden_states[t]] += self.observations[t]
        self.observed_count[self.hidden_states[t]] += 1

    def sample_hidden_states_on_last_state(self, t):
        # j is the hidden state value, ranging from 0 to T - 1
        last_state = self.hidden_states[t - 1]

        # exclude the counts of the current state
        self.transition_count[last_state, self.hidden_states[t]] -= 1
        self.observed_data[self.hidden_states[t]] -= self.observations[t]
        self.observed_count[self.hidden_states[t]] -= 1

        # derive the current hidden state posterior over K states
        posterior = self.model.hidden_states_posterior_with_last_state(last_state, self.observations[t],
                                                                       self.transition_count, self.K,
                                                                       self.emission_pdf)

        # update the current hidden state by multinomial
        self.hidden_states[t] = np.where(np.random.multinomial(1, posterior))[0][0]

        # update beta_vec, n_mat when having a new state
        have_new_state = (self.hidden_states[t] == self.K)

        if have_new_state:
            self.model.update_beta_with_new_state()

            # Extend the transition matrix with the new state
            # a new column of zero is being added to the right side of n_mat
            self.transition_count = np.hstack((self.transition_count, np.zeros((self.K, 1))))
            # a new row of zero is being added to the bottom of n_mat
            self.transition_count = np.vstack((self.transition_count, np.zeros((1, self.K + 1))))
            # both ysum and ycnt is a 1D array, just append 0 at the end of array
            self.observed_data = np.hstack((self.observed_data, 0))
            self.observed_count = np.hstack((self.observed_count, 0))

            # Add a new state
            self.K += 1

        self.transition_count[last_state, self.hidden_states[t]] += 1
        self.observed_data[self.hidden_states[t]] += self.observations[t]
        self.observed_count[self.hidden_states[t]] += 1

    def sample_hidden_states_on_last_next_state(self, t):
        # define last_state(j), next_state(l)
        last_state = self.hidden_states[t - 1]
        next_state = self.hidden_states[t + 1]

        # exclude the counts of the current state
        self.transition_count[last_state, self.hidden_states[t]] -= 1
        self.transition_count[self.hidden_states[t], next_state] -= 1
        self.observed_data[self.hidden_states[t]] -= self.observations[t]
        self.observed_count[self.hidden_states[t]] -= 1

        # derive the current hidden state posterior over K states
        posterior = self.model.hidden_states_posterior(last_state, next_state, self.observations[t],
                                                                       self.transition_count, self.K,
                                                                       self.emission_pdf)

        # update the current hidden state by multinomial
        self.hidden_states[t] = np.where(np.random.multinomial(1, posterior))[0][0]

        # update beta_vec, n_mat when having a new state
        have_new_state = (self.hidden_states[t] == self.K)

        if have_new_state:
            self.model.update_beta_with_new_state()

            # Extend the transition matrix with the new state
            # a new column of zero is being added to the right side of n_mat
            self.transition_count = np.hstack((self.transition_count, np.zeros((self.K, 1))))
            # a new row of zero is being added to the bottom of n_mat
            self.transition_count = np.vstack((self.transition_count, np.zeros((1, self.K + 1))))
            # both ysum and ycnt is a 1D array, just append 0 at the end of array
            self.observed_data = np.hstack((self.observed_data, 0))
            self.observed_count = np.hstack((self.observed_count, 0))

            # Add a new state
            self.K += 1

        self.transition_count[last_state, self.hidden_states[t]] += 1
        self.transition_count[self.hidden_states[t], next_state] += 1
        self.observed_data[self.hidden_states[t]] += self.observations[t]
        self.observed_count[self.hidden_states[t]] += 1

    def update_K(self):
        # rem_ind contains all the unique value of zt
        remain_index = np.unique(self.hidden_states)

        # a dictionary with key = state, value = the sorted index (map old states to new states)
        state_mapping = {k: v for v, k in enumerate(sorted(remain_index))}
        # use indexes as the new states
        self.hidden_states = np.array([state_mapping[state] for state in self.hidden_states])

        # only select rows and columns of n_mat according to the index of rem_ind
        self.transition_count = self.transition_count[remain_index][:, remain_index]
        self.observed_data = self.observed_data[remain_index]
        self.observed_count = self.observed_count[remain_index]
        self.model.beta_vec = self.model.beta_vec[remain_index]

        # update the new state space
        self.K = len(remain_index)


    def compute_log_marginal_likelihood(self, test_observations, start_point=-1):
        # if zt is -1, then yt is a brand-new sequence starting with state 0
        # if zt is not -1, then it's the state of time point before the first time point of yt
        length = len(test_observations)
        a_mat = np.zeros((length + 1, self.K + 1))
        c_vec = np.zeros(length)
        if start_point != -1:
            a_mat[0, start_point] = 1  # np.log(ss.norm.pdf(yt[0],0,sigma0));

        # TODO: abstract emission distribution params posterior
        # TODO: compare and contrast with multinomial emission posterior
        # compute mu sigma posterior
        # similar to the emission pdf
        varn = 1 / (1 / (self.sigma_prior ** 2) + self.observed_count / (self.sigma0 ** 2))
        mun = ((self.mu0 / (self.sigma_prior ** 2)) + (self.observed_data / (self.sigma0 ** 2))) * varn

        varn = np.hstack(
            (np.sqrt((self.sigma0 ** 2) + varn), np.sqrt((self.sigma0 ** 2) + (self.sigma_prior ** 2))))
        mun = np.hstack((mun, self.mu0))

        for t in range(length):
            if t == 0 and start_point == -1:
                j = 0
                a_mat[t + 1, j] = stats.norm.pdf(test_observations[t], mun[j], varn[j])
            else:
                for j in range(self.K + 1):
                    a_mat[t + 1, j] = sum(a_mat[t, :] * self.pi_mat[:, j]) * stats.norm.pdf(test_observations[t],
                                                                                            mun[j], varn[j])
            c_vec[t] = sum(a_mat[t + 1, :])
            a_mat[t + 1, :] /= c_vec[t]

        log_marginal_lik = sum(np.log(c_vec))
        return a_mat, log_marginal_lik
