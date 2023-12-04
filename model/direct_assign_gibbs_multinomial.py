from hdp_hmm import HDPHMM
import numpy as np
import scipy.special as ssp
from direct_assign_gibbs_base import DirectAssignment


class DirectAssignmentMultinomial(DirectAssignment):
    def __init__(self, model, observations, dir0=1):
        DirectAssignment.__init__(self, model, observations)

        # (ysum) data points at each state (cluster), for multinomial
        self.observed_data = np.array([observations[0]])
        self.state_length = len(observations[0]) # m_multi
        self.n_multi = sum(self.observations[0]) # n_multi

        # Multinomial params
        self.dir0 = dir0 * np.ones(self.state_length) # shape[1] is number of columns
        self.dir0sum = np.sum(self.dir0)

    def emission_pdf(self):
        return lambda x: np.exp(np.real(
            (ssp.loggamma(self.dir0sum + self.observed_data.sum(axis=1)) - ssp.loggamma(
                self.dir0sum + self.observed_data.sum(axis=1) + self.n_multi)) + np.sum(
                ssp.loggamma(self.dir0 + x + self.observed_data), axis=1) - np.sum(
                ssp.loggamma(self.dir0 + self.observed_data), axis=1))), lambda x: np.exp(
            np.real(ssp.loggamma(self.dir0sum) - ssp.loggamma(self.dir0sum + self.n_multi) + np.sum(
                ssp.loggamma(self.dir0 + x)) - np.sum(ssp.loggamma(self.dir0))))

    def sample_one_step_ahead(self, t):
        # j is the hidden state value, ranging from 0 to T - 1
        last_state = self.hidden_states[t - 1]
        # don't have to exclude any counts or statistics since it is the first iter

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

            self.observed_data = np.vstack((self.observed_data, np.zeros((1, self.state_length))))

            # Add a new state
            self.K += 1

        self.transition_count[last_state, self.hidden_states[t]] += 1
        self.observed_data[self.hidden_states[t]] += self.observations[t]


    def sample_hidden_states_on_last_state(self, t):
        # last time point

        # j is the hidden state value, ranging from 0 to T - 1
        last_state = self.hidden_states[t - 1]
        # exclude the counts of the current state
        self.transition_count[last_state, self.hidden_states[t]] -= 1
        self.observed_data[self.hidden_states[t]] -= self.observations[t]

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

            self.observed_data = np.vstack((self.observed_data, np.zeros((1, self.state_length))))

            # Add a new state
            self.K += 1

        self.transition_count[last_state, self.hidden_states[t]] += 1
        self.observed_data[self.hidden_states[t]] += self.observations[t]

    def sample_hidden_states_on_last_next_state(self, t):
        # define last_state(j), next_state(l)
        last_state = self.hidden_states[t - 1]
        next_state = self.hidden_states[t + 1]

        # exclude the counts of the current state
        self.transition_count[last_state, self.hidden_states[t]] -= 1
        self.transition_count[self.hidden_states[t], next_state] -= 1

        self.observed_data[self.hidden_states[t]] -= self.observations[t]

        # derive the current hidden state posterior over K states
        posterior = self.model.hidden_states_posterior(last_state, next_state, self.observations[t],
                                                       self.transition_count, self.K, self.emission_pdf)

        # update the current hidden state by multinomial
        self.hidden_states[t] = np.where(np.random.multinomial(1, posterior))[0][0]

        have_new_state = (self.hidden_states[t] == self.K)

        if have_new_state:
            self.model.update_beta_with_new_state()

            # Extend the transition matrix with the new state
            # a new column of zero is being added to the right side of n_mat
            self.transition_count = np.hstack((self.transition_count, np.zeros((self.K, 1))))
            # a new row of zero is being added to the bottom of n_mat
            self.transition_count = np.vstack((self.transition_count, np.zeros((1, self.K + 1))))

            self.observed_data = np.vstack((self.observed_data, np.zeros((1, self.state_length))))

            # Add a new state
            self.K += 1

        self.transition_count[last_state, self.hidden_states[t]] += 1
        self.transition_count[self.hidden_states[t], next_state] += 1
        self.observed_data[self.hidden_states[t]] += self.observations[t]

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
        self.model.beta_vec = self.model.beta_vec[remain_index]

        # update the new state space
        self.K = len(remain_index)


    def compute_log_marginal_likelihood(self, test_observations, start_point=-1):

        # if zt is -1, then yt is a brand-new sequence starting with state 0
        # if zt is not -1, then it's the state of time point before the first time point of yt

        length = len(test_observations)
        a_mat = np.zeros((length + 1, self.K + 1))
        c_vec = np.zeros(length)
        test_n_multi = sum(test_observations[0])
        if start_point != -1:
            a_mat[0, start_point] = 1  # np.log(ss.norm.pdf(yt[0],0,sigma0));

        # compute mu sigma posterior
        yt_dist = (ssp.loggamma(self.dir0sum + self.observed_data.sum(axis=1)) - ssp.loggamma(
            self.dir0sum + self.observed_data.sum(axis=1) + test_n_multi)) - np.sum(ssp.loggamma(self.dir0 + self.observed_data), axis=1)
        yt_knew_dist = ssp.loggamma(self.dir0sum) - ssp.loggamma(self.dir0sum + test_n_multi) - np.sum(ssp.loggamma(self.dir0))
        yt_dist = np.hstack((yt_dist, yt_knew_dist)) + ssp.loggamma(test_n_multi)
        yt_dist = np.real(yt_dist)

        single_term = np.vstack((self.dir0 + self.observed_data, self.dir0))

        for t in range(length):
            if t == 0 and start_point == -1:
                j = 0
                a_mat[t + 1, j] = np.exp(
                    yt_dist[j] + np.real(np.sum(ssp.loggamma(single_term[j] + test_observations[t]) - ssp.loggamma(1 + test_observations[t]))))
            else:
                for j in range(self.K + 1):
                    a_mat[t + 1, j] = sum(a_mat[t, :] * self.pi_mat[:, j]) * np.exp(
                        yt_dist[j] + np.real(
                            np.sum(ssp.loggamma(single_term[j] + test_observations[t]) - ssp.loggamma(1 + test_observations[t]))))

            c_vec[t] = sum(a_mat[t + 1, :])
            a_mat[t + 1, :] /= c_vec[t]

        log_marginal_lik = sum(np.log(c_vec))
        return a_mat, log_marginal_lik
