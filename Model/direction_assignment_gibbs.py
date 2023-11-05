from hdp_hmm import HDPHMM
import numpy as np
import scipy.stats as stats
from scipy import special as special


class DirectAssignmentGibbs:
    def __init__(self, model: HDPHMM, observations):
        self.model = model

        self.observations = observations
        self.seq_length = len(observations)
        self.hidden_states = np.zeros(self.seq_length, dtype='int')

        # (ysum) data points at each state (cluster), for Gaussian, we would calculate the sum of all observations for Gaussian use
        self.observed_data_each_state = np.array([observations[0]])
        # (ycnt) number of data points at each state (cluster)，observed_count_each_state[i] == 3 represents state i occurs 3 times
        self.observed_count_each_state = np.array([1])

        self.transition_count = np.array([[0]])  # (n_mat)
        self.m_mat = None
        self.pi_mat = None
        self.K = 1

    # def multinomial_emission_pdf(self):
    #     yt_dist = (ssp.loggamma(dir0_sum + self.observed_data_each_state.sum(axis=1)) - ssp.loggamma(
    #         dir0_sum + self.observed_data_each_state.sum(axis=1) + n_mul6ti)) + np.sum(ssp.loggamma(dir0 + yt[t] + ysum), axis=1) - np.sum(
    #         ssp.loggamma(dir0 + ysum), axis=1)
    #     yt_dist = np.real(yt_dist)
    #     yt_dist = np.exp(yt_dist)

    def gaussian_emission_pdf(self, mu0, sigma0, sigma0_pri):

        # compute y marginal likelihood
        varn = 1 / (1 / (sigma0_pri ** 2) + self.observed_count_each_state / (sigma0 ** 2))
        mun = ((mu0 / (sigma0_pri ** 2)) + (self.observed_data_each_state / (sigma0 ** 2))) * varn

        return (lambda x: stats.norm.pdf(x, mun, np.sqrt((sigma0 ** 2) + varn)),
                lambda x: stats.norm.pdf(x, mu0, np.sqrt((sigma0 ** 2) + (sigma0_pri ** 2))))

    def sample_hidden_states_on_last_state(self, t):
        # last time point

        # j is the hidden state value, ranging from 0 to T - 1
        last_state = self.hidden_states[t - 1]

        # derive the current hidden state posterior over K states
        posterior = self.model.hidden_states_posterior_with_last_state(last_state, self.observations[t],
                                                                       self.transition_count, self.K,
                                                                       self.gaussian_emission_pdf)

        # update the current hidden state by multinomial
        self.hidden_states[t] = np.where(np.multinomial(1, posterior))[0][0]

        # update beta_vec, n_mat when having a new state
        if self.hidden_states[t] == self.K:
            # Add a new state
            self.K += 1

            self.model.update_beta_with_new_state()

            # Extend the transition matrix with the new state
            # a new column of zero is being added to the right side of n_mat
            self.transition_count = np.hstack((self.transition_count, np.zeros((self.K, 1))))
            # a new row of zero is being added to the bottom of n_mat
            self.transition_count = np.vstack((self.transition_count, np.zeros((1, self.K + 1))))

            # both ysum and ycnt is a 1D array, just append 0 at the end of array
            self.observed_data_each_state = np.hstack((self.observed_data_each_state, 0))
            self.observed_count_each_state = np.hstack((self.observed_count_each_state, 0))

        else:
            self.transition_count[last_state, self.hidden_states[t]] += 1
            self.observed_data_each_state[self.hidden_states[t]] += self.observations[t]
            self.observed_count_each_state[self.hidden_states[t]] += 1

    def sample_hidden_states_on_last_next_state(self, t):

        # define last_state(j), next_state(l)
        last_state = self.hidden_states[t - 1]
        next_state = self.hidden_states[t + 1]

        # exclude the counts of the current state
        self.transition_count[last_state, self.hidden_states[t]] -= 1
        self.transition_count[self.hidden_states[t], next_state] -= 1

        # derive the current hidden state posterior over K states
        posterior = self.model.hidden_states_posterior(last_state, next_state, self.observations[t],
                                                       self.transition_count, self.K, self.gaussian_emission_pdf)

        # update the current hidden state by multinomial
        self.hidden_states[t] = np.where(np.multinomial(1, posterior))[0][0]

        if self.hidden_states[t] == self.K:
            # Add a new state
            self.K += 1

            self.model.update_beta_with_new_state()

            # Extend the transition matrix with the new state
            # a new column of zero is being added to the right side of n_mat
            self.transition_count = np.hstack((self.transition_count, np.zeros((self.K, 1))))
            # a new row of zero is being added to the bottom of n_mat
            self.transition_count = np.vstack((self.transition_count, np.zeros((1, self.K + 1))))

            # both ysum and ycnt is a 1D array, just append 0 at the end of array
            self.observed_data_each_state = np.hstack((self.observed_data_each_state, 0))
            self.observed_count_each_state = np.hstack((self.observed_count_each_state, 0))

        else:
            self.transition_count[last_state, self.hidden_states[t]] += 1
            self.transition_count[self.hidden_states[t], next_state] += 1
            self.observed_data_each_state[self.hidden_states[t]] += self.observations[t]
            self.observed_count_each_state[self.hidden_states[t]] += 1

    def update_K(self):
        # rem_ind contains all the unique value of zt
        remain_index = np.unique(self.hidden_states)

        # a dictionary with key = state, value = the sorted index (map old states to new states)
        state_mapping = {k: v for v, k in enumerate(sorted(remain_index))}
        # use indexes as the new states
        self.hidden_states = np.array([state_mapping[state] for state in self.hidden_states])

        # only select rows and columns of n_mat according to the index of rem_ind
        self.transition_count = self.transition_count[remain_index][:, remain_index]
        self.observed_data_each_state = self.observed_data_each_state[remain_index]
        self.observed_count_each_state = self.observed_count_each_state[remain_index]
        self.model.beta_vec = self.model.beta_new[remain_index]

        # update the new state space
        self.K = len(remain_index)

    def sample_m(self):
        self.m_mat = np.zeros((self.K, self.K))
        for j in range(self.K):
            for k in range(self.K):
                if self.transition_count[j, k] == 0:
                    self.m_mat[j, k] = 0
                else:
                    # move this to HDP_HMM, so that rho would be hidden from direct assignment sampler
                    x_vec = np.binomial(1, (self.model.alpha * self.model.beta_vec[k] + self.model.rho * (j == k)) / (
                            np.arange(self.transition_count[j, k]) + self.model.alpha * self.model.beta_vec[k] + self.model.rho * (j == k)))
                    x_vec = np.array(x_vec).reshape(-1)
                    self.m_mat[j, k] = sum(x_vec)

        self.m_mat[0, 0] += 1

    def sample_beta(self):
        param_vec = np.array(self.m_mat.sum(axis=0))
        self.model.update_beta_with_new_params(param_vec)

    def sample_alpha(self):
        # row sum of transition counts (n_(j.))
        transition_row_sum = self.transition_count.sum(axis=1)

        # sum given all axis of m_mat
        total_sum = self.m_mat.sum()

        self.model.update_alpha(transition_row_sum, total_sum)

    def sample_gamma(self):
        # sum given all axis of m_mat
        total_sum = self.m_mat.sum()
        self.model.update_gamma(total_sum, self.K)

    def sample_transition_distribution(self):
        self.pi_mat = np.zeros((self.K + 1, self.K + 1))
        for j in range(self.K):
            prob_vec = np.hstack((self.model.alpha * self.model.beta_vec + self.transition_count[j],
                                  self.model.alpha * self.model.beta_new))
            prob_vec[j] += self.model.rho
            prob_vec[prob_vec < 0.01] = 0.01  # clip step
            self.pi_mat[j] = np.dirichlet(prob_vec, size=1)[0]
        prob_vec = np.hstack(
            (self.model.alpha * self.model.beta_vec, self.model.alpha * self.model.beta_new + self.model.rho))
        prob_vec[prob_vec < 0.01] = 0.01  # clip step
        self.pi_mat[-1] = np.dirichlet(prob_vec, size=1)[0]

    def compute_log_marginal_likelihood(self, test_observations, start_point=-1):
        # if zt is -1, then yt is a brand-new sequence starting with state 0
        # if zt is not -1, then it's the state of time point before the first time point of yt
        length = len(test_observations)
        a_mat = np.zeros((length + 1, self.K + 1))
        c_vec = np.zeros(length)
        if start_point != -1:
            a_mat[0, start_point] = 1  # np.log(ss.norm.pdf(yt[0],0,sigma0));

        # compute mu sigma posterior
        varn = 1 / (1 / (self.model.sigma_prior ** 2) + self.observed_count_each_state / (self.model.sigma ** 2))
        mun = ((self.model.mu / (self.model.sigma_prior ** 2)) + (self.observed_data_each_state / (self.model.sigma ** 2))) * varn

        varn = np.hstack((np.sqrt((self.model.sigma ** 2) + varn), np.sqrt((self.model.sigma ** 2) + (self.model.sigma_prior ** 2))))
        mun = np.hstack((mun, self.model.mu))

        for t in range(length):
            if t == 0 and start_point == -1:
                j = 0
                a_mat[t + 1, j] = stats.norm.pdf(test_observations[t], mun[j], varn[j])
            else:
                for j in range(self.K + 1):
                    a_mat[t + 1, j] = sum(a_mat[t, :] * self.pi_mat[:, j]) * stats.norm.pdf(test_observations[t], mun[j], varn[j])
            c_vec[t] = sum(a_mat[t + 1, :])
            a_mat[t + 1, :] /= c_vec[t]

        log_marginal_lik = sum(np.log(c_vec))
        return a_mat, log_marginal_lik
