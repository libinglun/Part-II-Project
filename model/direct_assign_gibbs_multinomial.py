from hdp_hmm import HDPHMM
import numpy as np
import scipy.special as ssp
from direct_assign_gibbs_base import DirectAssignment


class DirectAssignmentMultinomial(DirectAssignment):
    def __init__(self, model: HDPHMM, observations):
        DirectAssignment.__init__(self, model, observations)

        # (ysum) data points at each state (cluster), for multinomial
        self.observed_data = np.array([observations[0]])
        self.state_length = len(observations[0]) # m_multi
        self.n_multi = sum(self.observations[0]) # n_multi

        # Multinomial params
        self.dir0 = 1 * np.ones(self.state_length) # shape[1] is number of columns
        self.dir0sum = np.sum(self.dir0)

    def emission_pdf(self):
        return lambda x: np.exp(np.real(
            (ssp.loggamma(self.dir0sum + self.observed_data.sum(axis=1)) - ssp.loggamma(
                self.dir0sum + self.observed_data.sum(axis=1) + self.n_multi)) + np.sum(
                ssp.loggamma(self.dir0 + x + self.observed_data), axis=1) - np.sum(
                ssp.loggamma(self.dir0 + self.observed_data), axis=1))), lambda x: np.exp(
            np.real(ssp.loggamma(self.dir0sum) - ssp.loggamma(self.dir0sum + self.n_multi) + np.sum(
                ssp.loggamma(self.dir0 + x)) - np.sum(ssp.loggamma(self.dir0))))

    def update_emission_statistics(self, new, t):
        if new:
            self.observed_data = np.vstack((self.observed_data, np.zeros((1, self.state_length))))

        self.observed_data[self.hidden_states[t]] += self.observations[t]

    def rearrange_emission_statistics(self, remain_index):
        self.observed_data = self.observed_data[remain_index]

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
