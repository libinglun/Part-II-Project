from hdp_hmm import HDPHMM
import numpy as np
import scipy.stats as stats
from direct_assign_gibbs_base import DirectAssignment
from scipy import special as special


class DirectAssignmentGaussian(DirectAssignment):
    def __init__(self, model: HDPHMM, observations):
        DirectAssignment.__init__(self, model, observations)

        # emission params
        # (ysum) data points sum at each state (cluster), for Gaussian, we would calculate the sum of all observations for Gaussian use
        self.observed_data = np.array([observations[0]])
        # (ycnt) number of data points at each state (cluster)，observed_count_each_state[i] == 3 represents state i occurs 3 times
        self.observed_count = np.array([1])

        # Gaussian params
        self.mu0 = np.mean(self.observations)
        self.sigma_prior = np.std(self.observations) # sigma0_pr
        self.sigma0 = 0.5 # sigma0, subject to change

    def emission_pdf(self):

        # compute y marginal likelihood
        varn = 1 / (1 / (self.sigma_prior ** 2) + self.observed_count / (self.sigma0 ** 2))
        mun = ((self.mu0 / (self.sigma_prior ** 2)) + (self.observed_data / (self.sigma0 ** 2))) * varn

        return (lambda x: stats.norm.pdf(x, mun, np.sqrt((self.sigma0 ** 2) + varn)),
                lambda x: stats.norm.pdf(x, self.mu0, np.sqrt((self.sigma0 ** 2) + (self.sigma_prior ** 2))))

    def update_emission_statistics(self, new, t):
        if new:
            # both ysum and ycnt is a 1D array, just append 0 at the end of array
            self.observed_data = np.hstack((self.observed_data, 0))
            self.observed_count = np.hstack((self.observed_count, 0))

        self.observed_data[self.hidden_states[t]] += self.observations[t]
        self.observed_count[self.hidden_states[t]] += 1

    def rearrange_emission_statistics(self, remain_index):
        self.observed_data = self.observed_data[remain_index]
        self.observed_count = self.observed_count[remain_index]


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
