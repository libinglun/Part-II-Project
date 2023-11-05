import numpy as np
from numpy.typing import NDArray

CONST_EPS = 1e-6


class HDPHMM:
    """
    Parameter rho revered for sticky-HDP, however, other versions might not be compatible.
    Some functions such as sampling and updating beta are implemented in the sampler class, might consider implement in the HDP-HMM model
    but require complex argument passing from training to sampler and then to the model. Also, did not find a good way for the hierarchy.
    """

    def __init__(self, alpha_a_prior, alpha_b_prior, gamma_a_prior, gamma_b_prior, rho0=0):

        self.alpha_a_prior = alpha_a_prior
        self.alpha_b_prior = alpha_b_prior
        # the reciprocal might be due to parameter specification -- need to double-check
        self.alpha = np.random.gamma(alpha_a_prior, 1 / alpha_b_prior)

        self.gamma_a_prior = gamma_a_prior
        self.gamma_b_prior = gamma_b_prior
        self.gamma = np.random.gamma(gamma_a_prior, 1 / gamma_b_prior)

        # the kappa parameter in direct assignment sampler
        self.rho = rho0

        tmp = np.dirichlet(np.array([1, self.gamma]), size=1)[0]
        self.beta_new = tmp[-1]
        self.beta_vec = tmp[:-1]

        # guassian params
        self.mu = 0
        self.sigma = 0
        self.sigma_prior = 0

    def hidden_states_posterior_with_last_state(self, last_state: int, observation, transition_count: NDArray, K: int,
                                                emission_func):
        tmp_vec = np.arange(K)
        # p(z_t = k|params)
        current_hidden_state_dist = (
                (self.alpha * self.beta_vec + transition_count[last_state] + self.rho * (last_state == tmp_vec))
                / (self.alpha + transition_count[last_state].sum() + self.rho))

        new_hidden_state_dist = (self.alpha ** 2) * self.beta_new / (
                self.alpha + transition_count[last_state].sum() + self.rho)

        emission_pdf, emission_pdf_new = emission_func(self.mu, self.sigma, self.sigma_prior)
        # prob of yt[t] give the normal distribution, both yt_dist and yt_knew_dist are arrays with length K
        observation_dist = emission_pdf(observation)
        # new hidden state (cluster) with new observation emission pdf
        observation_dist_with_new = emission_pdf_new(observation)

        # construct z's posterior over k
        # add new column at the end of distribution array
        hidden_states_posterior = np.hstack((current_hidden_state_dist * observation_dist,
                                             new_hidden_state_dist * observation_dist_with_new))
        # normalise the new distribution array
        hidden_states_posterior = hidden_states_posterior / hidden_states_posterior.sum()

        return hidden_states_posterior

    def hidden_states_posterior(self, last_state: int, next_state: int, observation, transition_count: NDArray, K: int,
                                emission_func):
        """
        :param transition_count:transition_counts[i][j] number of transitions from state i to state j
        :param next_state: l
        :param last_state: j
        :param observation: current observation at step t (t is the index of direct assignment sampler)
        :param K: Current number of states in-use
        :param emission_pdf: the PARTIAL function of the pdf of emission distribution (e.g. norm.pdf), other params are
        specified be the caller.
        :param emission_pdf_new:
        :return:
        """

        tmp_vec = np.arange(K)
        # p(z_t = k|params)
        current_hidden_state_dist = (
                (self.alpha * self.beta_vec + transition_count[last_state] + self.rho * (last_state == tmp_vec))
                / (self.alpha + transition_count[last_state].sum() + self.rho))
        # p(z_t+1 = l|params)
        next_hidden_state_dist = (
                (self.alpha * self.beta_vec[next_state] + transition_count[:, next_state] + self.rho * (
                        next_state == tmp_vec) + (
                         last_state == next_state) * (last_state == tmp_vec))
                / (self.alpha + transition_count.sum(axis=1) + self.rho + (last_state == tmp_vec)))

        emission_pdf, emission_pdf_new = emission_func(self.mu, self.sigma, self.sigma_prior)
        # prob of yt[t] give the normal distribution, both yt_dist and yt_knew_dist are a single float
        observation_dist = emission_pdf(observation)
        # new hidden state (cluster) with new observation emission pdf
        observation_dist_with_new = emission_pdf_new(observation)

        # knew_dist is a single value
        # simplified equation of zt_dist * ztplus1_dist when k = K + 1
        new_hidden_state_dist = (self.alpha ** 2) * self.beta_vec[next_state] * self.beta_new / (
                (self.alpha + self.rho) * (self.alpha + transition_count[last_state].sum() + self.rho))

        # construct z's posterior over k
        # add new column at the end of distribution array
        hidden_states_posterior = np.hstack((current_hidden_state_dist * next_hidden_state_dist * observation_dist,
                                             new_hidden_state_dist * observation_dist_with_new))
        # normalise the new distribution array
        hidden_states_posterior = hidden_states_posterior / hidden_states_posterior.sum()

        return hidden_states_posterior

    def update_beta_with_new_state(self):
        b = np.beta(1, self.gamma, size=1)
        self.beta_vec = np.hstack((self.beta_vec, b * self.beta_new))
        self.beta_new *= (1 - b)

    def update_beta_with_new_params(self, param_vec):
        self.beta_vec = np.dirichlet(np.hstack((param_vec, self.gamma)), size=1)[0]
        self.beta_new = self.beta_vec[-1]
        self.beta_vec = self.beta_vec[:-1]

    def update_alpha(self, transition_row_sum, m_total_sum):
        concentration = self.alpha + self.rho

        r_vec = []
        for val in transition_row_sum:
            if val > 0:
                r_vec.append(np.beta(concentration + 1, val))
        r_vec = np.array(r_vec)

        s_vec = np.binomial(1, transition_row_sum / (transition_row_sum + concentration))

        # minus 1 here is to offset the additional one added on m_mat[0, 0] ?
        # I added "- self.rho" to be consistent with Berkley's notes, but this is inconsistent with original codes.
        self.alpha = np.gamma(self.alpha_a_prior + m_total_sum - 1 - sum(s_vec),
                              1 / (self.alpha_b_prior - sum(np.log(r_vec + CONST_EPS)))) - self.rho

    def update_gamma(self, m_total_sum, K):
        """
        Zhou's code is really inconsistent with Berkley's notes
        :param m_total_sum:
        :param K:
        :return:
        """
        eta = np.beta(self.gamma + 1, m_total_sum)

        indicator = np.binomial(1, m_total_sum / m_total_sum + self.gamma)

        if indicator:
            self.gamma = np.gamma(self.gamma_a_prior + K, 1 / (self.gamma_b_prior - np.log(eta + CONST_EPS)))
        else:
            self.gamma = np.gamma(self.gamma_a_prior + K - 1, 1 / (self.gamma_b_prior - np.log(eta + CONST_EPS)))

        # alternative solution (still different from Zhou's implementation)
        # pi_m = (self.gamma_a_prior + K - 1) / (m_total_sum * (self.gamma_b_prior - np.log(eta + CONST_EPS)))
        # self.gamma = pi_m * np.gamma(self.gamma_a_prior + K, 1 / (self.gamma_b_prior - np.log(eta + CONST_EPS))) + \
        #              (1 - pi_m) * np.gamma(self.gamma_a_prior + K - 1, 1 / (self.gamma_b_prior - np.log(eta + CONST_EPS)))
