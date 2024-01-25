import numpy as np
from numpy.typing import NDArray

CONST_EPS = 1e-6

# change alpha_a_prior would change the distribution of posterior
ALPHA_a_PRIOR = 1
# change alpha_b_prior would change the number of states K
ALPHA_b_PRIOR = 0.01

GAMMA_a_PRIOR = 2
GAMMA_b_PRIOR = 1

class HDPHMM:
    def __init__(self,  alpha=None, gamma=None, beta=None, rho0=0):
        self.alpha = alpha if alpha is not None else np.random.gamma(ALPHA_a_PRIOR, 1 / ALPHA_b_PRIOR)
        self.gamma = gamma if gamma is not None else np.random.gamma(GAMMA_a_PRIOR, 1 / GAMMA_b_PRIOR)

        # the kappa parameter in direct assignment model
        self.rho = rho0

        beta_vec = beta if beta is not None else np.random.dirichlet(np.array([1, self.gamma]), size=1)[0]
        self.beta_new = beta_vec[-1]
        self.beta_vec = beta_vec[:-1]

    def hidden_states_posterior_with_last_state(self, last_state: int, observation, transition_count: NDArray, K: int,
                                                emission_func):
        tmp_vec = np.arange(K)
        # p(z_t = k|params)
        current_hidden_state_dist = (
                (self.alpha * self.beta_vec + transition_count[last_state] + self.rho * (last_state == tmp_vec))
                / (self.alpha + transition_count[last_state].sum() + self.rho))

        new_hidden_state_dist = (self.alpha ** 2) * self.beta_new / (
                self.alpha + transition_count[last_state].sum() + self.rho)

        emission_pdf, emission_pdf_new = emission_func()
        # prob of yt[t] give the normal distribution, both yt_dist and yt_knew_dist are arrays with length K
        observation_dist = emission_pdf(observation)
        # print("observation_dist:", observation_dist)
        if np.any(observation_dist < 0):
            raise ValueError("Probabilities in observation_dist must be greater than 0")
        # new hidden state (cluster) with new observation emission pdf
        observation_dist_with_new = emission_pdf_new(observation)
        # print("observation_dist_with_new:", observation_dist)

        # construct z's posterior over k
        # add new column at the end of distribution array
        hidden_states_posterior = np.hstack((current_hidden_state_dist * observation_dist,
                                             new_hidden_state_dist * observation_dist_with_new))
        # normalise the new distribution array
        hidden_states_posterior = hidden_states_posterior / hidden_states_posterior.sum()

        return hidden_states_posterior

    def hidden_states_posterior_with_next_state(self, next_state: int, observation, transition_count: NDArray, K: int,
                                                emission_func):
        tmp_vec = np.arange(K)
        # p(z_t = k|params)
        next_hidden_state_dist = (
                (self.alpha * self.beta_vec[next_state] + transition_count[:, next_state] + self.rho * (
                        next_state == tmp_vec)) / (self.alpha + transition_count.sum(axis=1) + self.rho))


        emission_pdf, emission_pdf_new = emission_func()
        # prob of yt[t] give the normal distribution, both yt_dist and yt_knew_dist are arrays with length K
        observation_dist = emission_pdf(observation)
        # print("observation_dist:", observation_dist)
        if np.any(observation_dist < 0):
            raise ValueError("Probabilities in observation_dist must be greater than 0")

        # construct z's posterior over k
        # add new column at the end of distribution array
        hidden_states_posterior = next_hidden_state_dist * observation_dist
        # normalise the new distribution array
        hidden_states_posterior = hidden_states_posterior / hidden_states_posterior.sum()

        return hidden_states_posterior

    def hidden_states_posterior(self, last_state: int, next_state: int, observation, transition_count, emission_count, K: int,
                                emission_func):
        """
        :param transition_count:transition_counts[i][j] number of transitions from state i to state j
        :param emission_count:
        :param next_state: l
        :param last_state: j
        :param observation: current observation at step t (t is the index of direct assignment model)
        :param K: Current number of states in-use
        :param emission_func: the PARTIAL function of the pdf of emission distribution (e.g. norm.pdf), other params are specified be the caller.
        :return:
        """

        tmp_vec = np.arange(K)
        # p(z_t = k|params)
        # print(self.beta_vec.shape)
        # print(transition_count.shape)
        # print(tmp_vec.shape)
        current_hidden_state_dist = (
                (self.alpha * self.beta_vec + transition_count[last_state] + self.rho * (last_state == tmp_vec))
                / (self.alpha + transition_count[last_state].sum() + self.rho))
        if np.any(current_hidden_state_dist < 0):
            print(current_hidden_state_dist)
            print(transition_count[last_state])
            raise ValueError("Probabilities in hidden_state(last) must be greater than 0")

        # p(z_t+1 = l|params)
        next_hidden_state_dist = (
                (self.alpha * self.beta_vec[next_state] + transition_count[:, next_state] + self.rho * (
                        next_state == tmp_vec) + (
                         last_state == next_state) * (last_state == tmp_vec))
                / (self.alpha + transition_count.sum(axis=1) + self.rho + (last_state == tmp_vec)))
        if np.any(next_hidden_state_dist < 0):
            print(last_state, next_state)
            print(next_hidden_state_dist)
            print(transition_count)
            raise ValueError("Probabilities in hidden_states(next) must be greater than 0")

        emission_pdf, emission_pdf_new = emission_func()
        # prob of yt[t] give the normal distribution, both yt_dist and yt_knew_dist are a single float
        observation_dist = emission_pdf(observation)
        if np.any(observation_dist < 0):
            print(observation)
            print(emission_count[observation])
            print(observation_dist)
            raise ValueError("Probabilities in emission must be greater than 0")

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
        b = np.random.beta(1, self.gamma, size=1)
        self.beta_vec = np.hstack((self.beta_vec, b * self.beta_new))
        self.beta_new = (1 - b) * self.beta_new

    def update_beta_with_new_params(self, param_vec):
        self.beta_vec = np.random.dirichlet(np.hstack((param_vec, self.gamma)), size=1)[0]
        self.beta_new = self.beta_vec[-1]
        self.beta_vec = self.beta_vec[:-1]

    def update_alpha(self, transition_row_sum, m_total_sum):
        concentration = self.alpha + self.rho

        r_vec = []
        for val in transition_row_sum:
            if val > 0:
                r_vec.append(np.random.beta(concentration + 1, val))
        r_vec = np.array(r_vec)
        r_vec = r_vec.reshape(-1)  # flatten the array for multinomial cases

        s_vec = np.random.binomial(1, transition_row_sum / (transition_row_sum + concentration))
        s_vec = np.array(s_vec).reshape(-1) # flatten the array for multinomial cases

        # minus 1 here is to offset the additional one added on m_mat[0, 0] ?
        # I added "- self.rho" to be consistent with Berkley's notes, but this is inconsistent with original codes.
        self.alpha = np.random.gamma(ALPHA_a_PRIOR + m_total_sum - 1 - sum(s_vec),
                              1 / (ALPHA_b_PRIOR - sum(np.log(r_vec + CONST_EPS))))

    def update_gamma(self, m_total_sum, K):
        """
        Zhou's code is really inconsistent with Berkley's notes
        :param m_total_sum:
        :param K:
        :return:
        """
        eta = np.random.beta(self.gamma + 1, m_total_sum)

        # indicator = np.random.binomial(1, m_total_sum / (m_total_sum + self.gamma))
        indicator = (GAMMA_a_PRIOR + K - 1) / (GAMMA_a_PRIOR+ K - 1 + m_total_sum * (GAMMA_b_PRIOR - np.log(eta + CONST_EPS)))

        if indicator:
            self.gamma = np.random.gamma(GAMMA_a_PRIOR + K, 1 / (GAMMA_a_PRIOR - np.log(eta + CONST_EPS)))
        else:
            self.gamma = np.random.gamma(GAMMA_a_PRIOR + K - 1, 1 / (GAMMA_b_PRIOR - np.log(eta + CONST_EPS)))

        # alternative solution (still different from Zhou's implementation)
        # pi_m = (self.gamma_a_prior + K - 1) / (m_total_sum * (self.gamma_b_prior - np.log(eta + CONST_EPS)))
        # self.gamma = pi_m * np.gamma(self.gamma_a_prior + K, 1 / (self.gamma_b_prior - np.log(eta + CONST_EPS))) + \
        #              (1 - pi_m) * np.gamma(self.gamma_a_prior + K - 1, 1 / (self.gamma_b_prior - np.log(eta + CONST_EPS)))
