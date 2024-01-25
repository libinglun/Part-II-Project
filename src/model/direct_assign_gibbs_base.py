from abc import ABC, abstractmethod

from .hdp_hmm import HDPHMM
import numpy as np


class DirectAssignment(ABC):
    def __init__(self, model, observations):
        self.model = model

        self.transition_count = np.array([[0]])  # (n_mat)
        self.m_mat = None
        self.pi_mat = None
        self.K = 1

    @abstractmethod
    def emission_pdf(self):
        pass

    @abstractmethod
    def compute_log_marginal_likelihood(self, test_observations, start_point):
        pass

    @abstractmethod
    def sample_one_step_ahead(self, t):
        pass

    def sample_hidden_states_on_last_state(self, t):
        pass

    @abstractmethod
    def sample_hidden_states_on_last_next_state(self, t):
        pass

    @abstractmethod
    def update_K(self):
        pass

    def sample_m(self):
        self.m_mat = np.zeros((self.K, self.K))

        for j in range(self.K):
            for k in range(self.K):
                if self.transition_count[j, k] == 0:
                    continue
                    # self.m_mat[j, k] = 0 # original code, changed to continue since m_mat is initialised to 0
                else:
                    # move this to HDP_HMM, so that rho would be hidden from direct assignment model
                    x_vec = np.random.binomial(1, (
                            self.model.alpha * self.model.beta_vec[k] + self.model.rho * (j == k)) / (
                                                       np.arange(self.transition_count[j, k]) + self.model.alpha *
                                                       self.model.beta_vec[k] + self.model.rho * (j == k)))
                    x_vec = np.array(x_vec).reshape(-1)
                    self.m_mat[j, k] = sum(x_vec)

        self.m_mat[0, 0] += 1

    def sample_beta(self):
        param_vec = np.array(self.m_mat.sum(axis=0))
        # print(self.m_mat.sum(axis=0))
        assert np.all(param_vec > 0), "All alpha values must be > 0"
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
            self.pi_mat[j] = np.random.dirichlet(prob_vec, size=1)[0]
        prob_vec = np.hstack(
            (self.model.alpha * self.model.beta_vec, self.model.alpha * self.model.beta_new + self.model.rho))
        prob_vec[prob_vec < 0.01] = 0.01  # clip step
        self.pi_mat[-1] = np.random.dirichlet(prob_vec, size=1)[0]



