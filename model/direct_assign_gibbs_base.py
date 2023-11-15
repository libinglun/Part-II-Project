from abc import ABC, abstractmethod

from hdp_hmm import HDPHMM
import numpy as np


class DirectAssignment(ABC):
    def __init__(self, model, observations):
        self.model = model

        self.observations = observations
        self.seq_length = len(observations)
        self.hidden_states = np.zeros(self.seq_length, dtype='int')

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
    def update_emission_statistics(self, new, t):
        pass

    @abstractmethod
    def rearrange_emission_statistics(self, remain_index):
        pass

    def sample_hidden_states_on_last_state(self, t):
        # last time point

        # j is the hidden state value, ranging from 0 to T - 1
        last_state = self.hidden_states[t - 1]

        # derive the current hidden state posterior over K states
        posterior = self.model.hidden_states_posterior_with_last_state(last_state, self.observations[t],
                                                                       self.transition_count, self.K,
                                                                       self.emission_pdf)

        # update the current hidden state by multinomial
        self.hidden_states[t] = np.where(np.random.multinomial(1, posterior))[0][0]

        # update beta_vec, n_mat when having a new state
        have_new_state = self.hidden_states[t] == self.K
        self.update_emission_statistics(have_new_state, t)
        if have_new_state:
            self.model.update_beta_with_new_state()

            # Extend the transition matrix with the new state
            # a new column of zero is being added to the right side of n_mat
            self.transition_count = np.hstack((self.transition_count, np.zeros((self.K, 1))))
            # a new row of zero is being added to the bottom of n_mat
            self.transition_count = np.vstack((self.transition_count, np.zeros((1, self.K + 1))))

            # Add a new state
            self.K += 1

        self.transition_count[last_state, self.hidden_states[t]] += 1

    def sample_hidden_states_on_last_next_state(self, t):

        # define last_state(j), next_state(l)
        last_state = self.hidden_states[t - 1]
        next_state = self.hidden_states[t + 1]

        # exclude the counts of the current state
        self.transition_count[last_state, self.hidden_states[t]] -= 1
        self.transition_count[self.hidden_states[t], next_state] -= 1

        # derive the current hidden state posterior over K states
        posterior = self.model.hidden_states_posterior(last_state, next_state, self.observations[t],
                                                       self.transition_count, self.K, self.emission_pdf)

        # update the current hidden state by multinomial
        self.hidden_states[t] = np.where(np.random.multinomial(1, posterior))[0][0]

        have_new_state = self.hidden_states[t] == self.K
        self.update_emission_statistics(have_new_state, t)
        if have_new_state:
            self.model.update_beta_with_new_state()

            # Extend the transition matrix with the new state
            # a new column of zero is being added to the right side of n_mat
            self.transition_count = np.hstack((self.transition_count, np.zeros((self.K, 1))))
            # a new row of zero is being added to the bottom of n_mat
            self.transition_count = np.vstack((self.transition_count, np.zeros((1, self.K + 1))))

            # Add a new state
            self.K += 1

        self.transition_count[last_state, self.hidden_states[t]] += 1
        self.transition_count[self.hidden_states[t], next_state] += 1

    def update_K(self):
        # rem_ind contains all the unique value of zt
        remain_index = np.unique(self.hidden_states)

        # a dictionary with key = state, value = the sorted index (map old states to new states)
        state_mapping = {k: v for v, k in enumerate(sorted(remain_index))}
        # use indexes as the new states
        self.hidden_states = np.array([state_mapping[state] for state in self.hidden_states])

        # only select rows and columns of n_mat according to the index of rem_ind
        self.transition_count = self.transition_count[remain_index][:, remain_index]
        self.rearrange_emission_statistics(remain_index)
        self.model.beta_vec = self.model.beta_vec[remain_index]

        # update the new state space
        self.K = len(remain_index)

    def sample_m(self):
        self.m_mat = np.zeros((self.K, self.K))
        for j in range(self.K):
            for k in range(self.K):
                if self.transition_count[j, k] == 0:
                    continue
                    # self.m_mat[j, k] = 0 # original code, changed to continue since m_mat is initialised to 0
                else:
                    # move this to HDP_HMM, so that rho would be hidden from direct assignment sampler
                    x_vec = np.random.binomial(1, (
                            self.model.alpha * self.model.beta_vec[k] + self.model.rho * (j == k)) / (
                                                       np.arange(self.transition_count[j, k]) + self.model.alpha *
                                                       self.model.beta_vec[k] + self.model.rho * (j == k)))
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
            self.pi_mat[j] = np.random.dirichlet(prob_vec, size=1)[0]
        prob_vec = np.hstack(
            (self.model.alpha * self.model.beta_vec, self.model.alpha * self.model.beta_new + self.model.rho))
        prob_vec[prob_vec < 0.01] = 0.01  # clip step
        self.pi_mat[-1] = np.random.dirichlet(prob_vec, size=1)[0]



