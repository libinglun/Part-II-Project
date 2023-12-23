import numpy as np
import scipy.stats as stats
from direct_assign_gibbs_base import DirectAssignment

class DirectAssignmentPOS(DirectAssignment):
    def __init__(self, model, observations, vocab_size):
        DirectAssignment.__init__(self, model, observations)

        # tokens are indices of word
        self.vocab_size = vocab_size
        # self.token2count = self.initialize_emission_matrix(dataset, self.vocab_size)
        self.token_state_matrix = np.zeros((self.vocab_size, 1), dtype='int')
        print(self.token_state_matrix.shape)

    # def initialize_emission_matrix(self, dataset: list[list[int]], vocab_size):
    #     token2count = np.zeros((vocab_size, 1))
    #     for sentence in dataset:
    #         for word in sentence:
    #             token2count[word] += 1
    #
    #     return token2count


    def emission_pdf(self):
        # smoothing for 0 counts + uniform distribution for a new state since all 0 counts
        return (lambda x: (self.token_state_matrix[x] + 1) / (self.token_state_matrix[x].sum() + self.K)), (lambda x: 1 / self.K)

    def new_observation(self, new_observation):
        # reinitialise observations and hidden states for each new sentence
        self.observations = new_observation
        self.seq_length = len(new_observation)
        self.hidden_states = np.zeros(self.seq_length, dtype='int')
        # TODO: reinitialise transition_count or not?
        self.transition_count = np.zeros((self.K, self.K))


    # TODO: use sample_one_step_ahead as burn-in, don't need to initialise!!!
    def sample_one_step_ahead(self, t):
        # j is the hidden state value, ranging from 0 to T - 1
        last_state = self.hidden_states[t - 1]

        # derive the current hidden state posterior over K states
        posterior = self.model.hidden_states_posterior_with_last_state(last_state, self.observations[t],
                                                                       self.transition_count, self.K,
                                                                       self.emission_pdf)

        # print("posterior:", posterior)

        # update the current hidden state by multinomial
        self.hidden_states[t] = np.where(np.random.multinomial(1, posterior))[0][0]

        # update beta_vec, n_mat when having a new state
        have_new_state = (self.hidden_states[t] == self.K)

        if have_new_state:
            self.model.update_beta_with_new_state()
            # Extend the transition matrix with the new state
            self.transition_count = np.hstack((self.transition_count, np.zeros((self.K, 1))))
            self.transition_count = np.vstack((self.transition_count, np.zeros((1, self.K + 1))))
            # initialise a new column of token_state_matrix as a 0 vector
            self.token_state_matrix = np.hstack((self.token_state_matrix, np.zeros((self.vocab_size, 1))))

            # Add a new state
            self.K += 1

        # print("K:", self.K)

        self.transition_count[last_state, self.hidden_states[t]] += 1
        self.token_state_matrix[self.observations[t]][self.hidden_states[t]] += 1
        # print(self.token_state_matrix.sum())

    def sample_hidden_states_on_last_state(self, t):
        # j is the hidden state value, ranging from 0 to T - 1
        last_state = self.hidden_states[t - 1]

        # exclude the counts of the current state
        self.transition_count[last_state, self.hidden_states[t]] -= 1
        # print("before decrement: " , self.token_state_matrix[self.observations[t]], self.hidden_states[t])
        self.token_state_matrix[self.observations[t]][self.hidden_states[t]] -= 1
        # print("after decrement: ", self.token_state_matrix[self.observations[t]])

        # derive the current hidden state posterior over K states
        posterior = self.model.hidden_states_posterior_with_last_state(last_state, self.observations[t],
                                                                       self.transition_count, self.K,
                                                                       self.emission_pdf)

        if np.any(posterior < 0):
            raise ValueError("Probabilities in posterior must be greater than 0")
        if np.any(posterior > 1):
            raise ValueError("Probabilities in posterior must be smaller than 1")
        if np.any(np.isnan(posterior)):
            raise ValueError("Posterior contains NaNs")

        # update the current hidden state by multinomial
        self.hidden_states[t] = np.where(np.random.multinomial(1, posterior))[0][0]

        # update beta_vec, n_mat when having a new state
        have_new_state = (self.hidden_states[t] == self.K)

        if have_new_state:
            self.model.update_beta_with_new_state()
            # Extend the transition matrix with the new state
            self.transition_count = np.hstack((self.transition_count, np.zeros((self.K, 1))))
            self.transition_count = np.vstack((self.transition_count, np.zeros((1, self.K + 1))))
            self.token_state_matrix = np.hstack((self.token_state_matrix, np.zeros((self.vocab_size, 1))))

            self.K += 1

        self.transition_count[last_state, self.hidden_states[t]] += 1
        self.token_state_matrix[self.observations[t]][self.hidden_states[t]] += 1

    def sample_hidden_states_on_last_next_state(self, t):
        # define last_state(j), next_state(l)
        last_state = self.hidden_states[t - 1]
        next_state = self.hidden_states[t + 1]

        # exclude the counts of the current state
        self.transition_count[last_state, self.hidden_states[t]] -= 1
        self.transition_count[self.hidden_states[t], next_state] -= 1
        print("before decrement: " , self.token_state_matrix[self.observations[t]], self.hidden_states[t])
        self.token_state_matrix[self.observations[t]][self.hidden_states[t]] -= 1
        print("after decrement: ", self.token_state_matrix[self.observations[t]])

        # derive the current hidden state posterior over K states
        posterior = self.model.hidden_states_posterior(last_state, next_state, self.observations[t],
                                                                       self.transition_count, self.K,
                                                                       self.emission_pdf)

        # print(posterior)
        if np.any(posterior < 0):
            raise ValueError("Probabilities in posterior must be greater than 0")
        if np.any(posterior > 1):
            raise ValueError("Probabilities in posterior must be smaller than 1")
        if np.any(np.isnan(posterior)):
            raise ValueError("Posterior contains NaNs")

        # update the current hidden state by multinomial
        self.hidden_states[t] = np.where(np.random.multinomial(1, posterior))[0][0]

        # update beta_vec, n_mat when having a new state
        have_new_state = (self.hidden_states[t] == self.K)

        if have_new_state:
            self.model.update_beta_with_new_state()
            # Extend the transition matrix with the new state
            self.transition_count = np.hstack((self.transition_count, np.zeros((self.K, 1))))
            self.transition_count = np.vstack((self.transition_count, np.zeros((1, self.K + 1))))
            self.token_state_matrix = np.hstack((self.token_state_matrix, np.zeros((self.vocab_size, 1))))

            self.K += 1

        self.transition_count[last_state, self.hidden_states[t]] += 1
        self.transition_count[self.hidden_states[t], next_state] += 1
        self.token_state_matrix[self.observations[t]][self.hidden_states[t]] += 1

    def update_K(self):
        # rem_ind contains all the unique value of zt
        remain_index = np.unique(self.hidden_states)

        # a dictionary with key = state, value = the sorted index (map old states to new states)
        state_mapping = {k: v for v, k in enumerate(sorted(remain_index))}
        # use indexes as the new states
        self.hidden_states = np.array([state_mapping[state] for state in self.hidden_states])

        # only select rows and columns of n_mat according to the index of rem_ind

        self.transition_count = self.transition_count[remain_index][:, remain_index]
        # self.transition_count = self.transition_count[np.ix_(remain_index, remain_index)]
        # TODO: check token_state_matrix
        self.token_state_matrix = self.token_state_matrix[:][:, remain_index]
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

        # TODO: the meaning of new cluster here
        # varn = np.hstack(
        #     (np.sqrt((self.sigma0 ** 2) + varn), np.sqrt((self.sigma0 ** 2) + (self.sigma_prior ** 2))))
        # mun = np.hstack((mun, self.mu0))

        for t in range(length):
            if t == 0 and start_point == -1:
                j = 0
                # print("test_observations: ",test_observations[t])
                # print("token_state_matrix: ", self.token_state_matrix[test_observations[t]])
                a_mat[t + 1, j] = (self.token_state_matrix[test_observations[t]][j] + 1) / (self.token_state_matrix[test_observations[t]].sum() + self.K)
            else:
                # TODO: change range from K + 1 to K since delete one cluster as above
                for j in range(self.K):
                    a_mat[t + 1, j] = (sum(a_mat[t, :] * self.pi_mat[:, j]) *
                                       (self.token_state_matrix[test_observations[t]][j] + 1) / (self.token_state_matrix[test_observations[t]].sum() + self.K))
            c_vec[t] = sum(a_mat[t + 1, :])
            a_mat[t + 1, :] /= c_vec[t]

        log_marginal_lik = sum(np.log(c_vec))
        # output a matrix a_mat, a_mat[i, j] represents the probability of state j at time stamp i
        return a_mat, log_marginal_lik