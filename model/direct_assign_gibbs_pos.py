import numpy as np
import scipy.stats as stats

class DirectAssignmentPOS:
    def __init__(self, model, observations, vocab_size):
        # observations are all observation of training set
        self.observations = observations
        self.dataset_length = len(observations)
        # a list of sequence_length
        self.seq_length = [len(obs) for obs in self.observations]
        # matrix of number of sequence * sequence_length
        self.hidden_states = [np.zeros(seq_len, dtype='int') for seq_len in self.seq_length]

        # tokens are indices of word
        self.vocab_size = vocab_size
        # self.token2count = self.initialize_emission_matrix(dataset, self.vocab_size)
        self.token_state_matrix = np.zeros((self.vocab_size, 1), dtype='int')
        print(self.token_state_matrix.shape)

        self.model = model
        self.transition_count = np.array([[0]])  # (n_mat)
        self.m_mat = None
        self.pi_mat = None
        self.K = 1

    def emission_pdf(self):
        # smoothing for 0 counts + uniform distribution for a new state since all 0 counts
        return (lambda x: (self.token_state_matrix[x] + 1) / (self.token_state_matrix[x].sum() + self.K)), (lambda x: 1 / self.K)

    # def new_observation(self, new_observation, hidden_states):
    #     # reinitialise observations and hidden states for each new sentence
    #     self.observations = new_observation
    #     self.seq_length = len(new_observation)
    #     if hidden_states is None:
    #         self.hidden_states = np.zeros(self.seq_length, dtype='int')
    #     else:
    #         self.hidden_states = hidden_states
    #     # TODO: reinitialise transition_count or not?
    #     # self.transition_count = np.zeros((self.K, self.K))


    # TODO: use sample_one_step_ahead as burn-in, don't need to initialise!!!
    def sample_one_step_ahead(self, index, t):
        # j is the hidden state value, ranging from 0 to T - 1
        last_state = self.hidden_states[index][t - 1]

        # derive the current hidden state posterior over K states
        posterior = self.model.hidden_states_posterior_with_last_state(last_state, self.observations[index][t],
                                                                       self.transition_count, self.K,
                                                                       self.emission_pdf)

        # print("posterior:", posterior)

        # update the current hidden state by multinomial
        self.hidden_states[index][t] = np.where(np.random.multinomial(1, posterior))[0][0]

        # update beta_vec, n_mat when having a new state
        have_new_state = (self.hidden_states[index][t] == self.K)

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

        self.transition_count[last_state, self.hidden_states[index][t]] += 1
        self.token_state_matrix[self.observations[index][t]][self.hidden_states[index][t]] += 1
        # print(self.token_state_matrix.sum())

    def sample_hidden_states_on_last_state(self, index, t):
        # j is the hidden state value, ranging from 0 to T - 1
        last_state = self.hidden_states[index][t - 1]

        # exclude the counts of the current state
        self.transition_count[last_state, self.hidden_states[index][t]] -= 1
        assert np.any(self.transition_count > 0), "Negative transition count"

        # print("before decrement: " , self.token_state_matrix[self.observations[t]], self.hidden_states[t])
        self.token_state_matrix[self.observations[index][t]][self.hidden_states[index][t]] -= 1
        # print("after decrement: ", self.token_state_matrix[self.observations[t]])

        # derive the current hidden state posterior over K states
        posterior = self.model.hidden_states_posterior_with_last_state(last_state, self.observations[index][t],
                                                                       self.transition_count, self.K,
                                                                       self.emission_pdf)

        if np.any(posterior < 0):
            raise ValueError("Probabilities in posterior must be greater than 0")
        if np.any(posterior > 1):
            raise ValueError("Probabilities in posterior must be smaller than 1")
        if np.any(np.isnan(posterior)):
            raise ValueError("Posterior contains NaNs")

        # update the current hidden state by multinomial
        self.hidden_states[index][t] = np.where(np.random.multinomial(1, posterior))[0][0]

        # update beta_vec, n_mat when having a new state
        have_new_state = (self.hidden_states[index][t] == self.K)

        if have_new_state:
            self.model.update_beta_with_new_state()
            # Extend the transition matrix with the new state
            self.transition_count = np.hstack((self.transition_count, np.zeros((self.K, 1))))
            self.transition_count = np.vstack((self.transition_count, np.zeros((1, self.K + 1))))
            self.token_state_matrix = np.hstack((self.token_state_matrix, np.zeros((self.vocab_size, 1))))

            self.K += 1

        self.transition_count[last_state, self.hidden_states[index][t]] += 1
        self.token_state_matrix[self.observations[index][t]][self.hidden_states[index][t]] += 1

    def sample_hidden_states_on_last_next_state(self, index, t):
        # define last_state(j), next_state(l)
        last_state = self.hidden_states[index][t - 1]
        next_state = self.hidden_states[index][t + 1]

        if np.any(self.transition_count < 0):
            print("QAQ")

        # exclude the counts of the current state
        # print("transition count:", self.transition_count)
        self.transition_count[last_state, self.hidden_states[index][t]] -= 1
        self.transition_count[self.hidden_states[index][t], next_state] -= 1
        assert np.any(self.transition_count > 0), "Negative transition count"

        # print("observation: ", self.observations[index])
        # print("before decrement: ", self.token_state_matrix[self.observations[index][t]], self.hidden_states[index][t], self.K)
        self.token_state_matrix[self.observations[index][t]][self.hidden_states[index][t]] -= 1
        # print("after decrement: ", self.token_state_matrix[self.observations[index][t]])

        # derive the current hidden state posterior over K states
        posterior = self.model.hidden_states_posterior(last_state, next_state, self.observations[index][t],
                                                                       self.transition_count, self.K,
                                                                       self.emission_pdf)

        # print(posterior)
        if np.any(posterior < 0):
            print(posterior)
            raise ValueError("Probabilities in posterior must be greater than 0")
        if np.any(posterior > 1):
            raise ValueError("Probabilities in posterior must be smaller than 1")
        if np.any(np.isnan(posterior)):
            raise ValueError("Posterior contains NaNs")

        # update the current hidden state by multinomial
        self.hidden_states[index][t] = np.where(np.random.multinomial(1, posterior))[0][0]

        # update beta_vec, n_mat when having a new state
        have_new_state = (self.hidden_states[index][t] == self.K)

        if have_new_state:
            self.model.update_beta_with_new_state()
            # Extend the transition matrix with the new state
            self.transition_count = np.hstack((self.transition_count, np.zeros((self.K, 1))))
            self.transition_count = np.vstack((self.transition_count, np.zeros((1, self.K + 1))))
            self.token_state_matrix = np.hstack((self.token_state_matrix, np.zeros((self.vocab_size, 1))))

            self.K += 1

        if np.any(self.transition_count < 0):
            print("index: ", index)
            raise ValueError("Negative transition count -- ?")

        self.transition_count[last_state, self.hidden_states[index][t]] += 1
        self.transition_count[self.hidden_states[index][t], next_state] += 1
        # print("before increment: ", self.token_state_matrix[self.observations[index][t]])
        self.token_state_matrix[self.observations[index][t]][self.hidden_states[index][t]] += 1
        # print("after increment: ", self.token_state_matrix[self.observations[index][t]])

    def update_K(self):
        # remain_index = sorted unique elements of hidden states
        remain_index = np.unique(np.concatenate(self.hidden_states)).astype(int)

        # a dictionary with key = state, value = the sorted index (map old states to new states)
        state_mapping = {state: idx for idx, state in enumerate(remain_index)}
        # use indexes as the new states
        for i in range(self.dataset_length):
            self.hidden_states[i] = np.array([state_mapping[state] for state in self.hidden_states[i]])

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