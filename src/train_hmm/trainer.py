import numpy as np
import tqdm
from utils.utils import compute_cost


loglik_test_sample = []
hidden_states_sample = []
hyperparams_sample = []

def train_sampler(sampler, args, dataset):
    iterations = args.iter

    # the hidden states are empty initially. Fill in hidden states for the first iteration only based on last state j
    for t in range(1, sampler.seq_length):
        sampler.sample_one_step_ahead(t)

    for iteration in tqdm.tqdm(range(iterations), desc="training model:"):
        for t in range(1, sampler.seq_length - 1):
            sampler.sample_hidden_states_on_last_next_state(t)
        sampler.sample_hidden_states_on_last_state(sampler.seq_length - 1)
        sampler.update_K()
        sampler.sample_m()
        sampler.sample_beta()
        sampler.sample_alpha()
        sampler.sample_gamma()

        # sample transition distribution matrix based on the result of direct assignment sampling (every 10 iters)
        if iteration % 10 == 0:
            sampler.sample_transition_distribution()
            # calculate the log likelihood of test observation sequence based on the new sampled transition distribution and result of direct assignment sampling (every 10 iters)
            _, loglik = sampler.compute_log_marginal_likelihood(dataset.test_obs)
            # output a matrix a_mat, a_mat[i, j] represents the probability of state j at time stamp i
            loglik_test_sample.append(loglik)

            # save the result of sampled hidden states and hyperparameter (every 10 iters)
            hidden_states_sample.append(sampler.hidden_states.copy())
            hyperparams_sample.append(np.array([sampler.model.alpha, sampler.model.gamma]))


mismatch_vec = []
zt_sample_permute = []
K_real = len(np.unique(real_hidden_states))
for ii in range(len(hidden_states_sample)):
    cost, indexes = compute_cost(hidden_states_sample[ii], real_hidden_states)
    dic = dict((v, k) for k, v in indexes)
    tmp = np.array([dic[hidden_states_sample[ii][t]] for t in range(len(hidden_states_sample[0]))])

    zt_sample_permute.append(tmp.copy())
    mismatch_vec.append(np.sum(tmp != real_hidden_states))

print(mismatch_vec, loglik_test_sample)

# save results
# seed = int((int(sys.argv[1])-1)%10);
# np.savez(rlt_path + file_name + '_full_bayesian_rlt_' + str(seed) + '.npz', zt=hidden_states_sample, hyper=hyperparams_sample,
#          hamming=mismatch_vec, zt_permute=zt_sample_permute, loglik=loglik_test_sample)
