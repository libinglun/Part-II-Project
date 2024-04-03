import numpy as np
import tqdm
from model.hdp_hmm import HDPHMM
from model.direct_assign_gibbs_multinomial import DirectAssignmentMultinomial
from utils.utils import compute_cost, flatten, calculate_variation_of_information
from utils.const import LOAD_PATH
from logger import mylogger


seed_vec = [111, 222, 333, 444, 555, 666, 777, 888, 999, 1000]
seed = 7  # random seed
np.random.seed(seed_vec[seed])  # fix randomness

# TODO: add system argument to distinguish multinomial and gaussian src
# emission_model = 'gaussian'

emission_model = 'multinomial'
file_name = "fix_8states_" + emission_model + "_same_trans_diff_stick"
train_data = np.load('./data/' + file_name + '.npz')
test_data = np.load('./data/test_' + file_name + '.npz')
real_hidden_states = train_data['zt']
real_observations = train_data['yt']
test_observations = test_data['yt']

# dataset_path = LOAD_PATH + "hmm_train_dataset(state-8_obs-8_length-5000).npz"
# loaded_npz = np.load(dataset_path, allow_pickle=True)
# real_hidden_states = loaded_npz['train_hidden_states']
# real_observations = loaded_npz['train_observations']
# print(real_observations[0].shape)
# test_observations = loaded_npz['test_observations']

loglik_test_sample = []
loss_sample = []
k_sample = []
hidden_states_sample = []
hyperparams_sample = []

if __name__ == "__main__":
    iterations = 200
    model = HDPHMM()

    sampler = DirectAssignmentMultinomial(model, real_observations)

    # the hidden states are empty initially. Fill in hidden states for the first iteration only based on last state j
    for t in range(1, sampler.seq_length):
        sampler.sample_one_step_ahead(t)

    # print(sampler.hidden_states)

    for iteration in tqdm.tqdm(range(iterations), desc="training model:"):
        mylogger.info(f"iteration {iteration} ----------------------------------------------------")
        if iteration == 0:
            mismatch_vec = (np.sum(sampler.hidden_states != real_hidden_states))
            zero_one_loss = round(mismatch_vec / len(real_hidden_states) * 100, 3)
            loss_sample.append(zero_one_loss)
            mylogger.info(f"Zero one loss rate is : {zero_one_loss}%")


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
            mylogger.info(f"New updated K is: {sampler.K}")
            k_sample.append(sampler.K)
            # print(f"New updated K is: {sampler.K}")

            sampler.sample_transition_distribution()
            # calculate the log likelihood of test observation sequence based on the new sampled transition distribution and result of direct assignment sampling (every 10 iters)
            _, loglik = sampler.compute_log_marginal_likelihood(test_observations)
            # output a matrix a_mat, a_mat[i, j] represents the probability of state j at time stamp i
            loglik_test_sample.append(loglik)
            mylogger.info(f"The log likelihood of test observation is: {loglik}")
            # print(f"The log likelihood of test observation is: {loglik}")

            # save the result of sampled hidden states and hyperparameter (every 10 iters)
            hidden_states_sample.append(sampler.hidden_states.copy())
            hyperparams_sample.append(np.array([sampler.model.alpha, sampler.model.gamma]))

            cost, indexes = compute_cost(sampler.hidden_states.copy(), real_hidden_states)
            dic = dict((v, k) for k, v in indexes)
            mylogger.info(dic)
            tmp = np.array([dic[sampler.hidden_states[t]] for t in range(len(hidden_states_sample[0]))])

            mismatch_vec = (np.sum(tmp != real_hidden_states))
            zero_one_loss = round(mismatch_vec / len(real_hidden_states) * 100, 3)
            loss_sample.append(zero_one_loss)
            mylogger.info(f"Zero one loss rate is : {zero_one_loss}%")
            # print(f"Zero one loss rate is : {round(mismatch_vec / len(real_hidden_states) * 100, 3)}%")

    print(loss_sample)


# mismatch_vec = []
# zt_sample_permute = []
# K_real = len(np.unique(real_hidden_states))
# for ii in range(len(hidden_states_sample)):
#     cost, indexes = compute_cost(hidden_states_sample[ii], real_hidden_states)
#     dic = dict((v, k) for k, v in indexes)
#     tmp = np.array([dic[hidden_states_sample[ii][t]] for t in range(len(hidden_states_sample[0]))])
#
#     zt_sample_permute.append(tmp.copy())
#     mismatch_vec.append(np.sum(tmp != real_hidden_states))
#
# print(mismatch_vec, loglik_test_sample)

# save results
# seed = int((int(sys.argv[1])-1)%10);
# np.savez(rlt_path + file_name + '_full_bayesian_rlt_' + str(seed) + '.npz', zt=hidden_states_sample, hyper=hyperparams_sample,
#          hamming=mismatch_vec, zt_permute=zt_sample_permute, loglik=loglik_test_sample)
