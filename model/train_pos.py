import sys

import numpy as np
import tqdm
from hdp_hmm import HDPHMM
from direct_assign_gibbs_pos import DirectAssignmentPOS
from sklearn.model_selection import train_test_split
from utils import compute_cost
import nltk
from nltk.corpus import treebank
from NLTK_treebank import Lang
nltk.download('treebank')

seed_vec = [111, 222, 333, 444, 555, 666, 777, 888, 999, 1000]

seed = 0  # random seed
np.random.seed(seed_vec[seed])  # fix randomness

treebank_lang = Lang("Penn Treebank", treebank.tagged_sents())
token_indices, pos_tags = treebank_lang.build_dataset()
vocab_size = treebank_lang.nth_words # index: 0 ~ 5750
# print("Dataset: ", token_indices[2000], file=sys.stderr)
print("Vocabulary Size: ", vocab_size, file=sys.stderr)

# TODO: split after preprocessing -- still possible that words in test_sentences not seen after training
train_sentences, test_sentences, train_tags, test_tags = train_test_split(
    token_indices, pos_tags, test_size=0.2, random_state=42)


loglik_test_sample = []
hidden_states_sample = []
hyperparams_sample = []

np.set_printoptions(suppress=True, precision=4)

if __name__ == "__main__":
    if token_indices == [] or pos_tags == []:
        print("Failed to load input!", file=sys.stderr)
    iterations = 200
    model = HDPHMM()
    sampler = DirectAssignmentPOS(model, token_indices[0], vocab_size)

    for iteration in tqdm.tqdm(range(iterations), desc="training sampler:"):
        # first iteration as burn-in
        if iteration == 0:
            for sentence in train_sentences:
                sampler.new_observation(sentence)
                for t in range(1, sampler.seq_length):
                    sampler.sample_one_step_ahead(t)
                    # print(sampler.K)
                # print("outside:", sampler.K)
            break
        else:
            for sentence in train_sentences:
                sampler.new_observation(sentence)
                for t in range(1, sampler.seq_length - 1):
                    sampler.sample_hidden_states_on_last_next_state(t)
                sampler.sample_hidden_states_on_last_state(sampler.seq_length - 1)
                sampler.update_K()
                sampler.sample_m()
                sampler.sample_beta()
                sampler.sample_alpha()
                sampler.sample_gamma()
                if iteration % 10 == 0:
                    hidden_states_sample.append(sampler.hidden_states.copy())
                    hyperparams_sample.append(np.array([sampler.model.alpha, sampler.model.gamma]))

        if iteration % 10 == 0:
            # sample transition distribution matrix based on the result of direct assignment sampling
            sampler.sample_transition_distribution()
            # calculate the log likelihood of test observation sequence based on the new sampled transition distribution and result of direct assignment sampling
            # TODO: compute log likelihood for the entire test dataset
            sum_loglik = 0
            for test_sentence in test_sentences:
                _, loglik = sampler.compute_log_marginal_likelihood(test_sentence)
                sum_loglik += loglik
            loglik_test_sample.append(sum_loglik)

            # save the result of sampled hyperparameter
            hyperparams_sample.append(np.array([sampler.model.alpha, sampler.model.gamma]))

    print(sampler.token_state_matrix.dtype)
    for i in range(10, 20):
        print(sampler.token_state_matrix[i].astype(int))

mismatch_vec = []
zt_sample_permute = []
# TODO: train_tags is a list of list, but want to have list type here
for tags in train_tags:
    K_real = len(np.unique(train_tags))

for ii in range(len(hidden_states_sample)):
    cost, indexes = compute_cost(hidden_states_sample[ii], train_tags)
    dic = dict((v, k) for k, v in indexes)
    tmp = np.array([dic[hidden_states_sample[ii][t]] for t in range(len(hidden_states_sample[0]))])

    zt_sample_permute.append(tmp.copy())
    mismatch_vec.append(np.sum(tmp != train_tags))

print(mismatch_vec, loglik_test_sample)