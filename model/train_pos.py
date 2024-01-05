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
print("Dataset size: ", len(token_indices), file=sys.stderr)    # 3903 sentences
print("Vocabulary Size: ", vocab_size, file=sys.stderr)
# print(token_indices[:10])

# TODO: split after preprocessing -- still possible that words in test_sentences not seen after training
train_sentences, test_sentences, train_tags, test_tags = train_test_split(
    token_indices, pos_tags, test_size=0.2, random_state=42)


loglik_test_sample = []
hidden_states_sample = []
hyperparams_sample = []

np.set_printoptions(suppress=True, precision=4)
np.set_printoptions(linewidth=180)
np.set_printoptions(formatter={'int': '{:5d}'.format})

if __name__ == "__main__":
    if token_indices == [] or pos_tags == []:
        raise ValueError("Failed to load input!")
    iterations = 50
    model = HDPHMM()
    sampler = DirectAssignmentPOS(model, train_sentences, vocab_size)

    for iteration in tqdm.tqdm(range(iterations), desc="training sampler:"):
        # first iteration as burn-in
        if iteration == 0:
            for index in range(len(train_sentences)):
                # sampler.initialise_first_state(index)
                for t in range(1, sampler.seq_length[index]):
                    sampler.sample_one_step_ahead(index, t)
            print("Burn-in K:", sampler.K)
            print("transition count:", sampler.transition_count)
        else:
            for index in range(len(train_sentences)):
                # sampler.sample_hidden_states_on_next_state(index, 0)
                for t in range(1, sampler.seq_length[index] - 1):
                    sampler.sample_hidden_states_on_last_next_state(index, t)
                sampler.sample_hidden_states_on_last_state(index, sampler.seq_length[index] - 1)
                if np.any(sampler.transition_count < 0):
                    print(index)
                    raise ValueError("Negative transition count -- outside")

            sampler.update_K()
            # print("hidden states after update K:", sampler.hidden_states[:5])
            print("new K: ", sampler.K)
            sampler.sample_m()
            sampler.sample_beta()
            sampler.sample_alpha()
            sampler.sample_gamma()
            # print("alpha: ", sampler.model.alpha, "gamma: ", sampler.model.gamma)

            # if iteration % 10 == 0:
            #     hidden_states_sample.append(sampler.hidden_states.copy())
            #     hyperparams_sample.append(np.array([sampler.model.alpha, sampler.model.gamma]))
            # print("shape:" , sampler.token_state_matrix.shape)

        # print(sampler.token_state_matrix.dtype)
        print("token_state_matrix:")
        for i in range(10, 20):
            print(sampler.token_state_matrix[i].astype(int))

        """
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
        """


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