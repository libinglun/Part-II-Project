from hdp_hmm import HDPHMM
import numpy as np
import tqdm
from direct_assign_gibbs_pos import DirectAssignmentPOS

seed_vec = [111, 222, 333, 444, 555, 666, 777, 888, 999, 1000]
seed = 0  # random seed
np.random.seed(seed_vec[seed])  # fix randomness

np.set_printoptions(suppress=True, precision=4)
np.set_printoptions(linewidth=180)
np.set_printoptions(formatter={'int': '{:5d}'.format})

num_states = 10
num_observations = 100

def euclidean_distance(A, B):
    return np.sqrt(np.sum((A - B) ** 2))


def kl_divergence(P, Q):
    mask = (P != 0) & (Q != 0)
    filtered_P = P[mask]
    filtered_Q = Q[mask]
    return np.sum(filtered_P * np.log(filtered_P / filtered_Q))

def prepare_dataset():
    loaded_npz = np.load("../data/hmm_synthetic_dataset.npz", allow_pickle=True)
    observations = list(loaded_npz['observation'])
    real_hidden_states = list(loaded_npz['real_hidden'])
    noisy_hidden_states = list(loaded_npz['noisy_hidden'])
    real_trans_dist = np.vstack(loaded_npz['real_trans'])

    size = 50000

    real_trans_count = np.zeros((num_states, num_states), dtype='int')
    noisy_trans_count = np.zeros((num_states, num_states), dtype='int')
    emis_count = np.zeros((num_observations, num_states), dtype='int')

    for i in range(size):
        for t in range(len(observations[i])):
            emis_count[observations[i][t], noisy_hidden_states[i][t]] += 1
            if t > 0:
                noisy_trans_count[noisy_hidden_states[i][t - 1], noisy_hidden_states[i][t]] += 1
                real_trans_count[real_hidden_states[i][t - 1], real_hidden_states[i][t]] += 1
    print("real trans count: \n", real_trans_count)
    print("noisy trans count: \n", noisy_trans_count)
    print(np.sum(noisy_trans_count))
    print(euclidean_distance(real_trans_count, noisy_trans_count))

    return noisy_hidden_states, observations, noisy_trans_count, emis_count, real_trans_count, real_trans_dist
    # print(emis_count)

def train_sampler(hidden_states, observations, transition_count, emission_count, real_transition_count, real_trans_dist):
    model = HDPHMM()
    sampler = DirectAssignmentPOS(model, observations, num_observations)
    sampler.K = 10
    sampler.transition_count = transition_count.copy()
    sampler.token_state_matrix = emission_count.copy()
    sampler.hidden_states = hidden_states.copy()
    for i in range(9):
        sampler.model.update_beta_with_new_state()

    iterations = 20
    for iter in tqdm.tqdm(range(iterations), desc="training sampler:"):
        for index in range(len(observations)):
            for t in range(1, sampler.seq_length[index] - 1):
                sampler.sample_hidden_states_on_last_next_state(index, t)
            sampler.sample_hidden_states_on_last_state(index, sampler.seq_length[index] - 1)

        sampler.update_K()
        # print("hidden states after update K:", sampler.hidden_states[:5])
        print("new K: ", sampler.K)
        sampler.sample_m()
        sampler.sample_beta()
        sampler.sample_alpha()
        sampler.sample_gamma()
        # print(f"iteration {iter} has transition counts {sampler.transition_count.sum()} in total: \n {sampler.transition_count}")
        count_distance = euclidean_distance(sampler.transition_count[:10, :10], real_transition_count)
        print(f"Distance between sampled and real transition counts is {count_distance}")

        sampler.sample_transition_distribution()
        trans_distance = euclidean_distance(sampler.pi_mat[:10, :10], real_trans_dist)
        trans_KL_divergence = kl_divergence(sampler.pi_mat[:10, :10], real_trans_dist)
        print(f"Distance between sampled and real transition distribution is {trans_distance}")
        print(f"KL Divergence between sampled and real transition distribution is {trans_KL_divergence}")


if __name__ == "__main__":
    train_hidden, train_observed, noisy_trans_count, emis_count, real_trans_count, trans_dist = prepare_dataset()
    train_sampler(train_hidden, train_observed, noisy_trans_count, emis_count, real_trans_count, trans_dist)