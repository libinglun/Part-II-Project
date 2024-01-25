import numpy as np

from utils import euclidean_distance, difference
from const import LOAD_PATH

class Dataset:
    def __init__(self, real_hidden_states, noisy_hidden_states, real_trans_count, noisy_trans_count, real_trans_dist, observations, emis_count, total_count):
        self.real_hidden_states = real_hidden_states
        self.noisy_hidden_states = noisy_hidden_states
        self.real_trans_count = real_trans_count
        self.noisy_trans_count = noisy_trans_count
        self.real_trans_dist = real_trans_dist
        self.observations = observations
        self.emis_count = emis_count
        # self.total_count = total_count


def load_data(noisy_level, num_states, num_observations, size):
    dataset_path = LOAD_PATH + f"hmm_synthetic_dataset(noise-{noisy_level}_state-{num_states}_obs-{num_observations}_size-{size}).npz"
    loaded_npz = np.load(dataset_path, allow_pickle=True)
    observations = list(loaded_npz['observation'])
    real_hidden_states = list(loaded_npz['real_hidden'])
    noisy_hidden_states = list(loaded_npz['noisy_hidden'])
    real_trans_dist = np.vstack(loaded_npz['real_trans'])
    # noisy_level = loaded_npz['noisy_level']

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
    total_count = np.sum(noisy_trans_count)
    print(total_count)
    mis, tot = difference(real_hidden_states, noisy_hidden_states)
    print(f"The initial rate of missing states is:  {round(mis / tot * 100, 3)}%")
    # print(noisy_hidden_states[:3])
    # print(real_hidden_states[:3])
    print(euclidean_distance(real_trans_count, noisy_trans_count))

    return Dataset(real_hidden_states, noisy_hidden_states, real_trans_count, noisy_trans_count, real_trans_dist, observations, emis_count, total_count)