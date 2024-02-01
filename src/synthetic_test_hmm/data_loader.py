import numpy as np

from ..utils.utils import euclidean_distance, difference
from ..utils.const import LOAD_PATH

from ..logger import mylogger


class Dataset:
    def __init__(self, real_hidden_states, noisy_hidden_states, real_trans_count, noisy_trans_count, real_trans_dist, observations, real_emis_count, noisy_emis_count):
        self.real_hidden_states = real_hidden_states
        self.noisy_hidden_states = noisy_hidden_states
        self.real_trans_count = real_trans_count
        self.noisy_trans_count = noisy_trans_count
        self.real_trans_dist = real_trans_dist
        self.observations = observations
        self.real_emis_count = real_emis_count
        self.noisy_emis_count = noisy_emis_count


def load_data(noisy_level, num_states, num_observations, size):
    dataset_path = LOAD_PATH + f"hmm_synthetic_dataset(noise-{noisy_level}_state-{num_states}_obs-{num_observations}_size-{size}).npz"
    loaded_npz = np.load(dataset_path, allow_pickle=True)
    observations = list(loaded_npz['observation'])
    real_hidden_states = list(loaded_npz['real_hidden'])
    noisy_hidden_states = list(loaded_npz['noisy_hidden'])
    real_trans_dist = np.vstack(loaded_npz['real_trans'])

    real_trans_count = np.zeros((num_states, num_states), dtype='int')
    noisy_trans_count = np.zeros((num_states, num_states), dtype='int')
    real_emis_count = np.zeros((num_observations, num_states), dtype='int')
    noisy_emis_count = np.zeros((num_observations, num_states), dtype='int')

    for i in range(size):
        for t in range(len(observations[i])):
            real_emis_count[observations[i][t], real_hidden_states[i][t]] += 1
            noisy_emis_count[observations[i][t], noisy_hidden_states[i][t]] += 1
            if t > 0:
                noisy_trans_count[noisy_hidden_states[i][t - 1], noisy_hidden_states[i][t]] += 1
                real_trans_count[real_hidden_states[i][t - 1], real_hidden_states[i][t]] += 1

    mylogger.info(f"real trans count: \n {np.array2string(real_trans_count)}")
    # print("real trans count: \n", real_trans_count)
    mylogger.info(f"noisy trans count: \n {np.array2string(noisy_trans_count)}")
    # print("noisy trans count: \n", noisy_trans_count)
    total_count = np.sum(noisy_trans_count)
    mylogger.info(f"total transition counts: {str(total_count)}")
    # print(total_count)
    mis, tot = difference(real_hidden_states, noisy_hidden_states)
    mylogger.info(f"The initial rate of missing states is:  {str(round(mis / tot * 100, 3))}%")
    mylogger.info(f"Initial Euclidean Distance: {euclidean_distance(real_trans_count, noisy_trans_count)}")
    # print(euclidean_distance(real_trans_count, noisy_trans_count))

    return Dataset(real_hidden_states, noisy_hidden_states, real_trans_count, noisy_trans_count, real_trans_dist, observations, real_emis_count, noisy_emis_count)
