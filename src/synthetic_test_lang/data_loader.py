import numpy as np

from ..utils.utils import euclidean_distance, difference
from ..utils.const import LOAD_PATH

from ..logger import mylogger

class SynLangDataset:
    def __init__(self, real_hidden_states, noisy_hidden_states, real_trans_count, noisy_trans_count, observations, noisy_emis_count, num_states, num_obs, size):
        self.real_hidden_states = real_hidden_states
        self.noisy_hidden_states = noisy_hidden_states
        self.real_trans_count = real_trans_count
        self.noisy_trans_count = noisy_trans_count
        self.observations = observations
        self.noisy_emis_count = noisy_emis_count
        self.num_states = num_states
        self.num_obs = num_obs
        self.size = size


def load_data(dataset_name, noise_level):
    dataset_path = LOAD_PATH + dataset_name + f"_synthetic_dataset(noise-{noise_level}).npz"
    loaded_npz = np.load(dataset_path, allow_pickle=True)
    num_states = int(loaded_npz['num_states'])
    num_observations = int(loaded_npz['num_obs'])
    observations = list(loaded_npz['observation'])
    real_hidden_states = list(loaded_npz['real_hidden_universal'])
    noisy_hidden_states = list(loaded_npz['noisy_hidden_universal'])

    size = len(observations)

    real_trans_count = np.zeros((num_states, num_states), dtype='int')
    noisy_trans_count = np.zeros((num_states, num_states), dtype='int')
    noisy_emis_count = np.zeros((num_observations, num_states), dtype='int')

    for i in range(size):
        for t in range(1, len(observations[i])):  # starts from 1 to bypass -1 at the beginning
            noisy_emis_count[observations[i][t], noisy_hidden_states[i][t]] += 1

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

    return SynLangDataset(real_hidden_states, noisy_hidden_states, real_trans_count, noisy_trans_count, observations, noisy_emis_count, num_states, num_observations, size)
