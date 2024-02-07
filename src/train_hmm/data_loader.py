import numpy as np

from ..utils.const import LOAD_PATH

class TrainHMMDataset:
    def __init__(self, train_obs, train_hid, train_len, test_obs, test_hid, test_len):
        self.train_obs = train_obs
        self.train_hid = train_hid
        self.train_len = train_len
        self.test_obs = test_obs
        self.test_hid = test_hid
        self.test_len = test_len

def load_data(num_states, num_observations, train_length):
    dataset_path = LOAD_PATH + f"../../../data/hmm_train_dataset(state-{num_states}_obs-{num_observations}_length-{train_length}).npz"
    loaded_npz = np.load(dataset_path, allow_pickle=True)

    train_obs = list(loaded_npz['train_observations'])
    train_hid = list(loaded_npz['train_hidden_states'])
    train_len = list(loaded_npz['train_length'])

    test_obs = list(loaded_npz['test_observations'])
    test_hid = list(loaded_npz['test_hidden_states'])
    test_len = list(loaded_npz['test_length'])

    return TrainHMMDataset(train_obs, train_hid, train_len, test_obs, test_hid, test_len)


