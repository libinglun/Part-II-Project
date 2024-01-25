import re
import numpy as np

from .data_loader import load_data, Dataset
from .trainer import train_sampler
from model import HDPHMM, DirectAssignmentPOS
from const import NOISE_LEVEL, NUM_STATES, NUM_OBS, SIZE, SAVE_PATH


def train_mode(args):
    model = HDPHMM()
    dataset: Dataset = load_data(NOISE_LEVEL, NUM_STATES, NUM_OBS, SIZE)
    sampler = DirectAssignmentPOS(
        model, dataset.observations, NUM_OBS,
        hidden_states=dataset.noisy_hidden_states,
        transition_count=dataset.noisy_trans_count,
        emission_count=dataset.emis_count,
        K=NUM_STATES
    )
    # update beta_vec for train mode
    for i in range(NUM_STATES - 1):
        sampler.model.update_beta_with_new_state()

    train_sampler(sampler, args.iter, dataset)

def resume_mode(args):
    if args.state is None:
        raise ValueError("Please specify the path of stored params!")
    match = re.search(r'iter-(\d+)', args.state)
    prev_iters = int(match.group(1))
    load_path = SAVE_PATH + args.state + ".npz"
    loaded_model = np.load(load_path, allow_pickle=True)
    observations = list(loaded_model['observation'])
    hidden_states = list(loaded_model['hidden_state'])
    K = int(loaded_model['K'])
    trans_count = np.array(loaded_model['trans_count'])
    emis_count = np.vstack(loaded_model['emis_count'])
    alpha = float(loaded_model['alpha'])
    gamma = float(loaded_model['gamma'])
    beta = np.array(loaded_model['beta'])

    model = HDPHMM(alpha, gamma, beta)
    sampler = DirectAssignmentPOS(model, observations, NUM_OBS, hidden_states=hidden_states, emission_count=emis_count, transition_count=trans_count, K=K)

    dataset: Dataset = load_data(NOISE_LEVEL, NUM_STATES, NUM_OBS, SIZE)

    train_sampler(sampler, args.iter, dataset, prev_iters=prev_iters)


def run(args):
    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'resume':
        resume_mode(args)
