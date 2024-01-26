import re
import numpy as np
import os

from .data_loader import load_data, LanguageDataset
from .trainer import train_sampler

from ..model import HDPHMM, DirectAssignmentPOS
from ..dataset import create_lang_dataset_ptb, create_lang_dataset_childes
from ..utils.const import SAVE_PATH, LOAD_PATH
from ..utils.utils import set_print_options
from ..logger import mylogger

set_print_options()


def train_mode(args):
    model = HDPHMM()
    dataset: LanguageDataset = load_data(args.name, args.noise)
    sampler = DirectAssignmentPOS(
        model, dataset.observations, dataset.size,
        hidden_states=dataset.noisy_hidden_states,
        transition_count=dataset.noisy_trans_count,
        emission_count=dataset.noisy_emis_count,
        K=dataset.num_states
    )
    # update beta_vec for train mode
    for i in range(dataset.num_states - 1):
        sampler.model.update_beta_with_new_state()

    train_sampler(sampler, args, dataset)


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
    sampler = DirectAssignmentPOS(model, observations, args.obs, hidden_states=hidden_states, emission_count=emis_count,
                                  transition_count=trans_count, K=K)

    dataset: LanguageDataset = load_data(args.name, args.noise)

    train_sampler(sampler, args, dataset, prev_iters=prev_iters)


def run(args):
    if not os.path.exists(LOAD_PATH + args.name + f"_synthetic_dataset(noise-{args.noise}).npz"):
        if args.name == 'PTB':
            create_lang_dataset_ptb(args.noise)
        if args.name == 'Childes':
            create_lang_dataset_childes(args.noise)

    mylogger.info("Start Training...")
    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'resume':
        resume_mode(args)
