import argparse

from .data_loader import load_data, TrainHMMDataset
from .trainer import train_sampler

from ..model import HDPHMM, DirectAssignmentPOS
from ..logger import mylogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-iter', type=int, default=20)
    parser.add_argument('-states', type=int, default=10)
    parser.add_argument('-obs', type=int, default=500)
    parser.add_argument('-len', type=int, default=5000)

    args = parser.parse_args()

    mylogger.info("Start Training...")
    mylogger.info(
        f"Number of States: {args.states}, Number of Observations: {args.obs}, Dataset Length: {args.len}")

    model = HDPHMM()
    dataset: TrainHMMDataset = load_data(args.states, args.obs, args.len)
    # TODO: modify the sampler to let it fit with only one long sequence
    sampler = DirectAssignmentPOS(model, dataset.train_obs, args.obs)

    train_sampler(sampler, args, dataset)