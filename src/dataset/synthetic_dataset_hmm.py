import numpy as np
import argparse
import tqdm

from ..utils.const import LOAD_PATH
from ..logger import mylogger

def create_hmm_dataset(noisy_level, num_states, num_obs, size):
    def simulate_hmm(num_sequences, min_length, max_length, start_prob, trans_prob, emis_prob):
        sequences = []
        hidden_states = []
        for _ in tqdm.tqdm(range(num_sequences), desc="Creating HMM Dataset"):
            sequence_length = np.random.randint(min_length, max_length + 1)
            current_state = np.random.choice(num_states, p=start_prob)
            observation_sequence = []
            state_sequence = []
            for _ in range(sequence_length):
                state_sequence.append(current_state)
                observation = np.random.choice(num_obs, p=emis_prob[current_state])
                observation_sequence.append(observation)
                current_state = np.random.choice(num_states, p=trans_prob[current_state])
            sequences.append(observation_sequence)
            hidden_states.append(state_sequence)

        return sequences, hidden_states

    def add_noise_to_states(hidden_states, number_states, flip_prob=0.5):
        noisy_hidden_states = []
        for sequence in hidden_states:
            noisy_sequence = []
            for state in sequence:
                if np.random.rand() < flip_prob:
                    # Flip the state to a different random state
                    possible_states = list(range(number_states))
                    possible_states.remove(state)  # Remove the current state from possibilities
                    new_state = np.random.choice(possible_states)
                    noisy_sequence.append(new_state)
                else:
                    noisy_sequence.append(state)
            noisy_hidden_states.append(noisy_sequence)
        return noisy_hidden_states

    transition_probs = np.random.dirichlet(np.ones(num_states), size=num_states)
    emission_probs = np.random.dirichlet(np.ones(num_obs), size=num_states)
    initial_state_dist = np.ones(num_states) / num_states

    syn_sequences, syn_hidden_states = simulate_hmm(
        num_sequences=size,
        min_length=10,
        max_length=30,
        start_prob=initial_state_dist,
        trans_prob=transition_probs,
        emis_prob=emission_probs
    )

    noisy_hidden_states = add_noise_to_states(syn_hidden_states, num_states, flip_prob=noisy_level)

    file_path = LOAD_PATH + f"hmm_synthetic_dataset(noise-{noisy_level}_state-{num_states}_obs-{num_obs}_size-{size}).npz"
    seq_object = np.array(syn_sequences, dtype=object)
    hid_object = np.array(syn_hidden_states, dtype=object)
    noisy_hid_object = np.array(noisy_hidden_states, dtype=object)
    trans_object = transition_probs
    emis_object = emission_probs
    np.savez(file_path, observation=seq_object, real_hidden=hid_object, noisy_hidden=noisy_hid_object,
             real_trans=trans_object, emis=emis_object, noisy_level=noisy_level)
    mylogger.info(f"Dataset with noise-{noisy_level}_state-{num_states}_obs-{num_obs}_size-{size} created...")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-noise', type=float, default=0.5)
    parser.add_argument('-states', type=int, default=10)
    parser.add_argument('-obs', type=int, default=500)
    parser.add_argument('-size', type=int, default=50000)

    args = parser.parse_args()
    create_hmm_dataset(args.noise, args.states, args.obs, args.size)

if __name__ == "__main__":
    main()