import numpy as np
import tqdm
from collections import Counter
from collections import defaultdict

from ..utils.const import LOAD_PATH
from ..logger import mylogger


def create_lang_dataset_ptb(noise_level):
    print("Creating the PTB dataset ...")

    def read_sentences_with_pos_tags(path):
        sentences_with_pos_tags = []

        with open(path, 'r', encoding='utf-8') as file:
            current_sentence = []
            for line in file:
                if line.strip() and not line.startswith('#'):  # Skip empty lines and comments
                    fields = line.split('\t')
                    if len(fields) > 3:  # Ensure there are enough fields
                        word = fields[1].lower()  # Word form is the second field
                        upos = fields[3]  # Universal POS tag is the fourth field
                        xpos = fields[4]  # Language specific POS tag is the fifth field
                        if upos not in ['NUM', 'PUNCT', 'INTJ',
                                        'X']:  # Remove punctuation, interjection, and undefined pos tags
                            current_sentence.append((word, upos, xpos))

                # New sentence
                elif current_sentence:
                    sentences_with_pos_tags.append(current_sentence)
                    current_sentence = []

        return sentences_with_pos_tags

    def replace_low_frequency_words(sentences_with_pos_tags, filter_count=1):
        # Count the frequencies of each word
        word_counts = Counter(word for sentence in sentences_with_pos_tags for word, _, _ in sentence)

        # Replace words with count less than filter_count to 'UNK' and their tags to 'UNK_TAG'
        processed_sentences = []
        for sentence in sentences_with_pos_tags:
            new_sentence = []
            for word, upos, xpos in sentence:
                if word_counts[word] < filter_count:
                    new_word = 'UNK'  # Only set the word to UNK
                    new_upos = upos
                    new_xpos = xpos
                else:
                    new_word = word
                    new_upos = upos
                    new_xpos = xpos
                new_sentence.append((new_word, new_upos, new_xpos))
            processed_sentences.append(new_sentence)

        return processed_sentences

    def create_vocab_index(sentences_with_pos_tags):
        # Function to create a dictionary mapping each unique word/POS to an integer index
        # with specified start index
        def build_index(items, start_index=0):
            item_to_index = defaultdict(lambda: len(item_to_index) + start_index)
            for item in items:
                item_to_index[item]
            return dict(item_to_index)

        all_words = [word for sentence in sentences_with_pos_tags for word, upos, xpos in sentence]
        all_upos_tags = [upos for sentence in sentences_with_pos_tags for word, upos, xpos in sentence]
        all_xpos_tags = [xpos for sentence in sentences_with_pos_tags for word, upos, xpos in sentence]

        return build_index(all_words, start_index=0), build_index(all_upos_tags, start_index=1), build_index(
            all_xpos_tags, start_index=1)

    def convert_to_indexes(filtered_sentences_tags, word_to_index, upos_to_index, xpos_to_index):
        hidden_states_universal = []
        hidden_states_specific = []
        observations = []

        for sentence in filtered_sentences_tags:
            if len(sentence) <= 5:
                continue
            sentence_upos_indexes = [upos_to_index[upos] for _, upos, _ in sentence]
            sentence_xpos_indexes = [xpos_to_index[xpos] for _, _, xpos in sentence]
            sentence_word_indexes = [word_to_index[word] for word, _, _ in sentence]

            hidden_states_universal.append(sentence_upos_indexes)
            hidden_states_specific.append(sentence_xpos_indexes)
            observations.append(sentence_word_indexes)

        return hidden_states_universal, hidden_states_specific, observations

    def add_noise_to_states_ptb(hidden_states, number_states, flip_prob=0.5, remove=False):
        noisy_hidden_states = []
        for sequence in hidden_states:
            noisy_sequence = []
            for state in sequence:
                if np.random.rand() < flip_prob:
                    possible_states = list(range(1, number_states + 1))
                    if remove:
                        possible_states.remove(state)
                    new_state = np.random.choice(possible_states)
                    noisy_sequence.append(new_state)
                else:
                    noisy_sequence.append(state)
            noisy_hidden_states.append(noisy_sequence)
        return noisy_hidden_states

    file_path1 = LOAD_PATH + 'ptb/penn-train.conllu'
    file_path2 = LOAD_PATH + 'ptb/penn-test.conllu'
    file_path3 = LOAD_PATH + 'ptb/penn-dev.conllu'

    sentences_pos_tags = read_sentences_with_pos_tags(file_path1)
    filtered_sentences = replace_low_frequency_words(sentences_pos_tags, filter_count=10)

    word_to_index, upos_to_index, xpos_to_index = create_vocab_index(filtered_sentences)
    hidden_states_universal, hidden_states_specific, observations = convert_to_indexes(
        filtered_sentences, word_to_index, upos_to_index, xpos_to_index)

    print("Adding noise to the PTB dataset ...")

    noisy_hidden_states_universal = add_noise_to_states_ptb(hidden_states_universal, len(upos_to_index),
                                                            flip_prob=noise_level)
    noisy_hidden_states_specific = add_noise_to_states_ptb(hidden_states_specific, len(xpos_to_index),
                                                           flip_prob=noise_level)

    for i in range(len(hidden_states_universal)):
        hidden_states_universal[i].insert(0, 0)
        noisy_hidden_states_universal[i].insert(0, 0)

        hidden_states_specific[i].insert(0, 0)
        noisy_hidden_states_specific[i].insert(0, 0)

        observations[i].insert(0, -1)

    file_path = LOAD_PATH + f"PTB_synthetic_dataset(noise-{noise_level}).npz"
    obs_object = np.array(observations, dtype=object)
    uni_hid_object = np.array(hidden_states_universal, dtype=object)
    noisy_uni_hid_object = np.array(noisy_hidden_states_universal, dtype=object)
    spc_hid_object = np.array(hidden_states_specific, dtype=object)
    noisy_spc_hid_object = np.array(noisy_hidden_states_specific, dtype=object)
    np.savez(file_path, num_states=len(upos_to_index) + 1, num_obs=len(word_to_index), observation=obs_object,
             real_hidden_universal=uni_hid_object, noisy_hidden_universal=noisy_uni_hid_object,
             real_hidden_specific=spc_hid_object, noisy_hidden_specifc=noisy_spc_hid_object,
             noisy_level=noise_level)

    mylogger.info(f"PTB dataset with noise {noise_level} is created ...")
