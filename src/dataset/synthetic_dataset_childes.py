import numpy as np
import nltk
import re
from nltk.tag.perceptron import PerceptronTagger
from collections import Counter
from collections import defaultdict

from ..utils.const import LOAD_PATH, CONTRACTION_MAPPING
from ..logger import mylogger

tagger = PerceptronTagger()
tagset = 'universal'


def create_lang_dataset_childes(noise_level):
    def preprocess(text):
        text = text.lower()
        text = re.sub(r"(\w) '(\w)", r"\1'\2", text)  # remove the space before the apostrophe
        text = re.sub(r" n't", "n't", text)
        # print(text)
        text = re.sub(r'\(.*\)', '', text)  # remove word in parentheses

        words = text.split(' ')
        result = []
        for i in range(len(words)):
            if words[i] in CONTRACTION_MAPPING:
                result += CONTRACTION_MAPPING[words[i]].split(' ')
                continue
            result.append(words[i])

        text = ' '.join(result)
        text = text.replace("'s", '')
        text = re.sub(r'[^a-zA-Z0-9 ]', '', text)  # remove punctuations
        return text

    def filter_length(dataset, filter_len=7):
        return [sent for sent in dataset if len(sent) >= filter_len]

    def replace_low_frequency_words(sentences_with_pos_tags, filter_count=1):
        # Count the frequencies of each word
        word_counts = Counter(word for sentence in sentences_with_pos_tags for word, _ in sentence)

        # Replace words with count less than filter_count to 'UNK' and their tags to 'UNK_TAG'
        processed_sentences = []
        for sentence in sentences_with_pos_tags:
            new_sentence = []
            for word, pos in sentence:
                if pos in ['X', '.']:
                    continue
                if word_counts[word] < filter_count:
                    new_word = 'UNK'  # Only set the word to UNK
                    new_pos = pos
                else:
                    new_word = word
                    new_pos = pos
                new_sentence.append((new_word, new_pos))
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

        all_words = [word for sentence in sentences_with_pos_tags for word, _ in sentence]
        all_pos_tags = [pos for sentence in sentences_with_pos_tags for _, pos in sentence]

        return build_index(all_words, start_index=0), build_index(all_pos_tags, start_index=1)

    def convert_to_indexes(filtered_sentences_tags, word_to_index, pos_to_index):
        hidden_states_universal = []
        observations = []

        for sentence in filtered_sentences_tags:
            # if len(sentence) <= 5:
            #     continue
            sentence_pos_indexes = [pos_to_index[pos] for _, pos in sentence]
            sentence_word_indexes = [word_to_index[word] for word, _ in sentence]

            hidden_states_universal.append(sentence_pos_indexes)
            observations.append(sentence_word_indexes)

        return hidden_states_universal, observations

    def add_noise_to_states_childes(hidden_states, number_states, flip_prob=0.3):
        noisy_hidden_states = []
        for sequence in hidden_states:
            noisy_sequence = []
            for state in sequence:
                if np.random.rand() < flip_prob:
                    possible_states = list(range(1, number_states + 1))  # Flip the state to a different random state
                    possible_states.remove(state)  # Remove the current state from possibilities
                    new_state = np.random.choice(possible_states)
                    noisy_sequence.append(new_state)
                else:
                    noisy_sequence.append(state)
            noisy_hidden_states.append(noisy_sequence)
        return noisy_hidden_states

    print("Creating the Childes dataset ...")

    sentences = []
    with open(LOAD_PATH + "all_sents_shuffled.raw", 'r') as file:
        for line in file:
            line = line.strip()
            sentences.append(preprocess(line))

    tokenized_dataset = []
    for sent in sentences:
        tokens = nltk.word_tokenize(sent)
        tags = nltk.tag._pos_tag(tokens, tagset, tagger, lang='eng')
        tokenized_dataset.append(tags)

    filtered_dataset = filter_length(tokenized_dataset)
    filtered_sentences = replace_low_frequency_words(filtered_dataset, filter_count=10)

    word_to_index, pos_to_index = create_vocab_index(filtered_sentences)

    hidden_states_universal, observations = convert_to_indexes(
        filtered_sentences, word_to_index, pos_to_index)

    print("Adding noise to the Childes dataset ...")

    noisy_hidden_states_universal = add_noise_to_states_childes(hidden_states_universal, len(pos_to_index),
                                                                flip_prob=noise_level)

    for i in range(len(hidden_states_universal)):
        hidden_states_universal[i].insert(0, 0)
        noisy_hidden_states_universal[i].insert(0, 0)

        observations[i].insert(0, -1)

    file_path = LOAD_PATH + f"Childes_synthetic_dataset(noise-{noise_level}).npz"
    obs_object = np.array(observations, dtype=object)
    uni_hid_object = np.array(hidden_states_universal, dtype=object)
    noisy_uni_hid_object = np.array(noisy_hidden_states_universal, dtype=object)
    np.savez(file_path, num_states=len(pos_to_index) + 1, num_obs=len(word_to_index), observation=obs_object,
             real_hidden_universal=uni_hid_object, noisy_hidden_universal=noisy_uni_hid_object, noisy_level=noise_level)

    mylogger.info(f"Childes dataset with noise {noise_level} is created ...")
