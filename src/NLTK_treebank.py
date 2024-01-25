import string

class Lang:
    def __init__(self, name, dataset):
        self.name = name
        self.token2index = {"SOS": 0, "UNK": 1}
        self.index2token = {0: "SOS", 1: "UNK"}
        self.nth_words = 2                      # Count UNK and SOS
        self.token2count = {"SOS": 0, "UNK": 0}
        self.dataset = dataset                  # 3914 sentences in total for NLTK subset
        self.punctuations = set(string.punctuation)

    def preprocess(self):
        tokenized_sentences, pos_tags = [], []
        for sentence in self.dataset:
            filtered_sentence = [word_tag for word_tag in sentence if
                                 word_tag[0] not in self.punctuations and word_tag[1] != '-NONE-']
            if filtered_sentence:
                words, tags = zip(*filtered_sentence)
                tokenized_sentences.append(['SOS'] + list(words))
                pos_tags.append(['SOS_tag'] + list(tags))
        return tokenized_sentences, pos_tags

    def add_sentences(self, tokenized_sentences):
        for sentence in tokenized_sentences:
            for token in sentence:
                self.add_token(token)

    def add_token(self, token):
        if token not in self.token2count:
            self.token2count[token] = 1
        else:
            self.token2count[token] += 1

    def filter_freq(self, filter_level=2):
        filtered_dict = {"UNK": 0}
        for token, count in self.token2count.items():
            if count >= filter_level:
                filtered_dict[token] = count
            else:
                filtered_dict["UNK"] += count

        self.token2count = filtered_dict            # 5751 tokens after filter=2

    def create_indices(self):
        # self.token2count = dict(sorted(self.token2count.items(), key=lambda item: item[1]))
        for token in self.token2count.keys():
            if token not in self.token2index:
                self.token2index[token] = self.nth_words
                self.index2token[self.nth_words] = token
                self.nth_words += 1

    def convert_indices(self, tokenized_sentences):
        return [
            [self.token2index[token] if token in self.token2index else self.token2index["UNK"] for token in sentence]
            for sentence in tokenized_sentences]

    def build_dataset(self):
        tokenized_sentences, pos_tags = self.preprocess()
        self.add_sentences(tokenized_sentences)
        # print("count: ", list[self.token2count.items()][:10])
        self.filter_freq(filter_level=2)
        self.create_indices()
        token_indices = self.convert_indices(tokenized_sentences)
        # filter sentences that contains only one word
        filter_mask = [len(indices) > 1 for indices in token_indices]
        filtered_token_indices = [t for t, m in zip(token_indices, filter_mask) if m]
        filtered_pos_tags = [p for p, m in zip(pos_tags, filter_mask) if m]
        return filtered_token_indices, filtered_pos_tags

