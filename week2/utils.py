from collections import defaultdict
import numpy as np
from evaluation import precision_recall_f1


def read_data(file_path):
    tokens = []
    tags = []

    tweet_tokens = []
    tweet_tags = []
    for line in open(file_path, encoding='utf-8'):
        line = line.strip()
        if not line:
            if tweet_tokens:
                tokens.append(tweet_tokens)
                tags.append(tweet_tags)
            tweet_tokens = []
            tweet_tags = []
        else:
            token, tag = line.split()
            # Replace all urls with <URL> token
            # Replace all users with <USR> token

            ######################################
            ######### YOUR CODE HERE #############
            if token.startswith("http://") or token.startswith("https://"):
                token = "<URL>"
            if token.startswith("@"):
                token = "<USR>"
            ######################################

            tweet_tokens.append(token)
            tweet_tags.append(tag)

    return tokens, tags


def build_dict(x_or_y, x_or_y_special):
    """
        :x_or_y: a list of lists of Xs or Ys, like kist of docs with tokens within them
        :x_or_y_special: list of some special Xs or Ys
        :returns: tok2idx = {}\n
                  idx2tok = []

    """
    # Create a dictionary with default value 0
    tok2idx = defaultdict(lambda: 0)
    idx2tok = []

    # Create mappings from tokens (or tags) to indices and vice versa.
    # At first, add special tokens (or tags) to the dictionaries.
    # The first special token must have index 0.

    # Mapping tok2idx should contain each token or tag only once.
    # To do so, you should:
    # 1. extract unique tokens/tags from the tokens_or_tags variable, which is not
    #    occure in special_tokens (because they could have non-empty intersection)
    # 2. index them (for example, you can add them into the list idx2tok
    # 3. for each token/tag save the index into tok2idx).

    ######################################
    ######### YOUR CODE HERE #############
    unique_tokens = set([token for tokens in x_or_y for token in tokens]) - set(x_or_y_special)
    idx2tok.extend(x_or_y_special)
    idx2tok.extend(unique_tokens)
    for i in range(len(idx2tok)):
        tok2idx[idx2tok[i]] = i
    ######################################

    return tok2idx, idx2tok


class Helper:
    def __init__(self, token2idx, idx2token, tag2idx, idx2tag):
        self.token2idx = token2idx
        self.idx2token = idx2token
        self.tag2idx = tag2idx
        self.idx2tag = idx2tag

    def tokens2idxs(self, tokens):
        return [self.token2idx[token] for token in tokens]

    def tags2idxs(self, tags):
        return [self.tag2idx[tag] for tag in tags]

    def indexes2tokens(self, indexes):
        return [self.idx2token[index] for index in indexes]

    def indexes2tags(self, indexes):
        return [self.idx2tag[index] for index in indexes]

    def batches_generator(self, batch_size, tokens, tags, shuffle=True, allow_smaller_last_batch=True):
        """Generates padded batches of tokens and tags."""

        n_samples = len(tokens)
        if shuffle:
            order = np.random.permutation(n_samples)
        else:
            order = np.arange(n_samples)

        n_batches = n_samples // batch_size
        if allow_smaller_last_batch and n_samples % batch_size:
            n_batches += 1

        for k in range(n_batches):
            batch_start = k * batch_size
            batch_end = min((k + 1) * batch_size, n_samples)
            current_batch_size = batch_end - batch_start
            x_list = []
            y_list = []
            max_len_token = 0
            for idx in order[batch_start: batch_end]:
                x_list.append(self.tokens2idxs(tokens[idx]))
                y_list.append(self.tags2idxs(tags[idx]))
                max_len_token = max(max_len_token, len(tags[idx]))

            # Fill in the data into numpy nd-arrays filled with padding indices.
            x = np.ones([current_batch_size, max_len_token], dtype=np.int32) * self.token2idx['<PAD>']
            y = np.ones([current_batch_size, max_len_token], dtype=np.int32) * self.tag2idx['O']
            lengths = np.zeros(current_batch_size, dtype=np.int32)
            for n in range(current_batch_size):
                utt_len = len(x_list[n])
                x[n, :utt_len] = x_list[n]
                lengths[n] = utt_len
                y[n, :utt_len] = y_list[n]
            yield x, y, lengths

    def predict_tags(self, model, session, token_idxs_batch, lengths):
        """Performs predictions and transforms indices to tokens and tags."""

        tag_idxs_batch = model.predict_for_batch(session, token_idxs_batch, lengths)

        tags_batch, tokens_batch = [], []
        for tag_idxs, token_idxs in zip(tag_idxs_batch, token_idxs_batch):
            tags, tokens = [], []
            for tag_idx, token_idx in zip(tag_idxs, token_idxs):
                tags.append(self.idx2tag[tag_idx])
                tokens.append(self.idx2token[token_idx])
            tags_batch.append(tags)
            tokens_batch.append(tokens)
        return tags_batch, tokens_batch

    def eval_conll(self, model, session, tokens, tags, short_report=True):
        """Computes NER quality measures using CONLL shared task script."""

        y_true, y_pred = [], []
        for x_batch, y_batch, lengths in self.batches_generator(1, tokens, tags):
            tags_batch, tokens_batch = self.predict_tags(model, session, x_batch, lengths)
            if len(x_batch[0]) != len(tags_batch[0]):
                raise Exception("Incorrect length of prediction for the input, "
                                "expected length: %i, got: %i" % (len(x_batch[0]), len(tags_batch[0])))
            predicted_tags = []
            ground_truth_tags = []
            for gt_tag_idx, pred_tag, token in zip(y_batch[0], tags_batch[0], tokens_batch[0]):
                if token != '<PAD>':
                    ground_truth_tags.append(self.idx2tag[gt_tag_idx])
                    predicted_tags.append(pred_tag)

            # We extend every prediction and ground truth sequence with 'O' tag
            # to indicate a possible end of entity.
            y_true.extend(ground_truth_tags + ['O'])
            y_pred.extend(predicted_tags + ['O'])

        results = precision_recall_f1(y_true, y_pred, print_results=True, short_report=short_report)
        return results
