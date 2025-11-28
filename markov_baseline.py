# simple markov baseline for task recognition
import numpy as np
from collections import Counter

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

from data_emaki import load_emaki, build_windows

# special tokens for padding and unknown
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


# build integer vocab from token sequences
def build_vocab(sequences):
    # count all tokens across all sequences
    token_counts = Counter(tok for seq in sequences for tok in seq)

    # start vocab with pad and unk
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}

    # add every other token
    for tok in token_counts.keys():
        if tok not in vocab:
            vocab[tok] = len(vocab)

    return vocab


# encode a window of tokens into integer ids
def encode_seq(seq, vocab):
    # use unk id for unseen tokens
    unk_id = vocab[UNK_TOKEN]
    return [vocab.get(tok, unk_id) for tok in seq]


# compute log-likelihood of a sequence under a class-specific markov model
def markov_log_likelihood(seq, log_transitions):
    ll = 0.0
    # sum log probabilities for each transition
    for i in range(len(seq) - 1):
        s = seq[i]
        t = seq[i + 1]
        ll += log_transitions[s, t]
    return ll


def main():
    # load emaki dataframe
    df = load_emaki("EMAKI_utt.pkl")

    # build token windows and labels
    # sequences: list of [token strings], labels: np array of task IDs, window_users not used now
    sequences, labels, window_users = build_windows(df)

    print("number of windows:", len(sequences))
    print("unique tasks:", set(labels))

    # build vocab from all windows
    vocab = build_vocab(sequences)
    vocab_size = len(vocab)
    print("vocab size:", vocab_size)

    # encode every window as integer ids
    encoded_sequences = [encode_seq(seq, vocab) for seq in sequences]
    encoded_sequences = np.array(encoded_sequences)
    labels_np = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        encoded_sequences,      # all windows as integer ids
        labels_np,              # task labels for each window
        test_size=0.2,          # 20% of windows for test
        random_state=42,        # fixed seed for reproducibility
        stratify=labels_np      # keep class proportions similar in train/test
    )

    print("train shape:", X_train.shape, "test shape:", X_test.shape)

    # quick sanity check so we can see class counts
    print("train label counts:", Counter(y_train))
    print("test label counts:", Counter(y_test))

    # build transition matrices for each class (task)
    num_states = vocab_size
    class_transition_counts = {}

    # initialize counts with 1 for Laplace smoothing
    for c in np.unique(y_train):
        class_transition_counts[c] = np.ones((num_states, num_states), dtype=np.float64)

    # update transition counts from training windows
    for seq, label in zip(X_train, y_train):
        for i in range(len(seq) - 1):
            s = seq[i]
            t = seq[i + 1]
            class_transition_counts[label][s, t] += 1.0

    # convert counts to log probabilities
    class_log_transitions = {}
    for c, counts in class_transition_counts.items():
        # normalize each row
        row_sums = counts.sum(axis=1, keepdims=True)
        probs = counts / row_sums

        # avoid log(0)
        probs = np.clip(probs, 1e-12, 1.0)
        class_log_transitions[c] = np.log(probs)

    # predict labels for test windows
    y_pred = []
    for seq in X_test:
        # compute log-likelihood for each class
        scores = {}
        for c in class_log_transitions.keys():
            scores[c] = markov_log_likelihood(seq, class_log_transitions[c])

        # pick class with highest log-likelihood
        best_class = max(scores, key=scores.get)
        y_pred.append(best_class)

    y_pred = np.array(y_pred)

    # evaluate baseline
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")

    print(f"markov baseline accuracy: {acc:.4f}")
    print(f"markov baseline macro F1: {f1_macro:.4f}")
    print("confusion matrix:\n", confusion_matrix(y_test, y_pred))

    class_list = sorted(class_log_transitions.keys())
    y_prob = np.zeros((len(X_test), len(class_list)))
    for i, seq in enumerate(X_test):
        scores = []
        for c in class_list:
            scores.append(markov_log_likelihood(seq, class_log_transitions[c]))
        scores = np.array(scores)
        exp_scores = np.exp(scores - scores.max())
        probs = exp_scores / exp_scores.sum()
        y_prob[i, :] = probs

    import os
    os.makedirs("model_outputs", exist_ok=True)
    np.save("model_outputs/markov_y_true.npy", y_test)
    np.save("model_outputs/markov_y_pred.npy", y_pred)
    np.save("model_outputs/markov_y_prob.npy", y_prob)
    print("Saved Markov outputs to model_outputs/")


if __name__ == "__main__":
    main()
