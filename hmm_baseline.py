import numpy as np
from collections import Counter

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

from hmmlearn import hmm  # HMM library for discrete observations

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


def main():
    # load emaki dataframe
    df = load_emaki("EMAKI_utt.pkl")

    # build token windows and labels (same as markov)
    sequences, labels, window_users = build_windows(df)

    print("number of windows:", len(sequences))
    print("unique tasks:", set(labels))

    # build vocab from all windows
    vocab = build_vocab(sequences)
    vocab_size = len(vocab)
    print("vocab size:", vocab_size)

    # encode every window as integer ids
    encoded_sequences = [encode_seq(seq, vocab) for seq in sequences]
    encoded_sequences = np.array(encoded_sequences, dtype=object)
    labels_np = np.array(labels)

    # stratified 80/20 split over windows (same as markov)
    X_train, X_test, y_train, y_test = train_test_split(
        encoded_sequences,
        labels_np,
        test_size=0.2,
        random_state=42,
        stratify=labels_np,
    )

    print("train shape:", X_train.shape, "test shape:", X_test.shape)
    print("train label counts:", Counter(y_train))
    print("test label counts:", Counter(y_test))

    # group training sequences by class label
    train_by_class = {}
    for seq, label in zip(X_train, y_train):
        train_by_class.setdefault(label, []).append(np.array(seq, dtype=int))

    # number of hidden states for the HMM (you can tune this)
    NUM_STATES = 5

    # train one HMM per task label
    hmm_models = {}
    for c, seqs in train_by_class.items():
        # concatenate all sequences for this class
        lengths = [len(s) for s in seqs]
        concat = np.concatenate(seqs)

        # hmmlearn expects shape (n_samples, 1)
        X_concat = concat.reshape(-1, 1)

        print(f"training HMM for class {c} with {len(seqs)} windows")

        model = hmm.MultinomialHMM(
            n_components=NUM_STATES,
            n_iter=20,           # max EM iterations
            random_state=0,
            verbose=False,
        )

        # fit model to all sequences of this class
        model.fit(X_concat, lengths)
        hmm_models[c] = model

    # predict labels for test windows by sequence log-likelihood
    y_pred = []
    for seq in X_test:
        seq_arr = np.array(seq, dtype=int).reshape(-1, 1)

        # compute log-likelihood under each class HMM
        scores = {}
        for c, model in hmm_models.items():
            scores[c] = model.score(seq_arr)

        # pick class with highest log-likelihood
        best_class = max(scores, key=scores.get)
        y_pred.append(best_class)

    y_pred = np.array(y_pred)

    # evaluate baseline
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")

    print(f"HMM baseline accuracy: {acc:.4f}")
    print(f"HMM baseline macro F1: {f1_macro:.4f}")
    print("confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("confusion matrix:\n", confusion_matrix(y_test, y_pred))

    class_list = sorted(hmm_models.keys())
    y_prob = np.zeros((len(X_test), len(class_list)))
    for i, seq in enumerate(X_test):
        seq_arr = np.array(seq, dtype=int).reshape(-1, 1)
        scores = np.array([hmm_models[c].score(seq_arr) for c in class_list])
        exp_scores = np.exp(scores - scores.max())
        probs = exp_scores / exp_scores.sum()
        y_prob[i, :] = probs

    import os
    os.makedirs("model_outputs", exist_ok=True)
    np.save("model_outputs/hmm_y_true.npy", y_test)
    np.save("model_outputs/hmm_y_pred.npy", y_pred)
    np.save("model_outputs/hmm_y_prob.npy", y_prob)
    print("Saved HMM outputs to model_outputs/")


if __name__ == "__main__":
    main()
