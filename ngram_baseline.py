import numpy as np
from collections import Counter, defaultdict

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


# build n-gram counts for one class
def build_ngram_counts(sequences, n, vocab_size):
    # counts[context][token] = frequency
    counts = defaultdict(Counter)

    for seq in sequences:
        if len(seq) < n:
            continue

        # slide window over sequence
        for i in range(n - 1, len(seq)):
            context = tuple(seq[i - n + 1 : i])  # previous n-1 tokens
            token = seq[i]
            counts[context][token] += 1

    # precompute totals for each context
    context_totals = {}
    for ctx, counter in counts.items():
        context_totals[ctx] = sum(counter.values())

    return counts, context_totals


# compute log-likelihood of one sequence under one class n-gram model
def ngram_log_likelihood(seq, n, counts, context_totals, vocab_size):
    log_ll = 0.0

    if len(seq) < n:
        # if sequence is too short, give neutral score
        return 0.0

    for i in range(n - 1, len(seq)):
        context = tuple(seq[i - n + 1 : i])
        token = seq[i]

        # if context never seen in training, use uniform distribution
        if context not in context_totals:
            prob = 1.0 / vocab_size
        else:
            count_ctx = context_totals[context]
            count_tok = counts[context][token]

            # Laplace smoothing
            prob = (count_tok + 1.0) / (count_ctx + vocab_size)

        # accumulate log probability
        if prob <= 0.0:
            prob = 1e-12
        log_ll += np.log(prob)

    return log_ll


def main():
    # load emaki dataframe
    df = load_emaki("EMAKI_utt.pkl")

    # build token windows and labels
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

    # same 80/20 stratified split as markov and HMM
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

    # choose n for n-gram model (3 = trigram)
    N = 3

    # group training sequences by class label
    train_by_class = {}
    for seq, label in zip(X_train, y_train):
        train_by_class.setdefault(label, []).append(np.array(seq, dtype=int))

    # build n-gram models for each class
    ngram_models = {}
    for c, seqs in train_by_class.items():
        print(f"building {N}-gram model for class {c} with {len(seqs)} windows")
        counts, context_totals = build_ngram_counts(seqs, N, vocab_size)
        ngram_models[c] = (counts, context_totals)

    # predict labels for test windows
    y_pred = []
    for seq in X_test:
        seq_arr = np.array(seq, dtype=int)

        # compute log-likelihood for each class
        scores = {}
        for c, (counts, context_totals) in ngram_models.items():
            scores[c] = ngram_log_likelihood(seq_arr, N, counts, context_totals, vocab_size)

        # pick class with highest log-likelihood
        best_class = max(scores, key=scores.get)
        y_pred.append(best_class)

    y_pred = np.array(y_pred)

    # evaluate baseline
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")

    print(f"{N}-gram baseline accuracy: {acc:.4f}")
    print(f"{N}-gram baseline macro F1: {f1_macro:.4f}")
    print("confusion matrix:\n", confusion_matrix(y_test, y_pred))


    class_list = sorted(ngram_models.keys())
    y_prob = np.zeros((len(X_test), len(class_list)))
    for i, seq in enumerate(X_test):
        scores = []
        for c in class_list:
            counts, context_totals = ngram_models[c]
            scores.append(ngram_log_likelihood(np.array(seq, dtype=int), N, counts, context_totals, vocab_size))
        scores = np.array(scores)
        exp_scores = np.exp(scores - scores.max())
        probs = exp_scores / exp_scores.sum()
        y_prob[i, :] = probs

    import os
    os.makedirs("model_outputs", exist_ok=True)
    np.save("model_outputs/ngram_y_true.npy", y_test)
    np.save("model_outputs/ngram_y_pred.npy", y_pred)
    np.save("model_outputs/ngram_y_prob.npy", y_prob)
    print("Saved N-gram outputs to model_outputs/")


if __name__ == "__main__":
    main()
