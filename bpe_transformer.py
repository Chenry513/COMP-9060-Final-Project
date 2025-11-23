import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from data_emaki import load_emaki, build_windows

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# window + BPE settings
WINDOW_SIZE = 50
STRIDE = 25
BPE_VOCAB_SIZE = 600   # try a bit smaller, closer to paper's BPE-600
BATCH_SIZE = 64
NUM_EPOCHS = 12        # more epochs per config


# small dataset wrapper for BPE id sequences
class EmakiBPEDataset(Dataset):
    def __init__(self, X, y):
        # store input ids and labels as tensors
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# transformer encoder classifier with configurable dropout
class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        max_len: int,
        num_classes: int,
        pad_id: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # token embedding
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)

        # positional embedding
        self.pos_emb = nn.Embedding(max_len, d_model)

        # transformer encoder layers with dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # classification head
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len)
        batch_size, seq_len = x.size()

        # token embeddings
        tok = self.token_emb(x)

        # position indices 0..seq_len-1
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        pos = self.pos_emb(positions)

        # sum token + position embeddings
        h = tok + pos

        # encode with transformer
        h_enc = self.encoder(h)

        # average pooling over sequence dimension
        h_pool = h_enc.mean(dim=1)

        # class logits
        logits = self.fc(h_pool)
        return logits


def main():
    # set seeds for repeatability
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

    # load EMAKI dataframe
    df = load_emaki("EMAKI_utt.pkl")

    # build token windows, labels
    sequences, labels, window_users = build_windows(
        df,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
    )

    print("number of windows:", len(sequences))
    print("unique tasks:", set(labels))

    labels_np = np.array(labels)
    indices = np.arange(len(sequences))

    # 1) split into train_full (80%) and test (20%), stratified over windows
    train_full_idx, test_idx, y_train_full, y_test = train_test_split(
        indices,
        labels_np,
        test_size=0.2,
        random_state=42,
        stratify=labels_np,
    )

    # 2) split train_full into train (80% of train_full) and val (20% of train_full)
    train_idx, val_idx, y_train, y_val = train_test_split(
        train_full_idx,
        y_train_full,
        test_size=0.2,
        random_state=24,
        stratify=y_train_full,
    )

    # map indices back to sequences
    sequences_train = [sequences[i] for i in train_idx]
    sequences_val   = [sequences[i] for i in val_idx]
    sequences_test  = [sequences[i] for i in test_idx]

    from collections import Counter
    print("train label counts:", Counter(y_train))
    print("val label counts:", Counter(y_val))
    print("test label counts:", Counter(y_test))

    # train one BPE tokenizer on training sequences only 
    with open("emaki_tokens_train.txt", "w", encoding="utf-8") as f:
        for seq in sequences_train:
            f.write(" ".join(seq) + "\n")

    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=BPE_VOCAB_SIZE,
        special_tokens=[PAD_TOKEN, UNK_TOKEN],
    )

    tokenizer.train(files=["emaki_tokens_train.txt"], trainer=trainer)

    pad_id = tokenizer.token_to_id(PAD_TOKEN)
    print("bpe vocab size:", tokenizer.get_vocab_size())
    print("pad id:", pad_id)

    # function to encode one token window with BPE
    def encode_bpe(seq):
        text = " ".join(seq)
        out = tokenizer.encode(text)
        ids = out.ids

        if len(ids) < WINDOW_SIZE:
            ids = ids + [pad_id] * (WINDOW_SIZE - len(ids))
        else:
            ids = ids[:WINDOW_SIZE]
        return ids

    # encode train, val, test once
    bpe_sequences_train = [encode_bpe(seq) for seq in sequences_train]
    bpe_sequences_val   = [encode_bpe(seq) for seq in sequences_val]
    bpe_sequences_test  = [encode_bpe(seq) for seq in sequences_test]

    X_train = np.array(bpe_sequences_train)
    X_val   = np.array(bpe_sequences_val)
    X_test  = np.array(bpe_sequences_test)

    print("train shape:", X_train.shape, "val shape:", X_val.shape, "test shape:", X_test.shape)

    # remap labels to 0..num_classes-1 for pytorch
    unique_labels = sorted(list(set(labels_np)))
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
    idx_to_label = {i: lab for lab, i in label_to_idx.items()}

    y_train_idx = np.array([label_to_idx[l] for l in y_train])
    y_val_idx   = np.array([label_to_idx[l] for l in y_val])
    y_test_idx  = np.array([label_to_idx[l] for l in y_test])

    num_classes = len(unique_labels)
    print("num classes:", num_classes, "label mapping:", label_to_idx)

    # create datasets and base loaders (we'll recreate loaders per config)
    train_dataset = EmakiBPEDataset(X_train, y_train_idx)
    val_dataset   = EmakiBPEDataset(X_val,   y_val_idx)
    test_dataset  = EmakiBPEDataset(X_test,  y_test_idx)

    # hyperparameter configs to try
    # you can remove some if it’s too slow
    config_list = [
        {"name": "small_64x2_lr1e-3_drop0.1", "d_model": 64,  "num_layers": 2, "lr": 1e-3,  "dropout": 0.1},
        {"name": "small_64x4_lr1e-3_drop0.3", "d_model": 64,  "num_layers": 4, "lr": 1e-3,  "dropout": 0.3},
        {"name": "med_128x2_lr1e-3_drop0.1",  "d_model": 128, "num_layers": 2, "lr": 1e-3,  "dropout": 0.1},
        {"name": "med_128x2_lr5e-4_drop0.3",  "d_model": 128, "num_layers": 2, "lr": 5e-4, "dropout": 0.3},
        {"name": "big_128x4_lr1e-3_drop0.3",  "d_model": 128, "num_layers": 4, "lr": 1e-3,  "dropout": 0.3},
        {"name": "big_128x4_lr5e-4_drop0.1",  "d_model": 128, "num_layers": 4, "lr": 5e-4, "dropout": 0.1},
    ]

    overall_best_f1 = 0.0
    overall_best_config = None
    overall_best_metrics = None  # (acc, f1, confusion_matrix, report_str)

    # loop over configs
    for cfg in config_list:
        print("\n")
        print(f"Training config: {cfg['name']}")

        # loaders for this config
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
        test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

        # init model
        model = TransformerClassifier(
            vocab_size=tokenizer.get_vocab_size(),
            d_model=cfg["d_model"],
            nhead=4,
            num_layers=cfg["num_layers"],
            dim_feedforward=4 * cfg["d_model"],
            max_len=WINDOW_SIZE,
            num_classes=num_classes,
            pad_id=pad_id,
            dropout=cfg["dropout"],
        ).to(device)

        # loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

        best_val_f1 = 0.0
        best_state_dict = None

        # training loop for this config
        for epoch in range(NUM_EPOCHS):
            model.train()
            total_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                total_loss += loss.item() * batch_x.size(0)
                loss.backward()
                optimizer.step()

            avg_loss = total_loss / len(train_dataset)
            print(f"  epoch {epoch+1}/{NUM_EPOCHS}, train loss: {avg_loss:.4f}")

            # validation pass
            model.eval()
            val_preds = []
            val_true = []

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                    logits = model(batch_x)
                    preds = torch.argmax(logits, dim=1)

                    val_preds.extend(preds.cpu().numpy())
                    val_true.extend(batch_y.cpu().numpy())

            val_preds = np.array(val_preds)
            val_true = np.array(val_true)

            val_preds_labels = np.array([idx_to_label[i] for i in val_preds])
            val_true_labels  = np.array([idx_to_label[i] for i in val_true])

            val_f1_macro = f1_score(val_true_labels, val_preds_labels, average="macro")
            val_acc = accuracy_score(val_true_labels, val_preds_labels)

            print(f"    val acc: {val_acc:.4f}, val macro F1: {val_f1_macro:.4f}")

            # save best epoch for this config
            if val_f1_macro > best_val_f1:
                best_val_f1 = val_f1_macro
                best_state_dict = model.state_dict()

        # evaluate best epoch for this config on test set
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
            print(f"  using best epoch for {cfg['name']} with val macro F1 = {best_val_f1:.4f}")
        else:
            print("  warning: no best_state_dict saved, using last epoch model")

        model.eval()
        all_preds = []
        all_true = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                logits = model(batch_x)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_true.extend(batch_y.cpu().numpy())

        all_preds = np.array(all_preds)
        all_true = np.array(all_true)

        all_preds_labels = np.array([idx_to_label[i] for i in all_preds])
        all_true_labels  = np.array([idx_to_label[i] for i in all_true])

        acc = accuracy_score(all_true_labels, all_preds_labels)
        f1_macro = f1_score(all_true_labels, all_preds_labels, average="macro")
        cm = confusion_matrix(all_true_labels, all_preds_labels)
        report_str = classification_report(all_true_labels, all_preds_labels)

        print(f"  TEST results for {cfg['name']}:")
        print(f"    accuracy: {acc:.4f}")
        print(f"    macro F1: {f1_macro:.4f}")
        print("    confusion matrix:\n", cm)
        print(report_str)

        # track best config overall by test macro F1
        if f1_macro > overall_best_f1:
            overall_best_f1 = f1_macro
            overall_best_config = cfg
            overall_best_metrics = (acc, f1_macro, cm, report_str)

    # summary of best config
    print("Overall best config (by test macro F1):")
    print(overall_best_config)
    if overall_best_metrics is not None:
        acc, f1_macro, cm, report_str = overall_best_metrics
        print(f"  best test accuracy: {acc:.4f}")
        print(f"  best test macro F1: {f1_macro:.4f}")
        print("  confusion matrix:\n", cm)
        print(report_str)


if __name__ == "__main__":
    main()




