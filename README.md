# Task Recognition on the EMAKI Dataset  
Paper-style BPE + Transformer and Classical Baselines

This project is based on the EMAKI dataset from Zhang et al., *“Exploring Natural Language Processing Methods for Interactive Behaviour Modelling”* (INTERACT 2023).

Each record in EMAKI is a low-level interaction event (mouse move, click, key press, etc.) with a **task label**. In this project, we focus on **three tasks** (labels 3, 4, and 5) and our goal is:

> Given a **window of 50 interaction events**, predict **which task (3 / 4 / 5)** the user is doing.

All models in this repo use the **same windowing** and the **same train/test split**, so the results are directly comparable.

---

## 1. BPE + Transformer (paper-style model)

### 1.1 Intuition

The original paper treats user interaction sequences like a kind of **language**:

1. Each interaction event (mouse/keyboard event) is treated as a **token**.
2. We apply **Byte Pair Encoding (BPE)** to compress frequent patterns into a smaller sequence of sub-tokens.
3. We feed these BPE token IDs into a **Transformer encoder**.
4. The Transformer output is used to classify the **task** being performed in that window.

So the pipeline is:

> interactions → tokens → BPE IDs → Transformer → task label (3/4/5)

### 1.2 What `bpe_transformer.py` does

In our implementation (`bpe_transformer.py`), we do the following:

1. **Load the data**
   - We use `load_emaki("EMAKI_utt.pkl")` from `data_emaki.py` to load the EMAKI interactions into a DataFrame.

2. **Build windows**
   - We call `build_windows(df, window_size=50, stride=25)`.
   - This gives us:
     - `sequences`: lists of tokens (each is a 50-length window of events),
     - `labels`: task labels (3, 4, or 5).

3. **Split into train / val / test**
   - We do a stratified 80/20 split over all windows:
     - 80% → further split into **train (64%) + validation (16%)**
     - 20% → **test**
   - This keeps class proportions similar in each split.

4. **Train a BPE tokenizer (on train windows only)**
   - We write **only the training windows** to `emaki_tokens_train.txt`.
   - We train a BPE tokenizer with:
     - vocab size = 1000,
     - special tokens: `<PAD>` and `<UNK>`.
   - For each window (train/val/test), we:
     - encode it into BPE IDs,
     - pad or truncate to length 50.

5. **Transformer classifier**
   - We use a small Transformer encoder:
     - token embedding + positional embedding (`d_model = 128`),
     - 2 encoder layers (`num_layers = 2`),
     - 4 attention heads (`nhead = 4`),
     - feedforward size = 256,
     - average pooling over sequence → linear layer → 3 outputs (tasks 3/4/5).

6. **Training procedure**
   - Loss: cross-entropy.
   - Optimizer: Adam with learning rate `1e-3`.
   - Epochs: 8.
   - After each epoch:
     - we evaluate on the **validation set**,
     - compute accuracy and **macro F1**,
     - if macro F1 improves, we save that model as the “best so far”.
   - At the end:
     - we load the **best validation checkpoint** and evaluate it once on the **test set**.

### 1.3 How close this is to the paper

We are **following the spirit** of the paper:

- We treat interaction events as tokens (“interaction-as-language”).
- We use **BPE** to compress sequences.
- We use a **Transformer encoder** as the model.
- We evaluate using **macro F1** on a **3-class task recognition** problem.

However, there are important differences from the full paper setup:

- **Model scale**:
  - We use one small Transformer (128 hidden size, 2 layers),
  - The paper searches over several architectures (different sizes and depths).

- **Training budget**:
  - We train for 8 epochs on **CPU**.
  - The paper trains for many more epochs on a **GPU** (e.g., V100).

- **Hyperparameter search**:
  - We use a single configuration (with a small amount of validation-based tuning).
  - The paper does a broader hyperparameter search and reports the best F1 across multiple parameter sets.

- **Evaluation protocol**:
  - We use one stratified train/val/test split.
  - The paper uses **participant-independent** 5-fold cross-validation (splits by user, not by window).

Because of these differences, we do **not** expect to match the exact performance reported in the paper. Instead, we show how the paper’s method behaves under a smaller training budget and compare it to strong classical baselines.

### 1.4 Limitations and what we could still do

Things we did **not** do (but could improve performance):

- Try more Transformer sizes (`d_model` values), more layers, and different learning rates.
- Train for more epochs with early stopping (ideally on a GPU).
- Try different BPE vocab sizes and window lengths.
- Use label smoothing and heavier regularization (as the paper does).
- Implement full **5-fold participant-independent cross-validation** to match the paper’s evaluation protocol exactly.

### 1.5 BPE + Transformer results (our best run)

With our current setup (best validation checkpoint):

- **Accuracy:** ~0.632  
- **Macro F1:** ~0.624  

So the paper-style BPE + Transformer model is **competitive**, but under our constraints it does not beat the strongest classical baseline (the 3-gram model, see below).

---

## 2. Classical Baselines

All baselines use:

- the same 50-length windows (`window_size = 50`, `stride = 25`),
- the same three task labels (3, 4, 5),
- the same **80/20 stratified train/test split** over windows (`random_state = 42`),
- the same integer vocabulary over tokens.

This makes the comparison to the BPE + Transformer model fair.

### 2.1 Markov baseline (`markov_baseline.py`)

**Idea (bigram model per class)**

- For each task label, we build a **bigram Markov model**:
  - `P(token_t | token_{t-1})` is learned from training windows of that task.
- To classify a test window:
  - we compute the **log-likelihood** of its token sequence under each class’s Markov model,
  - we choose the class with the **highest log-likelihood**.

**Result:**

- **Accuracy:** ~0.651  
- **Macro F1:** ~0.647  

This simple Markov model is already a strong baseline.

---

### 2.2 3-gram baseline (`ngram_baseline.py`)

**Idea (trigram model per class)**

- This is similar to the Markov baseline, but we use a **trigram model** instead of bigram:
  - `P(token_t | token_{t-2}, token_{t-1})` for each class.
- We use Laplace smoothing to avoid zero probabilities.
- For each test window, we:
  - compute the log-likelihood under each class’s trigram model,
  - predict the class with the highest score.

**Result:**

- **Accuracy:** 0.6731  
- **Macro F1:** 0.6615  

This is our **best model overall** in this project. Using a slightly longer context (trigrams) helps capture short local patterns in the interaction sequences.

---

### 2.3 HMM baseline (`hmm_baseline.py`)

**Idea (Multinomial HMM per class)**

- For each task label, we train a **Multinomial Hidden Markov Model**:
  - discrete emissions (token IDs),
  - a small number of hidden states,
  - trained with EM using the `hmmlearn` library.
- To classify a test window, we:
  - compute the sequence log-likelihood under each class’s HMM,
  - choose the class with the highest log-likelihood.

**Result:**

- **Accuracy:** 0.2942  
- **Macro F1:** 0.1516  
- The confusion matrix shows that the HMM predicts **task 4 for every window**, so it collapses to a single-class predictor.

This tells us that, in our setup, naïvely training separate HMMs for each class is unstable and performs much worse than both n-gram models and the Transformer.

---

### 2.4 Summary of results

All models are tested on the **same task** with the **same split**:

| Model                   | Accuracy | Macro F1 |
|-------------------------|----------|----------|
| 3-gram baseline         | 0.6731   | 0.6615   |
| Markov (bigram)         | ~0.651   | ~0.647   |
| BPE + Transformer       | ~0.632   | ~0.624   |
| HMM baseline            | 0.2942   | 0.1516   |

Key takeaway:

- In our limited training setting, the **3-gram model** is the strongest.
- The **BPE + Transformer** model is competitive but slightly weaker than the best n-gram.
- The **HMM** baseline fails badly, collapsing to one class.

This is still useful: it shows that the deep paper-style model is not automatically better unless we can match the original training budget and tuning.

---

## 3. Repository Structure

Main files:

- `EMAKI_utt.pkl`  
  Preprocessed EMAKI dataset (utterance-level or interaction-level file used in this project).

- `data_emaki.py`  
  Contains helper functions:
  - `load_emaki(path)` – loads the dataset from the `.pkl` file.
  - `build_windows(df, window_size=50, stride=25)` – builds 50-length token windows and labels.

- `bpe_transformer.py`  
  Main implementation of the **BPE + Transformer** task recognition model.

- `bpe_transformer_search.py` (optional)  
  A script that tries multiple Transformer configurations and reports the best one based on validation macro F1.

- `markov_baseline.py`  
  Bigram Markov model baseline.

- `ngram_baseline.py`  
  3-gram (trigram) baseline.

- `hmm_baseline.py`  
  Multinomial HMM baseline.

- `requirements_bpe.txt`  
  Python package versions used for `bpe_transformer.py` (includes `torch`, `tokenizers`, and a NumPy version that works with this PyTorch build).

- `requirements_baselines.txt`  
  Python package versions for the three baselines (Markov, 3-gram, HMM).

---

## 4. Environments and Installation

Because the BPE + Transformer code depends on a specific combination of `torch`, `tokenizers`, and `numpy`, we recommend **two separate environments**:

- one for the **BPE + Transformer** model,
- one for the **classical baselines**.

You can use `conda`, `venv`, or any other virtual environment tool. Below is a simple `venv` example (on Windows PowerShell).

### 4.1 Environment 1: BPE + Transformer

1. Create and activate a virtual environment:

```bash
python -m venv bpe_env
bpe_env\Scripts\activate
```

2. Install dependencies for the BPE model:

```bash
pip install -r requirements_bpe.txt
```

3. Make sure EMAKI_utt.pkl and data_emaki.py are in the same folder as bpe_transformer.py.

4. Run the BPE + Transformer model (note this model will take a long time to finish training):

```bash
python bpe_transformer.py
```

This will train the BPE tokenizer and Transformer, print training / validation metrics, and finally report test accuracy, macro F1, and a confusion matrix.

### 4.2 Environment 2: Baselines (Markov, 3-gram, HMM)

1. Create and activate another environment:

```bash
python -m venv baseline_env
baseline_env\Scripts\activate
```

2. Install baseline dependencies:

```bash
pip install -r requirements_baselines.txt
```

3. Make sure EMAKI_utt.pkl and data_emaki.py are in the same folder as:

- markov_baseline.py

- ngram_baseline.py

- hmm_baseline.py

4. Run each baseline:

```bash
python markov_baseline.py
python ngram_baseline.py
python hmm_baseline.py
```
Each script will print its own accuracy, macro F1, and confusion matrix on the same EMAKI task.
## Visualization

## Model Output Saving

Each baseline and transformer model script has been modified to save predictions and probabilities for downstream visualization and analysis.

**What is saved:**
- `{model}_y_true.npy` — Ground truth labels from test set
- `{model}_y_pred.npy` — Model predictions on test set
- `{model}_y_prob.npy` — Predicted class probabilities (softmax output) on test set

**Where it is saved:**
All outputs are saved to `model_outputs/` folder in the project root.

**How it works:**
After training and evaluation on the test set, each model script computes softmax probabilities from raw scores/logits, then saves all three arrays using `np.save()`. This enables:
- Post-hoc evaluation metrics (ROC-AUC, PR-AUC, per-class metrics)
- Visualization of confusion matrices, ROC curves, precision-recall curves
- Model comparison across baselines and deep learning approaches
- Reproducibility without re-training

**Automatic generation:**
Run each model script to generate outputs, commands are the same as we've specified above.

After all scripts complete, `model_outputs/` will contain 12 files (3 per model × 4 models).

## Visualization

Professional visualization script that generates comprehensive performance metrics and plots for all trained models.

**Features:**
- Loads model outputs (y_true, y_pred, y_prob) from `model_outputs/` folder
- Generates per-model visualizations:
  - Confusion matrices (raw counts + normalized percentages)
  - Per-class metrics (Precision, Recall, F1)
  - ROC curves (One-vs-Rest for each class)
  - Precision-Recall curves (One-vs-Rest for each class)
  - Class distribution (True vs Predicted counts)
  - Summary report (single-page overview of all metrics)
- Generates model comparison chart (all models ranked by F1-score)
- High-resolution output (300 dpi) saved to `visualizations/` folder

**Usage:**
```bash
python3 visualize.py --input_dir model_outputs --output_dir visualizations --labels 3 4 5
```

Or with defaults:
```bash
python3 visualize.py
```

**Output files:**
- `00_model_comparison.png` — All models ranked by accuracy, F1, precision, recall
- `{model}_01_confusion_matrix.png` — Confusion matrix (counts + normalized %)
- `{model}_02_per_class_metrics.png` — Precision/Recall/F1 per class
- `{model}_03_roc_curves.png` — ROC curves for each class
- `{model}_04_pr_curves.png` — Precision-Recall curves for each class
- `{model}_05_class_distribution.png` — True vs Predicted class distribution
- `{model}_06_summary_report.png` — Single-page metrics summary

**Prerequisites:** 
- Model outputs must exist as `.npy` files in `model_outputs/` with naming: `{model}_y_true.npy`, `{model}_y_pred.npy`, `{model}_y_prob.npy`
- Dependencies: numpy, matplotlib, seaborn, scikit-learn