import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, precision_recall_fscore_support,
    roc_curve, auc, precision_recall_curve, roc_auc_score
)
from sklearn.preprocessing import label_binarize

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_arrays(path, model):
    a = {}
    for name in ("y_true", "y_pred", "y_prob"):
        p = os.path.join(path, f"{model}_{name}.npy")
        if os.path.exists(p):
            a[name] = np.load(p)
        else:
            a[name] = None
    return a

def compute_all_metrics(y_true, y_pred, y_prob, labels):
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    metrics = {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
    }
    
    if y_prob is not None and y_prob.shape[1] == len(labels):
        y_true_bin = label_binarize(y_true, classes=labels)
        roc_auc_macro = roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr")
        metrics["roc_auc_macro"] = roc_auc_macro
    
    prfs = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    metrics["per_class"] = {
        "precision": prfs[0],
        "recall": prfs[1],
        "f1": prfs[2],
        "support": prfs[3],
    }
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, labels, model_name, outdir):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax1, cbar_kws={"label": "Count"})
    ax1.set_xlabel("Predicted Label", fontsize=12, fontweight='bold')
    ax1.set_ylabel("True Label", fontsize=12, fontweight='bold')
    ax1.set_title(f"{model_name} - Confusion Matrix (Counts)", fontsize=13, fontweight='bold')
    
    sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap="RdYlGn", xticklabels=labels, yticklabels=labels, ax=ax2, cbar_kws={"label": "Proportion"}, vmin=0, vmax=1)
    ax2.set_xlabel("Predicted Label", fontsize=12, fontweight='bold')
    ax2.set_ylabel("True Label", fontsize=12, fontweight='bold')
    ax2.set_title(f"{model_name} - Confusion Matrix (Normalized %)", fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{model_name}_01_confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {model_name}_01_confusion_matrix.png")

def plot_per_class_metrics(y_true, y_pred, labels, model_name, metrics, outdir):
    per_class = metrics["per_class"]
    x_pos = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x_pos - width, per_class["precision"], width, label="Precision", color="#3498db", edgecolor="black")
    ax.bar(x_pos, per_class["recall"], width, label="Recall", color="#e74c3c", edgecolor="black")
    ax.bar(x_pos + width, per_class["f1"], width, label="F1-Score", color="#2ecc71", edgecolor="black")
    
    ax.set_xlabel("Class Label", fontsize=12, fontweight='bold')
    ax.set_ylabel("Score", fontsize=12, fontweight='bold')
    ax.set_title(f"{model_name} - Per-Class Performance Metrics", fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylim([0, 1])
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(axis='y', alpha=0.3)
    
    for i, (p, r, f) in enumerate(zip(per_class["precision"], per_class["recall"], per_class["f1"])):
        ax.text(i - width, p + 0.02, f"{p:.3f}", ha='center', fontsize=9)
        ax.text(i, r + 0.02, f"{r:.3f}", ha='center', fontsize=9)
        ax.text(i + width, f + 0.02, f"{f:.3f}", ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{model_name}_02_per_class_metrics.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {model_name}_02_per_class_metrics.png")

def plot_roc_curves(y_true, y_prob, labels, model_name, outdir):
    y_true_bin = label_binarize(y_true, classes=labels)
    n_classes = len(labels)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2.5, label=f"Class {labels[i]} (AUC={roc_auc:.4f})")
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label="Random Classifier (AUC=0.5)", alpha=0.7)
    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight='bold')
    ax.set_ylabel("True Positive Rate", fontsize=12, fontweight='bold')
    ax.set_title(f"{model_name} - ROC Curves (One-vs-Rest)", fontsize=13, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{model_name}_03_roc_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {model_name}_03_roc_curves.png")

def plot_pr_curves(y_true, y_prob, labels, model_name, outdir):
    y_true_bin = label_binarize(y_true, classes=labels)
    n_classes = len(labels)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i in range(n_classes):
        prec, rec, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        ap = auc(rec, prec)
        ax.plot(rec, prec, lw=2.5, label=f"Class {labels[i]} (AP={ap:.4f})")
    
    ax.set_xlabel("Recall", fontsize=12, fontweight='bold')
    ax.set_ylabel("Precision", fontsize=12, fontweight='bold')
    ax.set_title(f"{model_name} - Precision-Recall Curves (One-vs-Rest)", fontsize=13, fontweight='bold')
    ax.legend(loc="best", fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{model_name}_04_pr_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {model_name}_04_pr_curves.png")

def plot_class_distribution(y_true, y_pred, labels, model_name, outdir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    true_counts = [np.sum(y_true == l) for l in labels]
    pred_counts = [np.sum(y_pred == l) for l in labels]
    
    x_pos = np.arange(len(labels))
    width = 0.35
    
    ax1.bar(x_pos - width/2, true_counts, width, label="True", color="#9b59b6", edgecolor="black")
    ax1.bar(x_pos + width/2, pred_counts, width, label="Predicted", color="#f39c12", edgecolor="black")
    ax1.set_xlabel("Class Label", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Count", fontsize=12, fontweight='bold')
    ax1.set_title(f"{model_name} - Class Distribution (True vs Predicted)", fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels)
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    
    for i, (t, p) in enumerate(zip(true_counts, pred_counts)):
        ax1.text(i - width/2, t + 5, str(t), ha='center', fontsize=10, fontweight='bold')
        ax1.text(i + width/2, p + 5, str(p), ha='center', fontsize=10, fontweight='bold')
    
    true_pct = [100 * c / len(y_true) for c in true_counts]
    pred_pct = [100 * c / len(y_pred) for c in pred_counts]
    
    ax2.bar(x_pos - width/2, true_pct, width, label="True", color="#9b59b6", edgecolor="black")
    ax2.bar(x_pos + width/2, pred_pct, width, label="Predicted", color="#f39c12", edgecolor="black")
    ax2.set_xlabel("Class Label", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Percentage (%)", fontsize=12, fontweight='bold')
    ax2.set_title(f"{model_name} - Class Distribution (Percentage)", fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels)
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    
    for i, (t, p) in enumerate(zip(true_pct, pred_pct)):
        ax2.text(i - width/2, t + 1, f"{t:.1f}%", ha='center', fontsize=9, fontweight='bold')
        ax2.text(i + width/2, p + 1, f"{p:.1f}%", ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{model_name}_05_class_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {model_name}_05_class_distribution.png")

def plot_summary_metrics(y_true, y_pred, y_prob, labels, model_name, metrics, outdir):
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    metric_names = ["Accuracy", "Precision\n(Macro)", "Recall\n(Macro)", "F1-Score\n(Macro)"]
    metric_values = [metrics["accuracy"], metrics["precision_macro"], metrics["recall_macro"], metrics["f1_macro"]]
    colors_bars = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    
    ax1.barh(metric_names, metric_values, color=colors_bars, edgecolor="black", linewidth=1.5)
    ax1.set_xlim([0, 1])
    ax1.set_xlabel("Score", fontsize=11, fontweight='bold')
    ax1.set_title("Overall Metrics", fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    for i, v in enumerate(metric_values):
        ax1.text(v + 0.02, i, f"{v:.4f}", va='center', fontweight='bold', fontsize=10)
    
    per_class_f1 = metrics["per_class"]["f1"]
    ax2.bar(range(len(labels)), per_class_f1, color="#2ecc71", edgecolor="black", linewidth=1.5)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("F1-Score", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Class", fontsize=11, fontweight='bold')
    ax2.set_title("F1-Score by Class", fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(axis='y', alpha=0.3)
    for i, v in enumerate(per_class_f1):
        ax2.text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold', fontsize=10)
    
    per_class_support = metrics["per_class"]["support"].astype(int)
    ax3.bar(range(len(labels)), per_class_support, color="#9b59b6", edgecolor="black", linewidth=1.5)
    ax3.set_xticks(range(len(labels)))
    ax3.set_xticklabels(labels)
    ax3.set_ylabel("Count", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Class", fontsize=11, fontweight='bold')
    ax3.set_title("Test Set Support (Samples per Class)", fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    for i, v in enumerate(per_class_support):
        ax3.text(i, v + 20, str(v), ha='center', fontweight='bold', fontsize=10)
    
    text_str = f"Model: {model_name}\n\n"
    text_str += f"Accuracy:     {metrics['accuracy']:.4f}\n"
    text_str += f"Precision:    {metrics['precision_macro']:.4f}\n"
    text_str += f"Recall:       {metrics['recall_macro']:.4f}\n"
    text_str += f"F1-Score:     {metrics['f1_macro']:.4f}\n"
    if "roc_auc_macro" in metrics:
        text_str += f"ROC AUC:      {metrics['roc_auc_macro']:.4f}\n"
    text_str += f"\nTest Samples: {len(y_true)}\n"
    text_str += f"Classes:      {len(labels)}"
    
    ax4.text(0.1, 0.5, text_str, transform=ax4.transAxes, fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), family='monospace', fontweight='bold')
    ax4.axis('off')
    
    plt.suptitle(f"{model_name} - Summary Report", fontsize=14, fontweight='bold', y=0.995)
    plt.savefig(os.path.join(outdir, f"{model_name}_06_summary_report.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {model_name}_06_summary_report.png")

def plot_model_comparison(all_metrics, outdir):
    models = list(all_metrics.keys())
    accuracies = [all_metrics[m]["accuracy"] for m in models]
    f1_scores = [all_metrics[m]["f1_macro"] for m in models]
    precisions = [all_metrics[m]["precision_macro"] for m in models]
    recalls = [all_metrics[m]["recall_macro"] for m in models]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = ['#27ae60' if i == np.argmax(accuracies) else '#3498db' for i in range(len(models))]
    
    axes[0, 0].bar(models, accuracies, color=colors, edgecolor="black", linewidth=1.5)
    axes[0, 0].set_ylabel("Accuracy", fontsize=11, fontweight='bold')
    axes[0, 0].set_title("Accuracy Comparison", fontsize=12, fontweight='bold')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold', fontsize=10)
    plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    colors = ['#27ae60' if i == np.argmax(f1_scores) else '#e74c3c' for i in range(len(models))]
    axes[0, 1].bar(models, f1_scores, color=colors, edgecolor="black", linewidth=1.5)
    axes[0, 1].set_ylabel("Macro F1", fontsize=11, fontweight='bold')
    axes[0, 1].set_title("F1-Score Comparison", fontsize=12, fontweight='bold')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(f1_scores):
        axes[0, 1].text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold', fontsize=10)
    plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    axes[1, 0].bar(models, precisions, color="#2ecc71", edgecolor="black", linewidth=1.5, alpha=0.7, label="Precision")
    axes[1, 0].bar(models, recalls, color="#f39c12", edgecolor="black", linewidth=1.5, alpha=0.7, label="Recall")
    axes[1, 0].set_ylabel("Score", fontsize=11, fontweight='bold')
    axes[1, 0].set_title("Precision vs Recall Comparison", fontsize=12, fontweight='bold')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(axis='y', alpha=0.3)
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    sorted_idx = np.argsort(f1_scores)[::-1]
    sorted_models = [models[i] for i in sorted_idx]
    sorted_f1 = [f1_scores[i] for i in sorted_idx]
    colors_rank = ['#27ae60', '#3498db', '#e74c3c', '#f39c12'][:len(sorted_models)]
    
    axes[1, 1].barh(sorted_models, sorted_f1, color=colors_rank, edgecolor="black", linewidth=1.5)
    axes[1, 1].set_xlabel("Macro F1", fontsize=11, fontweight='bold')
    axes[1, 1].set_title("Model Ranking (by F1-Score)", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlim([0, 1])
    axes[1, 1].grid(axis='x', alpha=0.3)
    for i, v in enumerate(sorted_f1):
        axes[1, 1].text(v + 0.02, i, f"{v:.4f}", va='center', fontweight='bold', fontsize=10)
    
    plt.suptitle("Model Comparison - All Metrics", fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "00_model_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: 00_model_comparison.png")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default="model_outputs")
    p.add_argument("--output_dir", default="visualizations")
    p.add_argument("--labels", nargs="+", type=int, default=[3, 4, 5])
    args = p.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not os.path.isdir(args.input_dir):
        print("input_dir not found:", args.input_dir)
        return
    
    models = set()
    for fname in os.listdir(args.input_dir):
        if fname.endswith("_y_true.npy"):
            models.add(fname.rsplit("_y_true.npy", 1)[0])
    
    models = sorted(models)
    
    if not models:
        print("no model outputs found in", args.input_dir)
        return
    
    labels = args.labels
    all_metrics = {}
    
    print("\n" + "="*70)
    print("PROFESSIONAL VISUALIZATION GENERATION")
    print("="*70 + "\n")
    
    for m in models:
        print(f"\nProcessing: {m}")
        print("-" * 50)
        
        arrs = load_arrays(args.input_dir, m)
        
        if arrs["y_true"] is None or arrs["y_pred"] is None:
            print(f"  ⚠ Skipping {m} (missing y_true or y_pred)")
            continue
        
        y_true = arrs["y_true"]
        y_pred = arrs["y_pred"]
        y_prob = arrs["y_prob"]
        
        metrics = compute_all_metrics(y_true, y_pred, y_prob, labels)
        all_metrics[m] = metrics
        
        print(f"  Accuracy:       {metrics['accuracy']:.4f}")
        print(f"  Precision:      {metrics['precision_macro']:.4f}")
        print(f"  Recall:         {metrics['recall_macro']:.4f}")
        print(f"  F1-Score:       {metrics['f1_macro']:.4f}")
        if "roc_auc_macro" in metrics:
            print(f"  ROC AUC:        {metrics['roc_auc_macro']:.4f}")
        
        print(f"\n  Generating visualizations...")
        plot_confusion_matrix(y_true, y_pred, labels, m, args.output_dir)
        plot_per_class_metrics(y_true, y_pred, labels, m, metrics, args.output_dir)
        plot_class_distribution(y_true, y_pred, labels, m, args.output_dir)
        plot_summary_metrics(y_true, y_pred, y_prob, labels, m, metrics, args.output_dir)
        
        if y_prob is not None and y_prob.shape[0] == y_true.shape[0] and y_prob.shape[1] == len(labels):
            plot_roc_curves(y_true, y_prob, labels, m, args.output_dir)
            plot_pr_curves(y_true, y_prob, labels, m, args.output_dir)
        else:
            print(f"  ⚠ Skipping ROC/PR (y_prob shape mismatch or missing)")
    
    if all_metrics:
        print(f"\nGenerating model comparison...")
        plot_model_comparison(all_metrics, args.output_dir)
    
    print("\n" + "="*70)
    print(f"✓ All visualizations saved to: {args.output_dir}/")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()