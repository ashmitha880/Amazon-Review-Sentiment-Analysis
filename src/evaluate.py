"""
Shared evaluation utilities so both models are scored the same way and
results are directly comparable in notebook 05.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from src.config import LABELS


def print_report(y_true, y_pred, model_name: str = "Model"):
    print(f"\n=== {model_name} — Classification Report ===")
    print(classification_report(y_true, y_pred, labels=LABELS))


def plot_confusion_matrix(y_true, y_pred, model_name: str = "Model", save_path=None, cmap="Blues"):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model_name} — Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved confusion matrix to {save_path}")
    plt.show()
