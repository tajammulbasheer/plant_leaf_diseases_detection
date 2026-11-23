import torch
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def classification_metrics(labels, preds):
    """Accuracy and macro precision / recall / F1 over the labels present in labels or preds."""
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "macro_precision": precision_score(labels, preds, average="macro", zero_division=0),
        "macro_recall": recall_score(labels, preds, average="macro", zero_division=0),
    }


def mcnemar(base_correct, new_correct):
    """Exact two-sided McNemar test on paired per-image correctness (boolean arrays): images the new model
    gets right that the baseline got wrong ("gained"), the reverse ("lost"), and the p-value."""
    gained = int((~base_correct & new_correct).sum())
    lost = int((base_correct & ~new_correct).sum())
    p = binomtest(gained, gained + lost, 0.5).pvalue if gained + lost else 1.0
    return {"gained": gained, "lost": lost, "p_value": float(p)}


@torch.no_grad()
def evaluate_classifier(model, dataloader, device, class_names):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in dataloader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    return classification_metrics(all_labels, all_preds)
