"""Breath detection eval: P / R / F1 / IoU for predicted vs true intervals."""
 
from __future__ import annotations
from dataclasses import dataclass
 
 
def compute_IOU(true, pred):
    intersection = max(0, min(true.end, pred.end) - max(true.begin, pred.begin))
    union = (true.end - true.begin) + (pred.end - pred.begin) - intersection
    return intersection / union if union > 0 else 0.0
 
 
@dataclass
class Metrics:
    precision: float
    recall: float
    f1: float
    mean_iou: float
    tp: int
    fp: int
    fn: int
 
 
def match_intervals(
    preds: list[set], trues: list[set]
) -> tuple[int, int, int, list[float]]:
    """
    Greedy one-to-one matching: each predicted interval is matched to the
    first overlapping ground-truth interval that hasn't been claimed yet.
    Returns (TP, FP, FN, list_of_matched_ious).
    """
    matched_gt = set()
    ious: list[float] = []
 
    for pred in preds:
        best_idx, best_iou = -1, -1.0
        for i, true in enumerate(trues):
            if i in matched_gt:
                continue
            if pred.end > true.begin and true.end > pred.begin:  # overlap exists
                iou = compute_IOU(true, pred)
                if iou > best_iou:
                    best_idx, best_iou = i, iou
        if best_idx >= 0:
            matched_gt.add(best_idx)
            ious.append(best_iou)
 
    tp = len(ious)
    fp = len(preds) - tp
    fn = len(trues) - tp
    return tp, fp, fn, ious
 
 
def evaluate(preds: list[set], trues: list[set]) -> Metrics:
    """Evaluate a single example."""
    tp, fp, fn, ious = match_intervals(preds, trues)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return Metrics(p, r, f1, sum(ious) / len(ious) if ious else 0.0, tp, fp, fn)
 
 
def evaluate_dataset(
    all_preds: list[list[set]], all_trues: list[list[set]]
) -> Metrics:
    """Micro-averaged metrics across a full dataset."""
    total_tp = total_fp = total_fn = 0
    all_ious: list[float] = []
    for preds, trues in zip(all_preds, all_trues):
        tp, fp, fn, ious = match_intervals(preds, trues)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_ious.extend(ious)
    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    mean_iou = sum(all_ious) / len(all_ious) if all_ious else 0.0
    return Metrics(p, r, f1, mean_iou, total_tp, total_fp, total_fn)
 