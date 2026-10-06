r"""
Evaluation Metrics for Trustworthy Remote Sensing VQA:
1. Standard Accuracy & Macro-F1
2. Risk-Coverage Curves & AURC (Area Under Risk-Coverage Curve)
3. Expected Calibration Error (ECE) & Brier Score
4. Selective Prediction & Abstention Precision/Recall
"""

import numpy as np

def compute_vqa_accuracy(predictions, ground_truths):
    """
    Computes Exact Match and Semantic Category Accuracy.
    """
    correct = 0
    total = len(predictions)
    if total == 0:
        return 0.0

    for pred, gt in zip(predictions, ground_truths):
        pred_clean = pred.lower().strip()
        gt_clean = gt.lower().strip()
        # Direct substring or exact match
        if gt_clean in pred_clean or pred_clean in gt_clean:
            correct += 1
    return float(correct / total)

def compute_expected_calibration_error(confidences, accuracies, num_bins=10):
    r"""
    Expected Calibration Error (ECE) across confidence bins.
    ECE = \sum_{b=1}^B \frac{|B_b|}{N} |acc(B_b) - conf(B_b)|
    """
    confidences = np.array(confidences)
    accuracies = np.array(accuracies)
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    total_samples = len(confidences)

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_count = np.sum(in_bin)

        if bin_count > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_count / total_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)

def compute_risk_coverage_curve(confidences, accuracies, thresholds=None):
    """
    Computes Risk-Coverage points across varying confidence thresholds.
    Coverage (phi): fraction of answered queries.
    Risk (R): error rate on answered queries (1 - selective_accuracy).
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    confidences = np.array(confidences)
    accuracies = np.array(accuracies)
    total_samples = len(confidences)

    coverages = []
    risks = []
    selective_accs = []

    for tau in thresholds:
        accepted = confidences >= tau
        num_accepted = np.sum(accepted)

        if num_accepted == 0:
            coverage = 0.0
            sel_acc = 1.0
            risk = 0.0
        else:
            coverage = float(num_accepted / total_samples)
            sel_acc = float(np.mean(accuracies[accepted]))
            risk = float(1.0 - sel_acc)

        coverages.append(coverage)
        risks.append(risk)
        selective_accs.append(sel_acc)

    # Sort by coverage descending for standard plotting
    sorted_indices = np.argsort(coverages)
    cov_arr = np.array(coverages)[sorted_indices]
    risk_arr = np.array(risks)[sorted_indices]

    # AURC: numerical trapezoidal integration of risk over coverage (NumPy 2.0+ compatible)
    if hasattr(np, 'trapezoid'):
        aurc = float(np.trapezoid(risk_arr, cov_arr))
    else:
        aurc = float(np.trapz(risk_arr, cov_arr))

    return {
        "coverages": coverages,
        "risks": risks,
        "selective_accuracies": selective_accs,
        "thresholds": list(thresholds),
        "aurc": aurc
    }

def compute_abstention_metrics(is_abstained_list, should_abstain_list):
    """
    Evaluates reject-option fidelity against true ambiguous ground truth.
    """
    is_abstained = np.array(is_abstained_list, dtype=bool)
    should_abstain = np.array(should_abstain_list, dtype=bool)

    tp = np.sum(is_abstained & should_abstain)
    fp = np.sum(is_abstained & ~should_abstain)
    fn = np.sum(~is_abstained & should_abstain)
    tn = np.sum(~is_abstained & ~should_abstain)

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)
    }
