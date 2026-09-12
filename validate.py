# validate.py
"""Сравнивает предсказания entrypoint с gold в examples.json.

Возвращает ровно два поля, как в metrics.yaml:
    {"fitness": <macro-F1>, "is_valid": 0|1}
"""
from helper import LABELS, load_examples


def validate(predictions, split=None):
    rows = load_examples()
    if split:
        rows = [r for r in rows if r["split"] == split]

    pairs = []
    invalid = 0
    for r in rows:
        p = predictions.get(r["id"])
        if not isinstance(p, dict) or p.get("intent") not in LABELS:
            invalid += 1
            p = {"intent": "technical", "needs_clarification": False}
        if p.get("needs_clarification") and p["intent"] != "data_catalog":
            invalid += 1
        pairs.append((r["intent"], p["intent"]))

    cm = {t: {p: 0 for p in LABELS} for t in LABELS}
    for g, p in pairs:
        cm[g][p] += 1

    f1s = []
    for label in LABELS:
        tp = cm[label][label]
        fp = sum(cm[t][label] for t in LABELS if t != label)
        fn = sum(cm[label][p] for p in LABELS if p != label)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)

    return {
        "fitness": round(sum(f1s) / len(LABELS), 5),
        "is_valid": 1 if invalid == 0 else 0,
    }
