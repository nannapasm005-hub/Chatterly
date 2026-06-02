"""
Chatterly v2 — Evaluation Script
Dataset  : Wisesight Sentiment (local .txt files)
Metrics  : Accuracy, Precision, Recall, F1-score, Confusion Matrix
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import numpy as np
from pythainlp.tokenize import word_tokenize
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import random
from tqdm import tqdm
import os

# ── Config ─────────────────────────────────────────────────────────────────
SAMPLE_SIZE = 200  # None = ทดสอบทั้งหมด
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABELS = ["pos", "neu", "neg"]
id2label = {0: "pos", 1: "neu", 2: "neg"}

# ── ไฟล์ที่โหลดมา ───────────────────────────────────────────────────────────
DATA_FILES = {
    "pos": "wisesight_pos.txt",
    "neu": "wisesight_neu.txt",
    "neg": "wisesight_neg.txt",
}

# ── Load Model ──────────────────────────────────────────────────────────────
print("=" * 60)
print("Chatterly v2 — Evaluation")
print("=" * 60)
print(f"\n📦 Loading WangchanBERTa... (device: {DEVICE})")

tokenizer = AutoTokenizer.from_pretrained("Pongsathorn/wangchanberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained(
    "Pongsathorn/wangchanberta-base-sentiment"
).to(DEVICE)
model.eval()
print("✅ Model loaded!\n")

# ── Load Wisesight from local .txt files ────────────────────────────────────
print("📂 Loading Wisesight from local files...")

raw = []
for label, filepath in DATA_FILES.items():
    if not os.path.exists(filepath):
        print(f"❌ ไม่พบไฟล์ {filepath}")
        print("   กรุณารันก่อน:")
        print(f"   curl -L https://raw.githubusercontent.com/PyThaiNLP/wisesight-sentiment/master/{filepath.replace('wisesight_', '')} -o {filepath}")
        exit(1)

    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    raw.extend([{"text": line, "label": label} for line in lines])
    print(f"  ✅ {filepath}: {len(lines)} samples")

print(f"\n✅ Total: {len(raw)} samples loaded\n")

# ── Sample ──────────────────────────────────────────────────────────────────
random.seed(42)
if SAMPLE_SIZE and SAMPLE_SIZE < len(raw):
    # sample แบบ balanced แต่ละ class
    per_class = SAMPLE_SIZE // len(LABELS)
    sampled = []
    for label in LABELS:
        items = [x for x in raw if x["label"] == label]
        sampled.extend(random.sample(items, min(per_class, len(items))))
    filtered = sampled
else:
    filtered = raw

random.shuffle(filtered)
print(f"🔢 Evaluating on {len(filtered)} samples\n")

# ── Predict ─────────────────────────────────────────────────────────────────
def predict(text: str):
    text = text[:512]  # truncate ก่อน tokenize
    segmented = word_tokenize(text, engine="longest")
    preprocessed = " ".join(segmented)
    inputs = tokenizer(
        preprocessed,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1)
    predicted_idx = torch.argmax(probs, dim=-1).item()
    confidence = float(probs[0][predicted_idx].cpu())
    return id2label[predicted_idx], confidence


y_true, y_pred, confidences = [], [], []

print("🔄 Running predictions...")
for item in tqdm(filtered, desc="Predicting"):
    pred_label, conf = predict(item["text"])
    y_true.append(item["label"])
    y_pred.append(pred_label)
    confidences.append(conf)

# ── Metrics ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("📊 EVALUATION RESULTS")
print("=" * 60)

acc = accuracy_score(y_true, y_pred)
print(f"\n🎯 Accuracy       : {acc * 100:.2f}%")
print(f"📈 Avg Confidence : {np.mean(confidences) * 100:.2f}%")

print("\n── Classification Report ──────────────────────────────────")
print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))

precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, labels=LABELS, average=None, zero_division=0
)

print("── Per-class Metrics ──────────────────────────────────────")
for i, label in enumerate(LABELS):
    print(f"  {label:4s} | P: {precision[i]:.3f} | R: {recall[i]:.3f} | F1: {f1[i]:.3f} | Support: {support[i]}")

macro_f1 = float(np.mean(f1))
print(f"\n  Macro F1 : {macro_f1:.3f}")

# ── Plots ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Chatterly v2 — Evaluation Results\n(WangchanBERTa on Wisesight Sentiment)", fontsize=13, fontweight="bold")

cm = confusion_matrix(y_true, y_pred, labels=LABELS)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=LABELS, yticklabels=LABELS, ax=axes[0], linewidths=0.5)
axes[0].set_title("Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

colors = ["#2ec4b6", "#f5c842", "#e8744a"]
bars = axes[1].bar(LABELS, f1, color=colors, alpha=0.85, edgecolor="white", linewidth=1.5)
axes[1].set_title("F1-score per Class")
axes[1].set_ylabel("F1-score")
axes[1].set_ylim(0, 1.1)
axes[1].axhline(macro_f1, color="gray", linestyle="--", linewidth=1.2,
                label=f"Macro F1 = {macro_f1:.3f}")
axes[1].legend()
for bar, score in zip(bars, f1):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{score:.3f}", ha="center", fontweight="bold", fontsize=11)

plt.tight_layout()
plt.savefig("chatterly_evaluation.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n💾 Saved: chatterly_evaluation.png")

# ── JSON ─────────────────────────────────────────────────────────────────────
summary = {
    "dataset": "Wisesight Sentiment (local txt files)",
    "sample_size": len(filtered),
    "accuracy": round(acc * 100, 2),
    "avg_confidence": round(float(np.mean(confidences)) * 100, 2),
    "macro_f1": round(macro_f1, 4),
    "per_class": {
        label: {
            "precision": round(float(precision[i]), 4),
            "recall":    round(float(recall[i]), 4),
            "f1":        round(float(f1[i]), 4),
            "support":   int(support[i]),
        }
        for i, label in enumerate(LABELS)
    },
}
with open("chatterly_eval_results.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("💾 Saved: chatterly_eval_results.json")
print("\n✅ Evaluation complete!")
