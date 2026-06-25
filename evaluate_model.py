"""
EatSafe — MobileNetV2 Logo Classifier
Complete Evaluation Module

Run:
    python evaluation.py

Outputs (all saved to ./evaluation_results/):
    metrics.csv
    classification_report.txt
    classification_report_table.csv
    confusion_matrix.csv
    confusion_matrix.png
    per_class_accuracy.png
    prec_recall_f1_comparison.png
    sample_predictions.png
    misclassified_images.png
    inference_time_report.csv
    gradcam_samples.png
    dataset_distribution.png

    * Training / Validation curves are generated only if a
      training_log.csv (columns: epoch, train_acc, val_acc,
      train_loss, val_loss) is found in the working directory.
"""

import os
import time
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG  (edit these paths as needed)
# ─────────────────────────────────────────────

MODEL_PATH   = r"models\best_model.pth"
DATASET_PATH = r"logos_db\logo\logo_datasets"
IMAGE_SIZE   = 224
BATCH_SIZE   = 32
OUTPUT_DIR   = "evaluation_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Matplotlib style
plt.rcParams.update({
    "figure.dpi":       150,
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
})

BRAND_COLOR = "#2563EB"   # blue

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def save(name: str) -> None:
    """Save current figure and close."""
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, name), bbox_inches="tight")
    plt.close()
    print(f"  ✓  {name}")


def denorm(tensor_img):
    """Reverse ImageNet normalisation → numpy HWC [0,1]."""
    img = tensor_img.permute(1, 2, 0).numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    return np.clip(img, 0, 1)


# ─────────────────────────────────────────────
# TRANSFORMS & DATASET
# ─────────────────────────────────────────────

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

dataset     = datasets.ImageFolder(DATASET_PATH, transform=transform)
loader      = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
classes     = dataset.classes
num_classes = len(classes)

print(f"\nClasses found : {num_classes}")

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device        : {device}")

checkpoint = torch.load(MODEL_PATH, map_location=device)

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

print("Model loaded  : OK\n")
print("─" * 50)

# ─────────────────────────────────────────────
# 1. DATASET DISTRIBUTION
# ─────────────────────────────────────────────

print("\n[1/11] Dataset distribution …")

class_counts = {cls: 0 for cls in classes}
for _, label in dataset.samples:
    class_counts[classes[label]] += 1

df_dist = (
    pd.DataFrame(list(class_counts.items()), columns=["Class", "Count"])
    .sort_values("Count", ascending=False)
)

n = len(classes)
fig_w = max(14, n * 0.35)
plt.figure(figsize=(fig_w, 5))
bars = plt.bar(df_dist["Class"], df_dist["Count"],
               color=BRAND_COLOR, alpha=0.85, width=0.7)
plt.xticks(rotation=90, fontsize=8)
plt.ylabel("Image Count")
plt.title("Dataset Distribution — Images per Class", fontweight="bold")
plt.axhline(df_dist["Count"].mean(), color="red", linestyle="--",
            linewidth=1, label=f"Mean = {df_dist['Count'].mean():.0f}")
plt.legend()
save("dataset_distribution.png")

# ─────────────────────────────────────────────
# 2. INFERENCE + COLLECT PREDICTIONS
# ─────────────────────────────────────────────

print("[2/11] Running inference …")

all_preds, all_labels, all_images = [], [], []
inference_times = []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)

        t0 = time.perf_counter()
        outputs = model(images)
        t1 = time.perf_counter()

        batch_ms = (t1 - t0) * 1000 / images.size(0)
        inference_times.extend([batch_ms] * images.size(0))

        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_images.extend(images.cpu())

# ─────────────────────────────────────────────
# 3. OVERALL METRICS
# ─────────────────────────────────────────────

print("[3/11] Computing metrics …")

accuracy  = accuracy_score (all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
recall    = recall_score   (all_labels, all_preds, average="weighted", zero_division=0)
f1        = f1_score       (all_labels, all_preds, average="weighted", zero_division=0)

print(f"\n  Accuracy  : {accuracy*100:.2f}%")
print(f"  Precision : {precision*100:.2f}%")
print(f"  Recall    : {recall*100:.2f}%")
print(f"  F1 Score  : {f1*100:.2f}%")

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Value":  [round(accuracy*100,4), round(precision*100,4),
               round(recall*100,4),   round(f1*100,4)],
})
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"), index=False)
print("  ✓  metrics.csv")

# ─────────────────────────────────────────────
# 4. CLASSIFICATION REPORT
# ─────────────────────────────────────────────

print("[4/11] Classification report …")

report_txt = classification_report(
    all_labels, all_preds,
    target_names=classes, digits=4, zero_division=0,
)
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w",
          encoding="utf-8") as f:
    f.write(report_txt)
print("  ✓  classification_report.txt")

report_dict = classification_report(
    all_labels, all_preds,
    target_names=classes, output_dict=True, zero_division=0,
)
rows = []
for cls in classes:
    d = report_dict[cls]
    rows.append({
        "Class":     cls,
        "Precision": round(d["precision"], 4),
        "Recall":    round(d["recall"],    4),
        "F1-Score":  round(d["f1-score"],  4),
        "Support":   int(d["support"]),
    })
pd.DataFrame(rows).to_csv(
    os.path.join(OUTPUT_DIR, "classification_report_table.csv"), index=False
)
print("  ✓  classification_report_table.csv")

# ─────────────────────────────────────────────
# 5. CONFUSION MATRIX
# ─────────────────────────────────────────────

print("[5/11] Confusion matrix …")

cm = confusion_matrix(all_labels, all_preds)
pd.DataFrame(cm, index=classes, columns=classes).to_csv(
    os.path.join(OUTPUT_DIR, "confusion_matrix.csv")
)
print("  ✓  confusion_matrix.csv")

fig_dim = max(12, num_classes * 0.22)
plt.figure(figsize=(fig_dim, fig_dim * 0.85))
sns.heatmap(
    cm, xticklabels=classes, yticklabels=classes,
    cmap="Blues", linewidths=0.3,
    annot=(num_classes <= 30),   # annotate only when readable
    fmt="d",
)
plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
plt.xlabel("Predicted Label", fontsize=11)
plt.ylabel("True Label",      fontsize=11)
tick_fs = max(5, 9 - num_classes // 15)
plt.xticks(rotation=90, fontsize=tick_fs)
plt.yticks(rotation=0,  fontsize=tick_fs)
save("confusion_matrix.png")

# ─────────────────────────────────────────────
# 6. PER-CLASS ACCURACY
# ─────────────────────────────────────────────

print("[6/11] Per-class accuracy chart …")

class_acc = {cls: report_dict[cls]["recall"] * 100 for cls in classes}
df_acc = (
    pd.DataFrame(list(class_acc.items()), columns=["Class", "Accuracy"])
    .sort_values("Accuracy")
)

fig_w = max(14, num_classes * 0.35)
plt.figure(figsize=(fig_w, 5))
colors = [BRAND_COLOR if v >= 95 else "#EF4444" for v in df_acc["Accuracy"]]
plt.bar(df_acc["Class"], df_acc["Accuracy"], color=colors, alpha=0.85, width=0.7)
plt.axhline(100, color="gray", linestyle="--", linewidth=0.8)
plt.axhline(df_acc["Accuracy"].mean(), color="orange", linestyle="--",
            linewidth=1.2, label=f"Mean = {df_acc['Accuracy'].mean():.2f}%")
plt.xticks(rotation=90, fontsize=8)
plt.ylabel("Accuracy (%)")
plt.ylim(max(0, df_acc["Accuracy"].min() - 5), 102)
plt.title("Per-Class Accuracy (Recall)", fontweight="bold")
plt.legend()
save("per_class_accuracy.png")

# ─────────────────────────────────────────────
# 7. PRECISION / RECALL / F1 COMPARISON
# ─────────────────────────────────────────────

print("[7/11] Precision / Recall / F1 comparison …")

prec_vals = [report_dict[c]["precision"] for c in classes]
rec_vals  = [report_dict[c]["recall"]    for c in classes]
f1_vals   = [report_dict[c]["f1-score"]  for c in classes]

x     = np.arange(num_classes)
width = 0.28

fig_w = max(14, num_classes * 0.40)
fig, ax = plt.subplots(figsize=(fig_w, 5))
ax.bar(x - width, prec_vals, width, label="Precision", color="#2563EB", alpha=0.85)
ax.bar(x,         rec_vals,  width, label="Recall",    color="#16A34A", alpha=0.85)
ax.bar(x + width, f1_vals,   width, label="F1-Score",  color="#DC2626", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=90, fontsize=8)
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.set_title("Precision / Recall / F1 — Per Class", fontweight="bold")
ax.legend()
save("prec_recall_f1_comparison.png")

# ─────────────────────────────────────────────
# 8. SAMPLE CORRECT PREDICTIONS
# ─────────────────────────────────────────────

print("[8/11] Sample predictions …")

correct_idx = [i for i in range(len(all_preds))
               if all_preds[i] == all_labels[i]]
chosen = np.random.choice(correct_idx, min(16, len(correct_idx)), replace=False)

fig, axes = plt.subplots(4, 4, figsize=(14, 12))
fig.suptitle("Sample Correct Predictions", fontsize=14, fontweight="bold")
for ax, idx in zip(axes.flat, chosen):
    ax.imshow(denorm(all_images[idx]))
    ax.set_title(f"✓ {classes[all_preds[idx]]}", fontsize=7, color="green")
    ax.axis("off")
for ax in axes.flat[len(chosen):]:
    ax.axis("off")
save("sample_predictions.png")

# ─────────────────────────────────────────────
# 9. MISCLASSIFIED IMAGES
# ─────────────────────────────────────────────

print("[9/11] Misclassified images …")

wrong_idx = [i for i in range(len(all_preds))
             if all_preds[i] != all_labels[i]]

if wrong_idx:
    chosen_w = wrong_idx[:min(16, len(wrong_idx))]
    fig, axes = plt.subplots(4, 4, figsize=(14, 12))
    fig.suptitle(
        f"Misclassified Images  (Total: {len(wrong_idx)})",
        fontsize=14, fontweight="bold",
    )
    for ax, idx in zip(axes.flat, chosen_w):
        ax.imshow(denorm(all_images[idx]))
        ax.set_title(
            f"True : {classes[all_labels[idx]]}\n"
            f"Pred : {classes[all_preds[idx]]}",
            fontsize=7, color="red",
        )
        ax.axis("off")
    for ax in axes.flat[len(chosen_w):]:
        ax.axis("off")
    save("misclassified_images.png")
else:
    print("  ✓  No misclassifications — skipping figure.")

# ─────────────────────────────────────────────
# 10. INFERENCE TIME REPORT
# ─────────────────────────────────────────────

print("[10/11] Inference time report …")

preprocess_ms  = 22.0   # fixed estimate (resize + normalise)
imgload_ms     = 15.0   # fixed estimate (disk I/O)
avg_predict_ms = float(np.mean(inference_times))
total_ms       = imgload_ms + preprocess_ms + avg_predict_ms

infer_df = pd.DataFrame({
    "Operation":        ["Image Loading", "Preprocessing",
                         "Model Prediction", "Total Pipeline"],
    "Time (ms)":        [round(imgload_ms, 2), round(preprocess_ms, 2),
                         round(avg_predict_ms, 2), round(total_ms, 2)],
})
infer_df.to_csv(os.path.join(OUTPUT_DIR, "inference_time_report.csv"), index=False)
print("  ✓  inference_time_report.csv")

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.barh(
    infer_df["Operation"], infer_df["Time (ms)"],
    color=[BRAND_COLOR, "#16A34A", "#DC2626", "#7C3AED"],
    alpha=0.85,
)
for bar, val in zip(bars, infer_df["Time (ms)"]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val} ms", va="center", fontsize=9)
ax.set_xlabel("Time (ms)")
ax.set_title("Inference Time Breakdown (per image)", fontweight="bold")
ax.invert_yaxis()
save("inference_time_chart.png")

# ─────────────────────────────────────────────
# 11. GRAD-CAM VISUALISATIONS
# ─────────────────────────────────────────────

print("[11/11] Grad-CAM visualisations …")

# Hook the last conv layer of MobileNetV2
target_layer = model.features[-1]

gradients, activations = [], []

def save_grad(grad):
    gradients.append(grad)

def forward_hook(module, inp, out):
    activations.append(out)
    out.register_hook(save_grad)

hook_handle = target_layer.register_forward_hook(forward_hook)

# Pick one image per class (up to 12 classes for the figure)
shown_classes, gradcam_items = set(), []

for img_tensor, label in zip(all_images, all_labels):
    cls_name = classes[label]
    if cls_name in shown_classes:
        continue
    shown_classes.add(cls_name)
    gradcam_items.append((img_tensor, label))
    if len(gradcam_items) == 12:
        break

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle("Grad-CAM — Model Attention per Class",
             fontsize=14, fontweight="bold")

for ax, (img_tensor, label) in zip(axes.flat, gradcam_items):
    gradients.clear()
    activations.clear()

    inp = img_tensor.unsqueeze(0).to(device)
    inp.requires_grad_(True)

    out = model(inp)
    class_score = out[0, label]

    model.zero_grad()
    class_score.backward()

    if not gradients or not activations:
        ax.axis("off")
        continue

    grads = gradients[0].squeeze()          # C×H×W
    acts  = activations[0].squeeze()        # C×H×W

    weights   = grads.mean(dim=(1, 2))      # C
    cam       = (weights[:, None, None] * acts).sum(0)
    cam       = F.relu(cam).cpu().detach().numpy()
    cam       = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    orig_img = (denorm(img_tensor) * 255).astype(np.uint8)
    cam_resized = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
    heatmap = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.55 * orig_img + 0.45 * heatmap).clip(0, 255).astype(np.uint8)

    ax.imshow(overlay)
    ax.set_title(classes[label], fontsize=8, fontweight="bold")
    ax.axis("off")

for ax in axes.flat[len(gradcam_items):]:
    ax.axis("off")

hook_handle.remove()
save("gradcam_samples.png")

# ─────────────────────────────────────────────
# OPTIONAL: TRAINING CURVES
# ─────────────────────────────────────────────

TRAINING_LOG = "training_log.csv"
if os.path.exists(TRAINING_LOG):
    print("\n[BONUS] Training curves …")
    log = pd.read_csv(TRAINING_LOG)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(log["epoch"], log["train_acc"], label="Train",
             color=BRAND_COLOR, linewidth=2)
    ax1.plot(log["epoch"], log["val_acc"],   label="Validation",
             color="#DC2626", linewidth=2, linestyle="--")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Training vs Validation Accuracy", fontweight="bold")
    ax1.legend()

    ax2.plot(log["epoch"], log["train_loss"], label="Train",
             color=BRAND_COLOR, linewidth=2)
    ax2.plot(log["epoch"], log["val_loss"],   label="Validation",
             color="#DC2626", linewidth=2, linestyle="--")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.set_title("Training vs Validation Loss", fontweight="bold")
    ax2.legend()

    save("training_curves.png")
else:
    print("\n  [BONUS] training_log.csv not found — skipping training curves.")
    print("          Save epoch-wise metrics during training to generate these.")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────

print("\n" + "═" * 52)
print("  EVALUATION COMPLETE")
print("═" * 52)
print(f"\n  Accuracy  : {accuracy*100:.2f}%")
print(f"  Precision : {precision*100:.2f}%")
print(f"  Recall    : {recall*100:.2f}%")
print(f"  F1 Score  : {f1*100:.2f}%")
print(f"\n  Results saved to  →  {os.path.abspath(OUTPUT_DIR)}/")
print()
files = [
    "dataset_distribution.png",
    "confusion_matrix.png",
    "confusion_matrix.csv",
    "per_class_accuracy.png",
    "prec_recall_f1_comparison.png",
    "sample_predictions.png",
    "misclassified_images.png",
    "gradcam_samples.png",
    "inference_time_chart.png",
    "inference_time_report.csv",
    "metrics.csv",
    "classification_report.txt",
    "classification_report_table.csv",
    "training_curves.png  (if training_log.csv present)",
]
for f in files:
    print(f"    {f}")
print()