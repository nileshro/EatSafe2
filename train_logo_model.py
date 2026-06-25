import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# ==================================================
# CONFIG
# ==================================================

DATASET_PATH = r"logos_db\logo\logo_datasets"

MODEL_PATH = r"models\logo_classifier.pth"
CHECKPOINT_PATH = r"models\checkpoint_latest.pth"
BEST_MODEL_PATH = r"models\best_model.pth"

BATCH_SIZE = 32
EPOCHS = 10
IMAGE_SIZE = 224
LEARNING_RATE = 0.001

EARLY_STOPPING_PATIENCE = 5

# ==================================================
# TRANSFORMS
# ==================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==================================================
# DATASET
# ==================================================

dataset = datasets.ImageFolder(
    DATASET_PATH,
    transform=transform
)

num_classes = len(dataset.classes)

print(f"\nClasses Found: {num_classes}")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE
)

# ==================================================
# MODEL
# ==================================================

model = models.mobilenet_v2(weights="DEFAULT")

# Freeze all layers

for param in model.parameters():
    param.requires_grad = False

# Unfreeze last feature block

for param in model.features[-1].parameters():
    param.requires_grad = True

# Replace classifier

model.classifier[1] = nn.Linear(
    model.last_channel,
    num_classes
)

# ==================================================
# DEVICE
# ==================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)

model = model.to(device)

# ==================================================
# LOSS + OPTIMIZER
# ==================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE
)

# ==================================================
# RESUME TRAINING
# ==================================================

start_epoch = 0
best_accuracy = 0
patience_counter = 0

if os.path.exists(CHECKPOINT_PATH):

    print("\nCheckpoint Found")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_epoch = checkpoint["epoch"] + 1

    best_accuracy = checkpoint["best_accuracy"]

    print(
        f"Resuming From Epoch {start_epoch}"
    )

# ==================================================
# TRAINING LOOP
# ==================================================

for epoch in range(start_epoch, EPOCHS):

    model.train()

    running_loss = 0

    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        train_total += labels.size(0)

        train_correct += (
            predicted == labels
        ).sum().item()

    train_accuracy = (
        100 * train_correct / train_total
    )

    # ======================================
    # VALIDATION
    # ======================================

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    val_accuracy = (
        100 * correct / total
    )

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {running_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Val Acc: {val_accuracy:.2f}%"
    )

    # ======================================
    # SAVE CHECKPOINT
    # ======================================

    os.makedirs(
        "models",
        exist_ok=True
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_accuracy": best_accuracy
        },
        CHECKPOINT_PATH
    )

    # ======================================
    # SAVE BEST MODEL
    # ======================================

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        patience_counter = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classes": dataset.classes,
                "num_classes": num_classes,
                "best_accuracy": best_accuracy
            },
            BEST_MODEL_PATH
        )

        print(
            f"New Best Model Saved "
            f"({best_accuracy:.2f}%)"
        )

    else:

        patience_counter += 1

    # ======================================
    # EARLY STOPPING
    # ======================================

    if patience_counter >= EARLY_STOPPING_PATIENCE:

        print(
            "\nEarly Stopping Triggered"
        )

        break

# ==================================================
# FINAL MODEL SAVE
# ==================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "classes": dataset.classes,
        "num_classes": num_classes,
        "best_accuracy": best_accuracy
    },
    MODEL_PATH
)

print("\nTraining Completed")
print("Best Validation Accuracy:", best_accuracy)

print("\nSaved Files:")
print(MODEL_PATH)
print(BEST_MODEL_PATH)
print(CHECKPOINT_PATH)









# """
# EatSafe Logo Classifier Training Script
# MobileNetV2 fine-tuned on food brand logos

# Dataset structure required:
#     logos_db/logo/logo_datasets/
#         Parle/
#             img1.jpg, img2.jpg, ...
#         Saffola/
#             img1.jpg, ...
#         Maggi/
#             img1.jpg, ...
#         <BrandName>/
#             ...

# Minimum 20-30 images per brand recommended.
# """

# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms, models
# from torch.utils.data import DataLoader, random_split

# # ── CONFIG ────────────────────────────────────────────────────────────────────
# DATASET_PATH    = r"logos_db\logo\logo_datasets"
# MODEL_PATH      = r"models\logo_classifier.pth"
# CHECKPOINT_PATH = r"models\checkpoint_latest.pth"
# BEST_MODEL_PATH = r"models\best_model.pth"

# BATCH_SIZE               = 32
# EPOCHS                   = 15
# IMAGE_SIZE               = 224
# LEARNING_RATE            = 0.001
# EARLY_STOPPING_PATIENCE  = 5

# # ── TRANSFORMS (with augmentation for training) ───────────────────────────────
# train_transform = transforms.Compose([
#     transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
#     transforms.RandomCrop(IMAGE_SIZE),
#     transforms.RandomHorizontalFlip(),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# val_transform = transforms.Compose([
#     transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # ── DATASET ───────────────────────────────────────────────────────────────────
# full_dataset = datasets.ImageFolder(DATASET_PATH)
# num_classes  = len(full_dataset.classes)

# print(f"\nClasses Found: {num_classes}")
# print("Brands:", full_dataset.classes)

# # Apply different transforms to train vs val
# train_size = int(0.8 * len(full_dataset))
# val_size   = len(full_dataset) - train_size

# train_dataset_raw, val_dataset_raw = random_split(
#     full_dataset, [train_size, val_size]
# )

# # Wrap with transforms
# class TransformDataset(torch.utils.data.Dataset):
#     def __init__(self, dataset, transform):
#         self.dataset   = dataset
#         self.transform = transform
#     def __len__(self):
#         return len(self.dataset)
#     def __getitem__(self, idx):
#         img, label = self.dataset[idx]
#         return self.transform(img), label

# # Need PIL images from ImageFolder
# full_dataset_pil = datasets.ImageFolder(DATASET_PATH)  # no transform
# train_ids = train_dataset_raw.indices
# val_ids   = val_dataset_raw.indices

# from torch.utils.data import Subset
# train_subset = Subset(full_dataset_pil, train_ids)
# val_subset   = Subset(full_dataset_pil, val_ids)

# train_set = TransformDataset(train_subset, train_transform)
# val_set   = TransformDataset(val_subset, val_transform)

# train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
# val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# print(f"Train: {len(train_set)} | Val: {len(val_set)}")

# # ── MODEL ─────────────────────────────────────────────────────────────────────
# model = models.mobilenet_v2(weights="DEFAULT")

# # Freeze all layers first
# for param in model.parameters():
#     param.requires_grad = False

# # Unfreeze last 2 feature blocks for better fine-tuning
# for param in model.features[-1].parameters():
#     param.requires_grad = True
# for param in model.features[-2].parameters():
#     param.requires_grad = True

# # Replace classifier
# model.classifier[1] = nn.Linear(model.last_channel, num_classes)

# # ── DEVICE ────────────────────────────────────────────────────────────────────
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"\nUsing Device: {device}")
# model = model.to(device)

# # ── LOSS + OPTIMIZER + SCHEDULER ─────────────────────────────────────────────
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(
#     filter(lambda p: p.requires_grad, model.parameters()),
#     lr=LEARNING_RATE
# )
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode="max", patience=2, factor=0.5, verbose=True
# )

# # ── RESUME FROM CHECKPOINT ────────────────────────────────────────────────────
# start_epoch    = 0
# best_accuracy  = 0
# patience_counter = 0

# if os.path.exists(CHECKPOINT_PATH):
#     print("\nCheckpoint Found — Resuming Training")
#     ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
#     model.load_state_dict(ckpt["model_state_dict"])
#     optimizer.load_state_dict(ckpt["optimizer_state_dict"])
#     start_epoch   = ckpt["epoch"] + 1
#     best_accuracy = ckpt["best_accuracy"]
#     print(f"Resuming from Epoch {start_epoch} | Best Acc so far: {best_accuracy:.2f}%")

# # ── TRAINING LOOP ─────────────────────────────────────────────────────────────
# os.makedirs("models", exist_ok=True)

# for epoch in range(start_epoch, EPOCHS):

#     model.train()
#     running_loss  = 0
#     train_correct = 0
#     train_total   = 0

#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss    = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss  += loss.item()
#         _, predicted   = torch.max(outputs, 1)
#         train_total   += labels.size(0)
#         train_correct += (predicted == labels).sum().item()

#     train_acc = 100 * train_correct / train_total

#     # Validation
#     model.eval()
#     correct = 0
#     total   = 0
#     with torch.no_grad():
#         for images, labels in val_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs, 1)
#             total   += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     val_acc = 100 * correct / total
#     scheduler.step(val_acc)

#     print(
#         f"Epoch {epoch+1}/{EPOCHS} | "
#         f"Loss: {running_loss:.4f} | "
#         f"Train: {train_acc:.2f}% | "
#         f"Val: {val_acc:.2f}%"
#     )

#     # Save checkpoint
#     torch.save({
#         "epoch": epoch,
#         "model_state_dict": model.state_dict(),
#         "optimizer_state_dict": optimizer.state_dict(),
#         "best_accuracy": best_accuracy
#     }, CHECKPOINT_PATH)

#     # Save best model
#     if val_acc > best_accuracy:
#         best_accuracy    = val_acc
#         patience_counter = 0
#         torch.save({
#             "model_state_dict": model.state_dict(),
#             "classes":          full_dataset.classes,
#             "num_classes":      num_classes,
#             "best_accuracy":    best_accuracy
#         }, BEST_MODEL_PATH)
#         print(f"  ✅ New Best Model Saved ({best_accuracy:.2f}%)")
#     else:
#         patience_counter += 1

#     # Early stopping
#     if patience_counter >= EARLY_STOPPING_PATIENCE:
#         print("\nEarly Stopping Triggered.")
#         break

# # Final save
# torch.save({
#     "model_state_dict": model.state_dict(),
#     "classes":          full_dataset.classes,
#     "num_classes":      num_classes,
#     "best_accuracy":    best_accuracy
# }, MODEL_PATH)

# print(f"\n✅ Training Complete | Best Val Accuracy: {best_accuracy:.2f}%")
# print(f"Saved: {BEST_MODEL_PATH}")