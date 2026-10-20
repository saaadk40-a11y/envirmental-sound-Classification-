"""Environmental Sound Classification with a CNN (ESC-50).

Usage:
    pip install torch librosa soundfile scikit-learn pandas matplotlib tqdm
    git clone https://github.com/karolpiczak/ESC-50.git
    python esc50_sound_classification.py
"""



# ============================================================
# 1. Setup: install libraries and download the dataset
# ============================================================


# ============================================================
# 2. Imports and settings
# ============================================================

import os, time, random
import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm

# Reproducibility
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---- Settings (you can change these) ----
SR = 22050          # sample rate we resample all audio to
DURATION = 5        # every ESC-50 clip is 5 seconds
N_MELS = 128        # number of mel frequency bins (height of the spectrogram)
N_FFT = 1024        # FFT window size
HOP = 512           # hop length -> 5 s * 22050 / 512 = 216 time frames
BATCH_SIZE = 32
EPOCHS = 40
LR = 1e-3


# ============================================================
# 3. Load the dataset metadata
# ============================================================

df = pd.read_csv("ESC-50/meta/esc50.csv")
print("Rows, columns:", df.shape)
print(df.head())

# class names ordered by their numeric label (0..49)
classes = df.drop_duplicates("target").sort_values("target")["category"].tolist()
print(len(classes), "classes:")
print(classes)

print("\nClips per fold:")
print(df["fold"].value_counts().sort_index())


# ============================================================
# 4. Feature extraction: audio -> log-mel spectrogram
# ============================================================

FEAT_FILE = "mel_features.npy"
n_samples = SR * DURATION


def clip_to_logmel(path):
    y, _ = librosa.load(path, sr=SR)
    if len(y) < n_samples:
        y = np.pad(y, (0, n_samples - len(y)))
    else:
        y = y[:n_samples]
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    return librosa.power_to_db(mel, ref=1.0, top_db=80.0).astype(np.float32)


if os.path.exists(FEAT_FILE):
    X = np.load(FEAT_FILE)
    print("Loaded cached features from disk.")
else:
    X = np.stack([clip_to_logmel(os.path.join("ESC-50/audio", f)) for f in tqdm(df["filename"])])
    np.save(FEAT_FILE, X)

y_all = df["target"].values
print("Feature array shape:", X.shape, "(clips, mel bins, time frames)")
print("Labels shape:", y_all.shape)

fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
for ax, i in zip(axes, [0, 500, 1500]):
    ax.imshow(X[i], origin="lower", aspect="auto", cmap="magma")
    ax.set_title(f"Class: {df.loc[i, 'category']}")
    ax.set_xlabel("time frames")
    ax.set_ylabel("mel bins")
plt.tight_layout()
plt.savefig("example_spectrograms.png", dpi=150)
plt.close()


# ============================================================
# 5. Train / validation / test split
# ============================================================

train_idx = df.index[df["fold"].isin([1, 2, 3])].values
val_idx = df.index[df["fold"] == 4].values
test_idx = df.index[df["fold"] == 5].values
print("train / val / test sizes:", len(train_idx), len(val_idx), len(test_idx))

mean, std = X[train_idx].mean(), X[train_idx].std()
print(f"Normalisation: mean={mean:.2f}, std={std:.2f}")
Xn = (X - mean) / std


# ============================================================
# 6. Dataset class with data augmentation
# ============================================================

class ESCDataset(Dataset):
    def __init__(self, features, labels, augment=False):
        self.X = torch.tensor(features, dtype=torch.float32).unsqueeze(1)  # (N, 1, mel, time)
        self.y = torch.tensor(labels, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i].clone()
        if self.augment:
            # 1) random circular time shift
            x = torch.roll(x, shifts=int(np.random.randint(-30, 31)), dims=2)
            # 2) frequency mask (hide up to 19 mel bins)
            f = int(np.random.randint(0, 20)); f0 = int(np.random.randint(0, x.shape[1] - f))
            x[:, f0:f0 + f, :] = 0
            # 3) time mask (hide up to 29 frames)
            t = int(np.random.randint(0, 30)); t0 = int(np.random.randint(0, x.shape[2] - t))
            x[:, :, t0:t0 + t] = 0
            # 4) light noise
            x = x + 0.05 * torch.randn_like(x)
        return x, self.y[i]


train_ds = ESCDataset(Xn[train_idx], y_all[train_idx], augment=True)
val_ds = ESCDataset(Xn[val_idx], y_all[val_idx])
test_ds = ESCDataset(Xn[test_idx], y_all[test_idx])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

xb, yb = next(iter(train_loader))
print("One batch:", xb.shape, yb.shape)


# ============================================================
# 7. The CNN model
# ============================================================

class SoundCNN(nn.Module):
    def __init__(self, n_classes=50, dropout=0.4):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128), block(128, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(256, n_classes))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


model = SoundCNN(n_classes=len(classes)).to(device)
print(model)
print("\nTrainable parameters:", f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


# ============================================================
# 8. Training
# ============================================================

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


def run_epoch(loader, train):
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            n += len(yb)
    return total_loss / n, correct / n


history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val, best_epoch = 0.0, 0
start = time.time()

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    va_loss, va_acc = run_epoch(val_loader, train=False)
    scheduler.step()

    history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
    history["val_loss"].append(va_loss); history["val_acc"].append(va_acc)

    marker = ""
    if va_acc > best_val:
        best_val, best_epoch = va_acc, epoch
        torch.save(model.state_dict(), "best_model.pt")
        marker = "  <-- best so far, saved"

    print(f"Epoch {epoch:02d}/{EPOCHS} | train loss {tr_loss:.3f} acc {tr_acc:.3f} "
          f"| val loss {va_loss:.3f} acc {va_acc:.3f}{marker}")

print(f"\nDone in {(time.time() - start) / 60:.1f} min. Best val accuracy {best_val:.3f} at epoch {best_epoch}.")


# ============================================================
# 9. Training curves
# ============================================================

epochs_range = range(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(epochs_range, history["train_loss"], label="train")
axes[0].plot(epochs_range, history["val_loss"], label="validation")
axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy"); axes[0].legend()

axes[1].plot(epochs_range, history["train_acc"], label="train")
axes[1].plot(epochs_range, history["val_acc"], label="validation")
axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy"); axes[1].legend()

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.close()


# ============================================================
# 10. Final evaluation on the held-out test fold
# ============================================================

model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval()

all_preds, all_true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        out = model(xb.to(device))
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_true.extend(yb.numpy())

all_preds, all_true = np.array(all_preds), np.array(all_true)
test_acc = (all_preds == all_true).mean()
print("=" * 40)
print(f"TEST ACCURACY: {test_acc * 100:.2f}%   ({(all_preds == all_true).sum()} / {len(all_true)} correct)")
print("=" * 40)
print("(Chance level is 2%. Human accuracy on ESC-50 is about 81%.)")


# ============================================================
# 11. Confusion matrix
# ============================================================

cm = confusion_matrix(all_true, all_preds, labels=list(range(len(classes))))
cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

fig, ax = plt.subplots(figsize=(16, 14))
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
ax.set_xticklabels(classes, rotation=90, fontsize=7)
ax.set_yticklabels(classes, fontsize=7)
ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
ax.set_title(f"Normalised confusion matrix (test accuracy {test_acc * 100:.1f}%)")
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()


# ============================================================
# 12. Which classes work well and which do not?
# ============================================================

per_class_acc = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
order = np.argsort(per_class_acc)

print("5 WORST classes:")
for i in order[:5]:
    print(f"  {classes[i]:<18} {per_class_acc[i] * 100:5.1f}%")
print("\n5 BEST classes:")
for i in order[::-1][:5]:
    print(f"  {classes[i]:<18} {per_class_acc[i] * 100:5.1f}%")

# most common mistakes
cm_off = cm.copy()
np.fill_diagonal(cm_off, 0)
pairs = [(cm_off[i, j], classes[i], classes[j])
         for i in range(len(classes)) for j in range(len(classes)) if cm_off[i, j] > 0]
pairs.sort(reverse=True)
print("\nMost common confusions (true class -> predicted class):")
for count, true_c, pred_c in pairs[:8]:
    print(f"  {true_c:<16} -> {pred_c:<16} {count} times")

plt.figure(figsize=(14, 4))
plt.bar(range(len(classes)), per_class_acc[np.argsort(per_class_acc)])
plt.xticks(range(len(classes)), [classes[i] for i in np.argsort(per_class_acc)], rotation=90, fontsize=7)
plt.ylabel("accuracy"); plt.title("Per-class test accuracy (sorted)")
plt.tight_layout()
plt.savefig("per_class_accuracy.png", dpi=150)
plt.close()


# ============================================================
# 13. Demo: listen to a test clip and see the model's prediction
# ============================================================


# ============================================================
# 14. Summary of results
# ============================================================

print("Dataset:           ESC-50 (2000 clips, 50 classes)")
print(f"Features:          log-mel spectrogram, {N_MELS} mel bins x {X.shape[2]} frames")
print(f"Split:             train {len(train_idx)} / val {len(val_idx)} / test {len(test_idx)} (folds 1-3 / 4 / 5)")
print(f"Model parameters:  {sum(p.numel() for p in model.parameters()):,}")
print(f"Epochs:            {EPOCHS}, batch size {BATCH_SIZE}, Adam lr={LR} + cosine schedule")
print(f"Best val accuracy: {best_val * 100:.2f}% (epoch {best_epoch})")
print(f"TEST accuracy:     {test_acc * 100:.2f}%")
print("\nSaved files (see the folder icon on the left): example_spectrograms.png, training_curves.png,")
print("confusion_matrix.png, per_class_accuracy.png, best_model.pt")


