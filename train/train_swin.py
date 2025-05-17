
import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import torchvision.transforms as T
import timm
import torch.onnx

DATASET_PATH = "./dataset"
BATCH_SIZE = 8
EPOCHS = 8
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------- Dataset ----------------
class BScanDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        arr = np.load(path).astype(np.float32)
        arr = np.clip(arr, 0.0, 1.0)
        arr = np.stack([arr]*3, axis=0)  # (3, 256, 512)
        if self.transform:
            arr = self.transform(torch.from_numpy(arr))
        return arr, label

def load_dataset(root):
    samples = []
    for label, cls in enumerate(['no_mine', 'with_mine']):
        cls_path = os.path.join(root, cls)
        for fname in os.listdir(cls_path):
            if fname.endswith('.npy'):
                samples.append((os.path.join(cls_path, fname), label))
    return train_test_split(samples, test_size=0.2, random_state=42, shuffle=True)

# -------------- Transform ----------------
transform = T.Compose([
    T.Resize((224, 224)),  # resnet expects 224×224
])

# --------------- Model -------------------
model = timm.create_model('resnet18', pretrained=False, num_classes=2)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -------------- DataLoaders --------------
train_samples, val_samples = load_dataset(DATASET_PATH)
train_loader = DataLoader(BScanDataset(train_samples, transform), batch_size=BATCH_SIZE, num_workers=2, shuffle=True)
val_loader   = DataLoader(BScanDataset(val_samples,   transform), batch_size=BATCH_SIZE, num_workers=2, shuffle=False)

# --------------- Training ----------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    correct_train, total_train = 0, 0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        inputs = inputs.to(DEVICE)
        labels = labels.long().to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, pred = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (pred == labels).sum().item()

    avg_train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct_train / total_train

    # -------- Validation --------
    model.eval()
    val_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.long().to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct / total
    print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

# ------------- Export ONNX ---------------
dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
torch.onnx.export(
    model, dummy, "resnet_mine_detector.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=12
)
print("✅ Saved model to resnet_mine_detector.onnx")