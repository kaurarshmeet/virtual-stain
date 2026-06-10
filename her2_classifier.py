"""
HER2 Downstream Classification Validation
Trains a ResNet18 on real BCI IHC images, evaluates on:
  1. Real BCI test IHC
  2. BCI-model generated BCI test IHC  (in-distribution)
  3. MIST-model generated BCI test IHC (cross-dataset)
Validates that BIM rankings agree with classification accuracy rankings.
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, classification_report

SCRATCH = '/scratch/users/arshmeet/virtual_stain'
TRAIN_DIR = f'{SCRATCH}/raw_data/BCI/BCI_dataset/IHC/train'
TEST_REAL  = f'{SCRATCH}/raw_data/BCI/BCI_dataset/IHC/test'
TEST_BCI   = f'{SCRATCH}/results/bci_her2_lambda_linear/test_latest/images/fake_B'
TEST_MIST  = f'{SCRATCH}/results/mist_her2_lambda_linear/test_latest/images/fake_B'

LABEL_MAP = {'0': 0, '1+': 1, '2+': 2, '3+': 3}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}", flush=True)

def parse_label(filename):
    stem = os.path.splitext(filename)[0]
    score = stem.split('_')[-1]
    return LABEL_MAP[score]

class BCIDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.files = [f for f in sorted(os.listdir(img_dir))
                      if f.endswith(('.png', '.jpg'))]
        self.labels = [parse_label(f) for f in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_dir, self.files[idx])).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# --- Train ---
print("Loading training data...", flush=True)
train_ds = BCIDataset(TRAIN_DIR, train_tf)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
print(f"Train: {len(train_ds)} images", flush=True)

model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 4)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

for epoch in range(15):
    model.train()
    losses = []
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    scheduler.step()
    print(f"Epoch {epoch+1}/15  loss={np.mean(losses):.4f}", flush=True)

# --- Evaluate ---
def evaluate(name, img_dir, fid=None, psnr=None, ssim=None, phv_l1=None, phv_l2=None, phv_l3=None, phv_l4=None, bim_ndc=None, bim_msic=None, bim_scorr=None):
    ds = BCIDataset(img_dir, test_tf)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            preds = model(imgs).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='weighted')
    print(f"\n=== {name} ===", flush=True)
    print(f"Accuracy: {acc:.4f}  F1: {f1:.4f}", flush=True)
    if fid is not None:
        print(f"FID: {fid}  PSNR: {psnr:.4f}  SSIM: {ssim:.4f}", flush=True)
    if phv_l4 is not None:
        print(f"PHV — L1: {phv_l1}  L2: {phv_l2}  L3: {phv_l3}  L4: {phv_l4}", flush=True)
    if bim_ndc is not None:
        print(f"BIM — NDC: {bim_ndc}  MSIC: {bim_msic}  Spearman: {bim_scorr}", flush=True)
    print(classification_report(all_labels, all_preds,
          target_names=['HER2-0','HER2-1+','HER2-2+','HER2-3+'], digits=3), flush=True)
    return acc, f1

evaluate("Real BCI test (upper bound)",  TEST_REAL)
evaluate("BCI model → BCI (in-dist.)",   TEST_BCI,
         fid=54.28, psnr=17.8650, ssim=0.4923,
         phv_l1=0.4995, phv_l2=0.3802, phv_l3=0.2319, phv_l4=0.7228,
         bim_ndc=0.6622, bim_msic=0.5759, bim_scorr=0.1537)
evaluate("MIST model → BCI (cross-ds.)", TEST_MIST,
         fid=119.17, psnr=14.9454, ssim=0.3512,
         phv_l1=0.6265, phv_l2=0.5439, phv_l3=0.3627, phv_l4=0.8495,
         bim_ndc=0.6214, bim_msic=0.3622, bim_scorr=0.1423)

print("\n=== SUMMARY ===", flush=True)
print("If BIM is valid: Real > BCI-model > MIST-model in accuracy", flush=True)


"""
This code was generated by Claude. 

Prompt 1: 
 Please generate a file that performs downstream classification validation, as follows:

  The model (ResNet) should predict the BCI classification of each patch (0, 1, 2, 3) based on content of
  the IHC patch. The labels can be extracted from the patchnames of the IHC patches themselves.

  (1) Train a ResNet on real BCI IHC images. Real IHC images sourced from
  /raw_data/BCI/BCI_dataset/IHC/train.

  (2) Evaluate the model on:
  1. real IHC at /raw_data/BCI/BCI_dataset/IHC/test
  2. IHC produced by the BCI-trained model at /results/bci_her2_lambda_linear/test_latest/images/fake_B
  3. IHC produced by the MIST-trained model at /results/mist_her2_lambda_linear/test_latest/images/fake_B

  Please measure: F1, Accuracy. As well as any other typical classification metrics you wish to measure.
  For the BCI-trained model, 
  For the MIST-trained model, 
  Please list these metrics alongside the classification results.

Prompt 2: 
Can you change the code so that we also report FID, PSNR, SSIM and layer-wise PHV along with everything. 
"""