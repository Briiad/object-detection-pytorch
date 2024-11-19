import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import torch
if torch.cuda.is_available():
    torch.cuda.init()
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

import torchvision.transforms as T
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from data import CustomDataset
from utils import train_one_epoch
from utils import *
from utils import DetectionMetrics
from utils import *
from models import CustomModel
from utils import train_one_epoch, evaluate
from utils.custom_utils import Averager

train_transforms = T.Compose([
    T.Resize((640, 640)),
    T.ToTensor(),
    T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)  # Ensure RGB
])

train_dataset = CustomDataset(
    IMAGE_DIR, 
    ANNOTATIONS_FILE, 
    transform=train_transforms
)

val_dataset = CustomDataset(
    VAL_IMAGE_DIR, 
    VAL_ANNOTATIONS_FILE, 
    transform=train_transforms
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Initialize model, optimizer, and scheduler
model = CustomModel(num_classes=NUM_CLASSES).to(DEVICE)
optimizer = Adam(
  model.parameters(),
  lr=LEARNING_RATE,
  weight_decay=1e-5
)
scheduler = MultiStepLR(optimizer, milestones=[5, 8], gamma=0.1)

# Training loop
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    train_loss = train_one_epoch(model, optimizer, train_loader, DEVICE, criterion=None, scheduler=scheduler)
    print(f"Training Loss: {train_loss:.4f}")
  
    if (epoch + 1) % 2 == 0:
        results = evaluate(model, val_loader, DEVICE, criterion=None)
        print(f"Validation Results: {results}")

    if scheduler:
        scheduler.step()

    # Save checkpoint
    torch.save(model.state_dict(), os.path.join(VAL_CHECKPOINT_FILE, f"model_epoch_{epoch+1}.pth"))