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
from models import DetectionModel
from data import CustomDataset
from utils import train_one_epoch
from utils import *
from utils import DetectionMetrics

# Load the dataset
train_dataset = CustomDataset(image_dir=IMAGE_DIR, annotations_file=ANNOTATIONS_FILE, transform=T.Compose([
    T.Resize((320, 320)),
    T.ToTensor(),
]))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Initialize the model, optimizer, and loss function
model = DetectionModel(num_classes=NUM_CLASSES).to(DEVICE)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
def custom_criterion(cls_preds, box_preds, targets):
    labels = [target['labels'] for target in targets]
    boxes = [target['boxes'] for target in targets]
    
    cls_loss = torch.nn.functional.cross_entropy(cls_preds, torch.stack(labels))
    box_loss = torch.nn.functional.mse_loss(box_preds, torch.stack(boxes))

    return cls_loss + box_loss

criterion = custom_criterion

# Initialize metrics
metrics = DetectionMetrics()

# Training loop
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    epoch_loss = train_one_epoch(model, optimizer, train_loader, DEVICE, criterion)

    # Optionally, calculate and log training metrics
    print(f"Training Loss: {epoch_loss:.4f}")

    # Save model checkpoint periodically
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pth")

print("Training complete!")