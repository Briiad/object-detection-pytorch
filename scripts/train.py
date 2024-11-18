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
    T.Resize((640, 640)),
    T.ToTensor(),
]))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Initialize the model, optimizer, and loss function
model = DetectionModel(num_classes=NUM_CLASSES).to(DEVICE)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

# Use detection-specific losses
# In train.py, update DetectionLoss class

# In train.py, update DetectionLoss class

# In train.py, update DetectionLoss class

class DetectionLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_loss = torch.nn.BCEWithLogitsLoss()
        self.box_loss = torch.nn.SmoothL1Loss()
        
    def forward(self, cls_preds, box_preds, targets):
        # Concatenate all target boxes and labels
        gt_boxes = []
        gt_labels = []
        batch_idx = []
        
        for idx, target in enumerate(targets):
            num_objs = len(target['boxes'])
            gt_boxes.append(target['boxes'])
            gt_labels.append(target['labels'])
            batch_idx.extend([idx] * num_objs)
            
        gt_boxes = torch.cat(gt_boxes)  # Shape: [N, 4]
        gt_labels = torch.cat(gt_labels)  # Shape: [N]
        batch_idx = torch.tensor(batch_idx, device=gt_boxes.device)
        
        # Reshape predictions
        B = len(targets)
        N = box_preds.size(1)
        num_classes = cls_preds.size(-1)
        
        box_preds = box_preds.view(-1, 4)  # [B*N, 4]
        cls_preds = cls_preds.view(-1, num_classes)  # [B*N, num_classes]
        
        # Match predictions to ground truth
        matched_box_preds = []
        matched_cls_preds = []
        matched_labels = []
        
        for i in range(B):
            curr_indices = (batch_idx == i)
            curr_boxes = gt_boxes[curr_indices]
            curr_labels = gt_labels[curr_indices]
            
            if len(curr_boxes) > 0:  # Only process if there are ground truth boxes
                curr_preds = box_preds[i*N:(i+1)*N]
                ious = box_iou(curr_boxes, curr_preds)
                matched_idx = ious.max(dim=1)[1]
                
                matched_box_preds.append(curr_preds[matched_idx])
                matched_cls_preds.append(cls_preds[i*N:(i+1)*N][matched_idx])
                matched_labels.append(curr_labels)
        
        if len(matched_box_preds) == 0:
            return torch.tensor(0.0, device=box_preds.device, requires_grad=True)
            
        matched_box_preds = torch.cat(matched_box_preds)
        matched_cls_preds = torch.cat(matched_cls_preds)
        matched_labels = torch.cat(matched_labels)
        
        # Create one-hot encoding
        gt_labels_one_hot = torch.zeros(
            (matched_labels.size(0), num_classes),
            dtype=matched_cls_preds.dtype,
            device=matched_cls_preds.device
        )
        
        # Ensure labels are valid indices
        valid_labels = matched_labels.clamp(0, num_classes - 1)
        gt_labels_one_hot.scatter_(1, valid_labels.unsqueeze(1), 1)
        
        # Calculate losses
        cls_loss = self.cls_loss(matched_cls_preds, gt_labels_one_hot)
        box_loss = self.box_loss(matched_box_preds, gt_boxes[batch_idx])
        
        return cls_loss + box_loss

def box_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes
    boxes1: [N, 4]
    boxes2: [M, 4]
    Returns: [N, M]
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    
    union = area1[:, None] + area2 - inter
    
    return inter / union
    
criterion = DetectionLoss()

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