import torch
from utils.metrics import DetectionMetrics
from tqdm import tqdm

def train_one_epoch(model, optimizer, data_loader, device, criterion, scheduler=None):
    model.train()
    epoch_loss = 0.0
    for images, targets in tqdm(data_loader, desc="Training"):
        # Stack images into a single batch tensor
        images = torch.stack(images).to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        cls_preds, box_preds = model(images)

        # Calculate losses
        loss = criterion(cls_preds, box_preds, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        if scheduler:
            scheduler.step()

    return epoch_loss / len(data_loader)

def evaluate(model, data_loader, device, criterion):
    model.eval()
    metrics = DetectionMetrics()
    epoch_loss = 0.0

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Model predictions
            cls_preds, box_preds = model(images)

            # Convert model outputs to expected format
            preds = []
            for cls_pred, box_pred in zip(cls_preds, box_preds):
                preds.append({
                    "boxes": box_pred.cpu(),
                    "scores": torch.sigmoid(cls_pred).cpu(),  # Adjust based on your activation function
                    "labels": torch.argmax(cls_pred, dim=1).cpu(),
                })

            # Update metrics
            metrics.update(preds, targets)

            # Compute loss for logging
            loss = criterion(cls_preds, box_preds, targets)
            epoch_loss += loss.item()

    # Compute final metrics
    metric_results = metrics.compute()
    metrics.reset()

    return epoch_loss / len(data_loader), metric_results
