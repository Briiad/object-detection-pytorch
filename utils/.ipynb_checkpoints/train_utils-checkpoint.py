import torch
from utils.metrics import DetectionMetrics
from tqdm import tqdm

from utils.custom_utils import Averager

train_loss_history = Averager()

def train_one_epoch(model, optimizer, data_loader, device, criterion, scheduler=None):
    model.train()
    
    progress = tqdm(data_loader, desc="Training", leave=False, total=len(data_loader))
    
    for i, data in enumerate(progress):
        optimizer.zero_grad()
        images, targets = data
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Check if targets are empty
        if any(len(t['boxes']) == 0 for t in targets):
            continue
        
        loss_dict = model(images, targets)
        
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        
        train_loss_history.send(loss_value)
        
        losses.backward()
        optimizer.step()
        
        progress.set_description(f"Training Loss: {train_loss_history.value:.4f}")
    return loss_value

def evaluate(model, data_loader, device, criterion):
    model.eval()
    progress = tqdm(data_loader, desc="Evaluating", leave=False, total=len(data_loader))
    
    all_targets = []
    all_preds = []
    
    for i, data in enumerate(progress):
        images, targets = data
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            outputs = model(images)
        
        for i in range(len(images)):
            true_dict = dict()
            preds_dict = dict()
            true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
            true_dict['labels'] = targets[i]['labels'].detach().cpu()
            preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
            preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
            preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
            all_preds.append(preds_dict)
            all_targets.append(true_dict)
    
    metrics = DetectionMetrics()
    metrics.update(all_preds, all_targets)
    results = metrics.compute()
    return results