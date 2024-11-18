import torch
from torch.utils.data import DataLoader
from models import DetectionModel
from data import CustomDataset
from utils import evaluate
from utils import *

# Load the dataset
val_dataset = CustomDataset(images_dir=VAL_IMAGE_DIR, annotations_file=VAL_ANNOTATIONS_FILE, transforms=None)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Initialize the model and load weights
model = DetectionModel(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(VAL_CHECKPOINT_FILE))

# Initialize metrics and evaluation
criterion = torch.nn.CrossEntropyLoss()
val_loss, metric_results = evaluate(model, val_loader, DEVICE, criterion)

# Print evaluation results
print(f"Validation Loss: {val_loss:.4f}")
print("Validation Metrics:")
for metric, value in metric_results.items():
    print(f"{metric}: {value:.4f}")
