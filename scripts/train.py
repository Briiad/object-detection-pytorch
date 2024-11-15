import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from models import DetectionModel
from data import CustomDataset
from utils import train_one_epoch
from utils import *
from utils import DetectionMetrics

# Load the dataset
train_dataset = CustomDataset(images_dir=IMAGE_DIR, annotations_file=ANNOTATIONS_FILE, transforms=None)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Initialize the model, optimizer, and loss function
model = DetectionModel(num_classes=NUM_CLASSES).to(DEVICE)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

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