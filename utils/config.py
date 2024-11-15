import os
import torch

# Dataset settings
DATASET_PATH = "path/to/dataset"
IMAGE_DIR = "path/to/images"
ANNOTATIONS_FILE = "path/to/annotations.json"

# Evaluation settings
VAL_IMAGE_DIR = "path/to/val/images"
VAL_ANNOTATIONS_FILE = "path/to/val/annotations.json"
VAL_CHECKPOINT_FILE = "path/to/checkpoint.pth"

# Model settings
NUM_CLASSES = 4
IMAGE_SIZE = (320, 320)

# Training settings
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"