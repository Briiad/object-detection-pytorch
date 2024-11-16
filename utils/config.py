import os
import torch

# Dataset settings
DATASET_PATH = "dataset"
IMAGE_DIR = "dataset/train"
ANNOTATIONS_FILE = "dataset/train/_annotations.coco.json"

# Evaluation settings
VAL_IMAGE_DIR = "dataset/valid"
VAL_ANNOTATIONS_FILE = "dataset/valid/_annotations.coco.json"
VAL_CHECKPOINT_FILE = "checkpoints"

# Model settings
NUM_CLASSES = 4
IMAGE_SIZE = (320, 320)

# Training settings
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"