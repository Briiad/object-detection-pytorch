import os
import json
from PIL import Image
from torch.utils.data import Dataset
import torch
import logging

logging.basicConfig(level=logging.INFO)

class CustomDataset(Dataset):
    """
    Dataset for COCO-format annotations.
    
    Args:
        image_dir (str): Directory containing images.
        annotations_file (str): Path to COCO-style annotations JSON file.
        transform (callable, optional): Transform to apply to images.
    """
    def __init__(self, image_dir, annotations_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.annotations = self.load_annotations(annotations_file)

    def load_annotations(self, annotations_file):
        with open(annotations_file, 'r') as file:
            coco = json.load(file)
        return coco

    def __getitem__(self, idx):
        # Load image and annotations
        image_info = self.annotations['images'][idx]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path).convert("RGB")

        annotations = [ann for ann in self.annotations['annotations'] if ann['image_id'] == image_info['id']]
        boxes = [ann['bbox'] for ann in annotations]
        boxes = [[x, y, x + w, y + h] for x, y, w, h in boxes]  # Convert from [x, y, w, h] to [x_min, y_min, x_max, y_max]

        # Ensure all bounding boxes have positive height and width
        valid_boxes = []
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            if x_max > x_min and y_max > y_min:
                valid_boxes.append(box)
            else:
                logging.info(f"Invalid box {box} found for image {image_info['file_name']} at index {idx}")

        target = {
            'boxes': torch.tensor(valid_boxes, dtype=torch.float32),
            'labels': torch.tensor([ann['category_id'] for ann in annotations], dtype=torch.int64)
        }

        if self.transform:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.annotations['images'])