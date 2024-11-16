import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

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

        # Map image_id to annotations
        annotations = {}
        for annotation in coco['annotations']:
            image_id = annotation['image_id']
            bbox = annotation['bbox']  # COCO format: [x, y, width, height]
            label = annotation['category_id']

            if image_id not in annotations:
                annotations[image_id] = {"boxes": [], "labels": []}
            annotations[image_id]["boxes"].append(bbox)
            annotations[image_id]["labels"].append(label)

        # Store image info
        self.image_info = {img['id']: img for img in coco['images']}
        return annotations

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        image_id = list(self.image_info.keys())[idx]
        image_info = self.image_info[image_id]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path).convert("RGB")

        # Get annotations for this image
        target = self.annotations[image_id]

        # Convert boxes to tensors and adjust format (COCO uses [x, y, w, h])
        target["boxes"] = torch.tensor(target["boxes"], dtype=torch.float32)
        target["boxes"][:, 2:] += target["boxes"][:, :2]  # Convert [x, y, w, h] -> [x_min, y_min, x_max, y_max]
        target["labels"] = torch.tensor(target["labels"], dtype=torch.int64)

        # Apply transforms to the image
        if self.transform:
            image = self.transform(image)

        return image, target
