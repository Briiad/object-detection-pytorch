import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CustomDataset(Dataset):
  """
  Example usage:
  transform = transforms.Compose([
      transforms.Resize((128, 128)),
      transforms.ToTensor(),
  ])
  dataset = CustomDataset(image_dir='path/to/images', annotations_file='path/to/annotations.json', transform=transform)
  """
  
  def __init__(self, image_dir, annotations_file, transform=None):
    self.image_dir = image_dir
    self.annotations = self.load_annotations(annotations_file)
    self.transform = transform

  def load_annotations(self, annotations_file):
    with open(annotations_file, 'r') as file:
      coco = json.load(file)
    
    annotations = {}
    for annotation in coco['annotations']:
      image_id = annotation['image_id']
      label = annotation['category_id']
      if image_id not in annotations:
        annotations[image_id] = []
      annotations[image_id].append(label)
    
    self.image_info = {img['id']: img for img in coco['images']}
    return annotations

  def __len__(self):
    return len(self.image_info)

  def __getitem__(self, idx):
    image_id = list(self.image_info.keys())[idx]
    image_info = self.image_info[image_id]
    image_path = os.path.join(self.image_dir, image_info['file_name'])
    image = Image.open(image_path).convert("RGB")
    labels = self.annotations[image_id]

    if self.transform:
      image = self.transform(image)

    return image, labels