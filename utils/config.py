import os
from torchvision import transforms
from data import CustomDataset

def initialize_dataset(image_dir, annotations_file):
  """
  Example usage:
  dataset = initialize_dataset('path/to/images', 'path/to/annotations.json')  
  """
  transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
  ])
  
  dataset = CustomDataset(image_dir=image_dir, annotations_file=annotations_file, transform=transform)
  return dataset