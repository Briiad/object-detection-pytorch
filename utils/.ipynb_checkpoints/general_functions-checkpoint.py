import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from data import CustomDataset
from torchvision import transforms

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

def visualize_predictions(image, boxes, labels, scores, class_names, threshold=0.5):
    """Visualize predictions on an image."""
    image = Image.fromarray((image.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'))
    draw = ImageDraw.Draw(image)
    for box, label, score in zip(boxes, labels, scores):
        if score > threshold:
            draw.rectangle(box.tolist(), outline="red", width=2)
            draw.text((box[0], box[1]), f"{class_names[label]}: {score:.2f}", fill="red")
    image.show()

def save_checkpoint(model, optimizer, epoch, filepath):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }, filepath)

def load_checkpoint(filepath, model, optimizer=None):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']
