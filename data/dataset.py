def load_annotations(self, annotations_file):
  with open(annotations_file, 'r') as file:
    coco = json.load(file)

  # Print dataset statistics
  print(f"Total images: {len(coco['images'])}")
  print(f"Total annotations: {len(coco['annotations'])}")
  
  # Count annotations per category
  category_counts = {}
  for ann in coco['annotations']:
    cat_id = ann['category_id']
    category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
  
  print("\nAnnotations per category:")
  for cat in coco['categories']:
    count = category_counts.get(cat['id'], 0)
    print(f"Category {cat['name']}: {count}")

  # Continue with the existing code...
  annotations = {}
  for annotation in coco['annotations']:
    image_id = annotation['image_id']
    bbox = annotation['bbox']
    label = annotation['category_id']

    if image_id not in annotations:
      annotations[image_id] = {"boxes": [], "labels": []}
    annotations[image_id]["boxes"].append(bbox)
    annotations[image_id]["labels"].append(label)

  self.image_info = {img['id']: img for img in coco['images']}
  return annotations
