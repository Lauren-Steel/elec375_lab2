import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2

class PetSnoutDataset(Dataset):
    def __init__(self, img_dir, labels_file, transform=None):
        self.img_labels = self.read_labels(labels_file)
        self.img_dir = img_dir
        self.transform = transform

    def read_labels(self, annotations_file):
        labels = {}
        with open(annotations_file, "r") as file:
            for line in file:
                parts = line.strip().split(',"(')
                if len(parts) == 2:
                    image_name = parts[0].strip()
                    keypoints_str = parts[1].strip(')"')
                    keypoints = tuple(map(int, keypoints_str.split(',')))
                    labels[image_name] = keypoints
        return labels

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name, keypoints = list(self.img_labels.items())[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load img
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Image {img_path} not found, skipping...")
            return self.__getitem__((idx + 1) % len(self.img_labels))

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create original size tuple for rescaling keypoints later
        original_width, original_height = image.shape[1], image.shape[0]
        original_size = (original_width, original_height)

        # Scale keypoints based on the original size
        x_scale = 227 / original_width
        y_scale = 227 / original_height
        keypoints = torch.tensor([keypoints[0] * x_scale, keypoints[1] * y_scale])

        # Convert the NumPy array to a PIL Image before transformation
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, keypoints, original_size

# Image transformation with data augmentation
data_transform = transforms.Compose([
    # Data aug random horizontal flip
    transforms.RandomHorizontalFlip(), 
    # Data aug random rotation within 15 degrees 
    transforms.RandomRotation(15),  
    transforms.Resize((227, 227)),
    # Convert the image to a tensor  
    transforms.ToTensor(),  
    # Normalize the img
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

