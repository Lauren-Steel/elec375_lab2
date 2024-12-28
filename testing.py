import torch
import argparse
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import PetSnoutDataset
from model import SnoutNet
import numpy as np
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Argument parser with augmentation option
parser = argparse.ArgumentParser(description='Test the SnoutNet model')
parser.add_argument('--num-images', type=int, default=10, help='Number of images to process and save')
parser.add_argument('--augment', action='store_true', help='Use data augmentation during testing')
args = parser.parse_args()

# Load model
model = SnoutNet().to(device)
model.load_state_dict(torch.load('/content/drive/MyDrive/ColabNotebooks/lab2/snoutnetmodel.pth', map_location=device))
model.eval()

# Apply augmentation based on flag
if args.augment:
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
else:
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Function to denormalize images
def denormalize(tensor, means, stds):
    denorm = transforms.Normalize(
        [-m / s for m, s in zip(means, stds)],
        [1 / s for s in stds]
    )
    return denorm(tensor)

means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]

test_image_dir = '/content/drive/MyDrive/ColabNotebooks/lab2/data/oxford-iiit-pet-noses/images-original/images'
test_labels_file = '/content/drive/MyDrive/ColabNotebooks/lab2/data/oxford-iiit-pet-noses/test_noses.txt'
test_dataset = PetSnoutDataset(test_image_dir, test_labels_file, transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

def euclidean_distance(predicted, true):
    return np.linalg.norm(predicted - true)

distances = []

# Directory to save imgs
output_dir = '/content/drive/MyDrive/ColabNotebooks/lab2/results_wout_aug'

# Create the directory 
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i, (images, true_keypoints, original_sizes) in enumerate(test_loader):
    if args.num_images is not None and i >= args.num_images:
        break

    images = images.to(device)
    predicted_keypoints = model(images)

    # Reshape predicted and true keypoints
    predicted_keypoints = predicted_keypoints.view(-1, 2).cpu().detach().numpy()
    true_keypoints = true_keypoints.view(-1, 2).cpu().detach().numpy()

    dist = euclidean_distance(predicted_keypoints, true_keypoints)
    distances.append(dist)

    # Denormalize the img for visualization
    img = denormalize(images[0], means, stds).cpu()
    img = transforms.ToPILImage()(img)

    # Plot and save the img
    plt.imshow(img)
    # Green for ground truth
    plt.scatter(*true_keypoints[0], color='green', label='Ground Truth') 
    # Red for prediction 
    plt.scatter(*predicted_keypoints[0], color='red', label='Prediction')  
    plt.legend()

    # Save the img
    img_save_path = os.path.join(output_dir, f'image_{i+1}.png')
    plt.savefig(img_save_path)
    plt.close()

# Calculate and print metrics
min_distance = np.min(distances)
mean_distance = np.mean(distances)
max_distance = np.max(distances)
std_distance = np.std(distances)

print(f'Min Distance: {min_distance}')
print(f'Mean Distance: {mean_distance}')
print(f'Max Distance: {max_distance}')
print(f'Standard Deviation: {std_distance}')

