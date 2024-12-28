import torch
from torch.utils.data import DataLoader
from dataset import PetSnoutDataset, data_transform  

# Initialize the dataset
dataset = PetSnoutDataset(img_dir='/content/drive/MyDrive/ColabNotebooks/lab2/data/oxford-iiit-pet-noses/images-original/images',
                         labels_file='/content/drive/MyDrive/ColabNotebooks/lab2/data/oxford-iiit-pet-noses/test_noses.txt',
                         transform=data_transform)

# Initialize the DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Reality check 
for idx, (images, keypoints, original_size) in enumerate(dataloader):
    print(f"Batch {idx+1}")
    print(f"Image batch shape: {images.shape}")
    print(f"Keypoints batch: {keypoints}")
    print(f"Original sizes: {original_size}")
    
    # Stop after checking the first 3 batches
    if idx >= 2:
        break
