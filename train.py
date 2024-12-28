import argparse
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import PetSnoutDataset  
from model import SnoutNet 
import torch.nn as nn
import time
import torchvision.transforms as transforms

# Transformations for the dataset w optional augs
def get_transforms(horizontal_flip=False, rotation=False):
    print("Applying data augmentation...")
    transform_list = []
    
    if horizontal_flip:
        # Random horizontal flip
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))  
    if rotation:
        # Random rotation within 15 degrees
        transform_list.append(transforms.RandomRotation(15))  
    
    transform_list.extend([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transforms.Compose(transform_list)

def rescale_keypoints(keypoints, original_size):
    original_width, original_height = original_size
    x_scale = original_width / 227
    y_scale = original_height / 227
    rescaled_keypoints = torch.tensor([keypoints[0] * x_scale, keypoints[1] * y_scale])
    return rescaled_keypoints

def train(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, save_path):
    model.train()
    epoch_losses = []
    epoch_val_losses = []

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Training 
        model.train()
        train_loss = 0
        num_train_samples = 0
        for images, keypoints, _ in train_loader:
            images = images.to(device)
            keypoints = keypoints.to(device)

            optimizer.zero_grad()
            outputs = model(images).to(device)

            loss = criterion(outputs, keypoints)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()
            num_train_samples += images.size(0)

        avg_train_loss = train_loss / num_train_samples
        epoch_losses.append(avg_train_loss)

        # Validation 
        model.eval()
        val_loss = 0
        num_samples = 0
        with torch.no_grad():
            for images, true_keypoints, original_sizes in test_loader:
                images = images.to(device)
                true_keypoints = true_keypoints.to(device)

                predicted_keypoints = model(images)

                # Rescale predicted keypoints
                rescaled_keypoints = []
                for i in range(len(predicted_keypoints)):
                    width, height = original_sizes[i * 2].item(), original_sizes[i * 2 + 1].item()
                    rescaled_keypoints.append(rescale_keypoints(predicted_keypoints[i], (width, height)))

                rescaled_keypoints = torch.stack(rescaled_keypoints).to(device)

                # Compute loss
                loss = criterion(rescaled_keypoints, true_keypoints)
                val_loss += loss.item()
                num_samples += images.size(0)

        avg_val_loss = val_loss / num_samples
        epoch_val_losses.append(avg_val_loss)

        scheduler.step()

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        print(f'Epoch [{epoch + 1}/{num_epochs}], Duration: {epoch_duration:.2f} sec, Training Loss: {avg_train_loss:.4f}')

    # Save the model w name depending on augmentation 
    torch.save(model.state_dict(), save_path)

    # Plot the training loss
    plt.plot(epoch_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.legend()
    plt.savefig('/content/drive/MyDrive/ColabNotebooks/lab2/training_loss_plot.png')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SnoutNet Training')
    parser.add_argument('-b', type=int, default=16, help='batch size')
    parser.add_argument('-e', type=int, default=20, help='number of epochs')
    parser.add_argument('--no-augment', action='store_true', help='Use no data augmentation for training')
    parser.add_argument('--flip', action='store_true', help='Use random horizontal flip for training')
    parser.add_argument('--rotate', action='store_true', help='Use random rotation for training')
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device {device}')

    train_image_dir = '/content/drive/MyDrive/ColabNotebooks/lab2/data/oxford-iiit-pet-noses/images-original/images/'
    train_labels_file = '/content/drive/MyDrive/ColabNotebooks/lab2/data/oxford-iiit-pet-noses/train_noses.txt'

    test_image_dir = '/content/drive/MyDrive/ColabNotebooks/lab2/data/oxford-iiit-pet-noses/images-original/images/'
    test_labels_file = '/content/drive/MyDrive/ColabNotebooks/lab2/data/oxford-iiit-pet-noses/test_noses.txt'

    # Determine which transformations to apply
    if args.no_augment:
        train_transform = get_transforms()
        save_path = '/content/drive/MyDrive/ColabNotebooks/lab2/snoutnetmodel.pth'
    elif args.flip and args.rotate:
        train_transform = get_transforms(horizontal_flip=True, rotation=True)
        save_path = '/content/drive/MyDrive/ColabNotebooks/lab2/snoutnetaugmented.pth'
    elif args.flip:
        train_transform = get_transforms(horizontal_flip=True)
        save_path = '/content/drive/MyDrive/ColabNotebooks/lab2/snoutnetflip.pth'
    elif args.rotate:
        train_transform = get_transforms(rotation=True)
        save_path = '/content/drive/MyDrive/ColabNotebooks/lab2/snoutnetrotate.pth'
    else:
        # No aug
        train_transform = get_transforms()  
        save_path = '/content/drive/MyDrive/ColabNotebooks/lab2/snoutnetmodel.pth'
    # No aug for testing
    test_transform = get_transforms()  

    train_dataset = PetSnoutDataset(train_image_dir, train_labels_file, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=True)

    test_dataset = PetSnoutDataset(test_image_dir, test_labels_file, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = SnoutNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)

    # Pass the save path to the train fn
    train(model, train_loader, test_loader, criterion, optimizer, scheduler, args.e, save_path)


