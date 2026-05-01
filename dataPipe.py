import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import constants as c

def get_data_loaders(batch_size=c.BATCH_SIZE):   
    #transformations (the rules for altering the images)
    image_prep = transforms.Compose([
        transforms.Resize((224, 224)), #Safety check for size
        transforms.ToTensor()          #Converts to a tensor (a multi-dimensional grid of numbers) and normalizes to decimals
    ])

    #Loading the datasets 
    train_dataset = datasets.ImageFolder(root=c.TRAIN_DIR, transform=image_prep)
    val_dataset = datasets.ImageFolder(root=c.VAL_DIR, transform=image_prep)
    test_dataset = datasets.ImageFolder(root=c.TESST_DIR, transform=image_prep)

    #Building the DataLoaders 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    return train_loader, val_loader, test_loader

#Testing Block 
if __name__ == "__main__":
    print("Testing the pipeline...")
    train, val, test = get_data_loaders(batch_size=32)
    
    test_images, test_labels = next(iter(train))
    print(f"Success! Image batch shape is: {test_images.shape}")