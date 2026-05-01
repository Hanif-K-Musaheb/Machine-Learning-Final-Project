import torch
import torch.nn as nn
import torch.optim as optim

from dataPipe import get_data_loaders
from model import CityGuesserCNN

import constants as c

def train_model():
    #Hardware Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training using device: {device}")

    print("Loading data trucks...")
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=32)

    print("Building the AI brain...")
    model = CityGuesserCNN(num_cities=c.NUM_CITIES).to(device)

    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=c.LEARNING_RATE)

    epochs = c.EPOCHS #5# 
    
    print("Starting training!")
    for epoch in range(epochs):
        model.train() 
        running_loss = 0.0 #A temporary counter to track our error score
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            predictions = model(images)

            loss = criterion(predictions, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        print(f"--- End of Epoch {epoch+1} | Average Loss: {avg_loss:.4f} ---")

if __name__ == "__main__":
    train_model()