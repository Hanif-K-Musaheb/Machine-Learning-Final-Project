import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter

from dataPipe import get_data_loaders
from model import CityGuesserTransfer

import constants as c
import math
import time

def train_model():

    if torch.backends.mps.is_available():
        device = torch.device("mps")

    elif torch.cuda.is_available():
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")
    print(f"Training using device: {device}")

    print("Loading data batchs...")
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=32)

    counts = Counter(train_loader.dataset.targets)

    print("Class counts:")
    for i, city in enumerate(train_loader.dataset.classes):
        print(city, counts[i])

    model = CityGuesserTransfer(num_cities=c.NUM_CITIES).to(device)

    class_counts = [counts[i] for i in range(len(train_loader.dataset.classes))]

    weights = [sum(class_counts) / (len(class_counts) * count) for count in class_counts]

    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=c.LEARNING_RATE)

   
    epochs = c.EPOCHS #5# 
    
    best_val_accuracy = 0

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

            time.sleep(c.COOLING_TIME)

            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f} | Accuracy: {(math.e**-(loss.item()))*100:.1f}%")

        avg_loss = running_loss / len(train_loader)
  
        model.eval() 
        val_loss = 0.0
        correct_guesses = 0
        total_images = 0

        with torch.no_grad(): 
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                predictions = model(images)
                
                loss = criterion(predictions, labels)
                val_loss += loss.item()
                
                _, predicted_class = torch.max(predictions, 1) 
                total_images += labels.size(0)
                correct_guesses += (predicted_class == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = (correct_guesses / total_images) * 100

        print(f"--- End of Epoch {epoch+1} ---")
        print(f"Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%")

        if val_accuracy > best_val_accuracy:
            print(f"higher accuracy found saving model... ({best_val_accuracy:.2f}% -> {val_accuracy:.2f}%)")
            best_val_accuracy = val_accuracy
            
            torch.save(model.state_dict(), "best_city_guesser.pth")




if __name__ == "__main__":
    train_model()