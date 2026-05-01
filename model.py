import torch
import torch.nn as nn

class CityGuesserCNN(nn.Module):
    def __init__(self, num_cities):
        super().__init__()
        

        #Takes in 3 color channels, outputs 16 pattern channels using a 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        

        self.flatten = nn.Flatten()#converts to 1D
        

        
        self.fc1 = nn.Linear(in_features=16 * 112 * 112, out_features=512) 
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5) 
        self.fc2 = nn.Linear(in_features=512, out_features=num_cities) #The final city guesses

    def forward(self, x):
        
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
#Testing Block
if __name__ == "__main__":
    print("Testing the CNN assembly line...")
    
    test_model = CityGuesserCNN(num_cities=23)

    dummy_batch = torch.randn(32, 3, 224, 224)
    
    final_guesses = test_model(dummy_batch)
    
    print(f"Input batch shape: {dummy_batch.shape}")
    print(f"Output guesses shape: {final_guesses.shape}")

    print(f"Raw estimates for Image 1:\n{final_guesses[0]}")