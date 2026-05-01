import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image 

from model import CityGuesserTransfer
import constants as c

def predict_city(image_path):
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    elif torch.cuda.is_available():
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")

    print(f"Loading AI on device: {device}...")

    model = CityGuesserTransfer(num_cities=c.NUM_CITIES).to(device)
    model.load_state_dict(torch.load("best_city_guesser.pth", map_location=device))
    
    
    model.eval()
    image_prep = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert("RGB")
    img_tensor = image_prep(img)
    
    img_batch = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        raw_scores = model(img_batch)
        
        percentages = F.softmax(raw_scores, dim=1)[0] * 100
        
        sorted_percentages, sorted_indices = torch.sort(percentages, descending=True)

    city_names = [
    "Bangkok", "Barcelona", "Boston", "Brussels", "BuenosAires", 
    "Chicago", "Lisbon", "London", "LosAngeles", "Madrid", 
    "Medellin", "Melbourne", "MexicoCity", "Miami", "Minneapolis", 
    "Osaka", "OSL", "Phoenix", "PRG", "PRS", 
    "Rome", "TRT", "WashingtonDC"       
    ]

    print("-" * 30)
    print("Top Predictions:")

    #Showing top 5 guesses
    top_k = 5

    for i in range(top_k):
        city = city_names[sorted_indices[i].item()]
        confidence = sorted_percentages[i].item()
        print(f"{i + 1}. {city}: {confidence:.2f}%")

    print("-" * 30)

if __name__ == "__main__":
    test_image = "pa.webp"
    predict_city(test_image)