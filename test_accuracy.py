import torch
from dataPipe import get_data_loaders
from model import CityGuesserTransfer
import constants as c

def run_final_exam():
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    elif torch.cuda.is_available():
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")
    print(f"Loading grading system on: {device}")

    _, _, test_loader = get_data_loaders(batch_size=32)

    print("Loading your trained AI...")
    model = CityGuesserTransfer(num_cities=c.NUM_CITIES).to(device)
    model.load_state_dict(torch.load("best_city_guesser.pth", map_location=device))
    
    model.eval()

    city_names = [
        "Bangkok", "Barcelona", "Boston", "Brussels", "BuenosAires", 
        "Chicago", "Lisbon", "London", "LosAngeles", "Madrid", 
        "Medellin", "Melbourne", "MexicoCity", "Miami", "Minneapolis", 
        "Osaka", "OSL", "Phoenix", "PRG", "PRS", 
        "Rome", "TRT", "WashingtonDC"
    ]
    
    overall_correct = 0
    overall_total = 0
    
    class_correct = {city: 0 for city in city_names}
    class_total = {city: 0 for city in city_names}

    print("Starting the Final Exam... (This might take a minute)")
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            predictions = model(images)
            _, predicted_classes = torch.max(predictions, 1)
            
            overall_total += labels.size(0)
            overall_correct += (predicted_classes == labels).sum().item()
            
            for i in range(labels.size(0)):
                true_label = labels[i].item()
                guessed_label = predicted_classes[i].item()
                
                city_name = city_names[true_label]
                
                class_total[city_name] += 1
                
                if true_label == guessed_label:
                    class_correct[city_name] += 1


    city_name_convert_dict={"OSL":"Oslo",
                            "PRG":"Prague",
                            "PRS":"Paris",
                            "TRT":"Toronto"}

    print("\n" + "="*40)
    print("TESTING RESULTS")
    print("="*40)
    
    overall_accuracy = (overall_correct / overall_total) * 100
    print(f"Overall Accuracy: {overall_accuracy:.2f}% ({overall_correct}/{overall_total} images)\n")
    
    print("--- Accuracy Breakdown by City ---")
    for city in city_names:
        if city in ["OSL", "PRG", "PRS", "TRT"]:
            city_print_name=city_name_convert_dict[city]
        else:
            city_print_name=city
        
        if class_total[city] > 0:
            city_acc = (class_correct[city] / class_total[city]) * 100
            print(f"{city_print_name.ljust(15)}: {city_acc:>6.2f}%  ({class_correct[city]}/{class_total[city]})")
        else:
            print(f"{city_print_name.ljust(15)}: No test images found!")
    print("="*40)

if __name__ == "__main__":
    run_final_exam()