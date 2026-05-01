import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from dataPipe import get_data_loaders
from model import CityGuesserTransfer
import constants as c


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def run_evaluation_metrics():
    device = get_device()
    print(f"Using device: {device}")

    _, _, test_loader = get_data_loaders(batch_size=c.BATCH_SIZE)

    model = CityGuesserTransfer(num_cities=c.NUM_CITIES).to(device)
    model.load_state_dict(torch.load("best_city_guesser.pth", map_location=device))
    model.eval()

    city_names = test_loader.dataset.classes

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    #Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=city_names,
        yticklabels=city_names
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted City")
    plt.ylabel("Actual City")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    #ROC Curve - Multi-class
    labels_bin = label_binarize(all_labels, classes=range(c.NUM_CITIES))

    plt.figure(figsize=(10, 8))

    for i in range(c.NUM_CITIES):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{city_names[i]} AUC={roc_auc:.2f}")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve - One vs Rest")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(fontsize="small", loc="lower right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_evaluation_metrics()