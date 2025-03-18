import os
import cv2
import numpy as np
import random
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mode
from PIL import Image
#rotate the picture in random angles
def random_rotate(image, angle_range=(0, 90, 180, 270)):
    angle = random.choice(angle_range)
    image = image.rotate(angle)
    return image

#get the different data base on the different mode
def get_data(data_type):
    if data_type == "corners":
        piece_dirs = {
            "top_left": "puzzle_pieces4/top_left",
            "bottom_left": "puzzle_pieces4/bottom_left",
            "top_right": "puzzle_pieces4/top_right",
            "bottom_right": "puzzle_pieces4/bottom_right",
        }
    elif data_type == "sides":
        piece_dirs = {
            "left": "puzzle_pieces2/left",
            "right": "puzzle_pieces2/right",
        }
    else:
        raise ValueError("Invalid data type. Choose 'corners' or 'sides'.")

    all_images, all_labels = [], []
    for idx, (label, path) in enumerate(piece_dirs.items()):
        images = glob(os.path.join(path, "*.jpg"))
        all_images.extend(images)
        all_labels.extend([idx] * len(images))

    data = list(zip(all_images, all_labels))
    random.shuffle(data)
    return zip(*data)

#load the picture and see whether it need rotation
def load_and_preprocess(image_paths, img_size=(128, 128), apply_rotation=False):
    data = []
    for path in image_paths:
        img = cv2.imread(path)
        img = cv2.resize(img, img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if apply_rotation:
            img = random_rotate(Image.fromarray(img))
            img = np.array(img)
        img = img.flatten()
        data.append(img)
    return np.array(data)

class PuzzleDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (128, 128))

        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        return image, label

class PuzzleCNN(nn.Module):
    def __init__(self, num_classes):
        super(PuzzleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

#Use the CNN to train the model and use back propagation to get the loss and train model
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct = 0, 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(dataloader), total_correct / len(dataloader.dataset)

#Test the model
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_correct += (outputs.argmax(1) == labels).sum().item()
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print("CNN Classification Report:")
    print(classification_report(all_labels, all_preds,zero_division=0))

    print("CNN Confusion Matrix is saved")

    sns.heatmap(confusion_matrix(all_labels, all_preds), annot=True, fmt='d')
    plt.savefig('CNN_confusion_matrix.png')
    plt.close()

    return total_loss / len(dataloader), total_correct / len(dataloader.dataset)


if __name__ == "__main__":
    #Base on the input (corners = 2*2, sides = 1*2) to choose the puzzle type, then gat and split the dataset into training data and testing data
    data_type = input("Choose dataset ('corners' or 'sides'): ")
    all_images, all_labels = get_data(data_type)
    train_images, test_images, train_labels, test_labels = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

    train_data = load_and_preprocess(train_images)
    test_data = load_and_preprocess(test_images)

    #SVM
    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(train_data, train_labels)
    svm_preds = svm_model.predict(test_data)
    svm_probs = svm_model.predict_proba(test_data)

    svm_accuracy = accuracy_score(test_labels, svm_preds)
    print(f"SVM Accuracy: {svm_accuracy:.4f}")

    print("SVM Classification Report:")
    print(classification_report(test_labels, svm_preds,zero_division=0))

    print("SVM Confusion Matrix is saved")
    sns.heatmap(confusion_matrix(test_labels, svm_preds), annot=True, fmt='d')
    plt.savefig('SVM_confusion_matrix.png')
    plt.close()

    '''
    #Use HOG to extract the features
    from skimage.feature import hog
    def extract_hog_features(image_paths):
        hog_features = []
        for path in image_paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            fd, _ = hog(img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
            hog_features.append(fd)
        return np.array(hog_features)
    
    train_data_hog = extract_hog_features(train_images)
    test_data_hog = extract_hog_features(test_images)

    svm_model_hog = SVC(kernel='linear', probability=True)
    svm_model_hog.fit(train_data_hog, train_labels)
    svm_preds_hog = svm_model_hog.predict(test_data_hog)
    svm_probs_hog = svm_model_hog.predict_proba(test_data_hog)
    svm_accuracy_hog = accuracy_score(test_labels, svm_preds_hog)

    print(f"SVM with HOG features accuracy: {svm_accuracy_hog:.4f}")
    print("SVM Classification Report:")
    print(classification_report(test_labels, svm_preds_hog,zero_division=0))

    print("SVM Confusion Matrix is saved")
    sns.heatmap(confusion_matrix(test_labels, svm_preds_hog), annot=True, fmt='d')
    plt.savefig('SVM_HOG_confusion_matrix.png')
    plt.close()
    
    '''

    #K-means
    num_clusters = 4 if data_type == "corners" else 2
    #num_clusters = 5
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(train_data)
    kmeans_preds = kmeans.predict(test_data)

    #Match clusters with true labels
    mapped_preds = np.zeros_like(kmeans_preds)
    for i in range(num_clusters):
        mask = (kmeans_preds == i)
        mapped_preds[mask] = mode(np.array(test_labels)[mask], keepdims=False)[0]

    
    kmeans_accuracy = accuracy_score(test_labels, mapped_preds)
    print(f"K-means Accuracy: {kmeans_accuracy:.4f}")

    print(f"K-means Adjusted Rand Index: {adjusted_rand_score(test_labels, kmeans_preds):.4f}")
    print(f"K-means Normalized Mutual Information: {normalized_mutual_info_score(test_labels, kmeans_preds):.4f}")


    #CNN
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Resize((128, 128))])

    num_clusters = 4 if data_type == "corners" else 2
    train_dataset = PuzzleDataset(train_images, train_labels, transform)
    test_dataset = PuzzleDataset(test_images, test_labels, transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PuzzleCNN(num_classes=num_clusters).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #train the data and get the epoch-loss/accurecy graph
    losses = []
    acc = []
    for epoch in range(10):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/10: Train Acc={train_acc:.4f}, Train Loss={train_loss:.4f}")
        losses.append(train_loss)
        acc.append(train_acc)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(losses) + 1), losses, label='Loss', marker='o', linestyle='-')
    plt.plot(range(1, len(acc) + 1), acc, label='Accuracy', marker='s', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/accuracy')
    plt.legend()
    plt.title('Loss/Accuracy per Epoch')

    plt.savefig(os.path.join('accuracy_Adam.png'))
    plt.close()

    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Acc={test_acc:.4f}, Test Loss={test_loss:.4f}")

    '''
    #SGD
    model = PuzzleCNN(num_classes=num_clusters).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    losses = []
    acc = []
    for epoch in range(10):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/10: Train Acc={train_acc:.4f}, Train Loss={train_loss:.4f}")
        losses.append(train_loss)
        acc.append(train_acc)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(losses) + 1), losses, label='Loss', marker='o', linestyle='-')
    plt.plot(range(1, len(acc) + 1), acc, label='Accuracy', marker='s', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/accuracy')
    plt.legend()
    plt.title('Loss/Accuracy per Epoch')

    plt.savefig(os.path.join('accuracy_SGD.png'))
    plt.close()
    
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Acc={train_acc:.4f}, Test Loss={train_loss:.4f}")
    '''

