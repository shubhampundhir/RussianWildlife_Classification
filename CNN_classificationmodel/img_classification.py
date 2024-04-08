import os 
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import wandb
import shutil

from model import Net

#Define class label mapping
class_labels = {'amur_leopard':0, 'amur_tiger':1, 'birds':2, 'black_bear':3,'brown_bear':4, 
                'dog':5,'roe_deer':6,'sika_deer':7, 'wild_boar':8, 'people':9}

#Define transforms to apply to the image
transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                                ])

#Create a custom Pytorch dataset class 
class RussianDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        #Load images and labels
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for image_name in os.listdir(class_dir):
                    image_path = os.path.join(class_dir, image_name)
                    self.images.append(image_path)
                    self.labels.append(class_labels[class_name])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def save_images_to_folders(self, save_dir, train_indices, val_indices, test_indices):
            train_dir = os.path.join(save_dir, 'train')
            val_dir = os.path.join(save_dir, 'val')
            test_dir = os.path.join(save_dir, 'test')

            for directory in [train_dir, val_dir, test_dir]:
                os.makedirs(directory, exist_ok=True)

            split_indices = [train_indices, val_indices, test_indices]
            split_names = ['train', 'val', 'test']

            for idx, (indices, split_name) in enumerate(zip(split_indices, split_names)):
                split_dir = os.path.join(save_dir, split_name)
                for i in indices:
                    image_path = self.images[i]
                    image_name = os.path.basename(image_path)
                    label = self.labels[i]
                    class_name = [key for key, value in class_labels.items() if value == label][0]

                    # Create subfolders based on class name
                    class_dir = os.path.join(split_dir, class_name)
                    os.makedirs(class_dir, exist_ok=True)

                    # Copy the image to the respective class folder
                    shutil.copy(image_path, os.path.join(class_dir, image_name))

#Set the root directory of the dataset
root_dir = '/home/shubhamp/Downloads/russian dataset/Cropped_final'

#Create a instance of custom dataset
dataset = RussianDataset(root_dir, transform = transform)

#Perform stratified random split 
train_val_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.2, stratify=dataset.labels, random_state=42)
train_indices, val_indices = train_test_split(train_val_indices, test_size=0.1, stratify=[dataset.labels[i] for i in train_val_indices], random_state=42)

#Split the dataset into train, validation, and test sets.
train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)

# Save images to specific folders
save_dir = '/home/shubhamp/Desktop/Russianwildlife_Classification/russian_dataset'
dataset.save_images_to_folders(save_dir, train_indices, val_indices, test_indices)

#Initialize the weights and biases(WandB)
wandb.init(project = 'Image Classification Russian Wildlife')

#Create data loaders for all the splits
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_classes = len(class_labels)
class_names = list(class_labels.keys())

#Compute the class frequency for the training set
train_labels = [labels for _,labels in train_dataset]
train_class_counts = np.bincount(train_labels, minlength=num_classes)


#Compute the class frequency for the validation set
val_labels = [labels for _,labels in val_dataset]
val_class_count = np.bincount(val_labels, minlength=num_classes)

# Create bar plots for the training and validation set class distribution
fig, ax = plt.subplots(2, 1, figsize=(12, 8))

# Plot training set class distribution 
ax[0].bar(np.arange(num_classes), train_class_counts, width=0.4, align='center', label='Training Set')
ax[0].set_title('Training Set Class Distribution')
ax[0].set_xlabel('Class label')
ax[0].set_ylabel('Frequency')
ax[0].set_xticks(np.arange(num_classes))
ax[0].set_xticklabels(class_names, rotation=45)
ax[0].legend()

# Plot Validation set class distribution
ax[1].bar(np.arange(num_classes), val_class_count, width=0.4, align='center', label='Validation Set')
ax[1].set_title('Validation Set Class Distribution')
ax[1].set_xlabel('Class label')
ax[1].set_ylabel('Frequency')
ax[1].set_xticks(np.arange(num_classes))
ax[1].set_xticklabels(class_names, rotation=45)
ax[1].legend()

# Adjust layout and display
# plt.tight_layout()
# plt.show()

#Create an instance of the CNN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
# print(model)

import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch.nn as nn
import seaborn as sns

#Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

#Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        # print("Data batch size:", inputs.size(0))
        # print("Label batch size:", labels.size(0))
        # break  # Print only the first batch
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to the GPU
        optimizer.zero_grad()
        
        outputs = model(inputs)
        # print(outputs.shape)

        loss = criterion(outputs, labels)
        loss.backward()

        # Apply gradient clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total +=labels.size(0)
        correct +=(predicted==labels).sum().item()

    train_loss = running_loss / len(train_loader.dataset)
    train_accuracy = correct / total

    #Validation loop
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for input, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the GPU
            outputs = model(input)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total +=labels.size(0)
            val_correct +=(predicted==labels).sum().item()

    val_loss = val_running_loss / len(val_loader.dataset)
    val_accuracy = val_correct / val_total

    #Log training and validation metrics to WandB
    wandb.log({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy 
        })
    
    print(f'Epoch [{epoch + 1}/{epochs}], '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

#Evaluate the model on the test set  
model.eval()
test_running_loss = 0.0
test_correct = 0
test_total = 0          
y_true = []
y_pred = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        test_running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.numpy())

#Calculate the accuracy and F1-score
test_accuracy = test_correct / test_total
f1 = f1_score(y_true, y_pred, average='weighted')

#Log accuracy anf F1-score to WandB
wandb.log({'test_accuracy': test_accuracy, 'f1_score': f1})

#Print the accuracy and F1-score
print(f'Test accuracy: {test_accuracy:.4f}, F1-Score: {f1:.4f}')


# Calculate and plot the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# Log confusion matrix as an image to WandB
wandb.log({"confusion_matrix": wandb.Image(plt)})

# Make predictions on some sample images from the test set
print("Predictions on sample images from the test set:")
model.eval()
with torch.no_grad():
    for i in range(5):  # Adjust the number of images you want to predict
        inputs, labels = test_dataset[i]  # Get an image and its label from the test dataset
        outputs = model(inputs.unsqueeze(0))  # Add an extra dimension for batch
        _, predicted = torch.max(outputs, 1)
        predicted_label = class_names[predicted.item()]
        true_label = class_names[labels]
        print(f"Predicted label: {predicted_label}, True label: {true_label}")

#Close the Wandb run
wandb.finish()

