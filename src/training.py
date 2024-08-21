import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import joblib

####
# This script trains and saves our hand gesture recognition model using the beforehand pre-processed hand landmarks data.
####

model_version = 2

# Dataset class to load the hand landmarks data from CSV
class HandLandmarksDataset(Dataset):
    def __init__(self, csv_file):
        # Load the CSV file
        data = pd.read_csv(csv_file)
        
        # The first column is the label, the rest are features
        self.labels = data.iloc[:, 0].values
        self.features = data.iloc[:, 1:].values.astype('float32')
        
        # Encode labels into integers
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Get features and label
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label

    def get_label_encoder(self):
        return self.label_encoder

# Define the neural network model (MLP)
class HandGestureMLP(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_classes=24):
        super(HandGestureMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Check if GPU is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the dataset
dataset = HandLandmarksDataset('../resources/data/hand_landmarks_train.csv')

# Split into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize model, loss function, and optimizer
model = HandGestureMLP(num_classes=len(dataset.get_label_encoder().classes_)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for features, labels in train_loader:
        # Move data to device
        features, labels = features.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%')

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in val_loader:
            # Move data to device
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    print(f'Validation Accuracy: {val_accuracy:.2f}%')

# Save the LabelEncoder
label_encoder = dataset.get_label_encoder()
joblib.dump(label_encoder, f'../resources/models/label_encoder_hand_landmark_model_{model_version}.pkl')

# Save the trained model
torch.save(model.state_dict(), f'../resources/models/hand_landmark_model_{model_version}.pth')
