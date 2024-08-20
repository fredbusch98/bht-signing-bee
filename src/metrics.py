import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Define the list of letter labels
letters = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y'
]

# Define the neural network model (MLP) for hand landmarks
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

# Load the dataset
test_df = pd.read_csv('hand_landmarks_test.csv')

# Split the data into features and labels
X_test = test_df.iloc[:, 1:].values.astype(np.float32)
y_test = test_df.iloc[:, 0].values

# Load the LabelEncoder
import joblib
label_encoder = joblib.load('../resources/models/label_encoder_hand_landmark_model_1.pkl')
num_classes = len(label_encoder.classes_)

# Define a custom Dataset class
class HandLandmarksDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Create the dataset and data loaders
test_dataset = HandLandmarksDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load model and switch it to evaluation mode
model = HandGestureMLP(input_size=X_test.shape[1], num_classes=num_classes)
model.load_state_dict(torch.load(f'../resources/models/hand_landmark_model_1.pth', map_location=torch.device('cpu')))
model.eval()

all_labels = []
all_predictions = []
all_probabilities = []

device = torch.device('cpu')
model.to(device)

correct = 0
total = 0

# Disable gradient computation for evaluation
with torch.no_grad():
    for data in test_loader:
        features, labels = data
        features, labels = features.to(device), labels.to(device)
        
        # Get model outputs
        outputs = model(features)
        
        # Convert outputs to predicted labels (multiclass classification)
        _, preds = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()
        
        # Append predictions and true labels to lists
        all_predictions.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Append outputs to get probability scores for ROC
        all_probabilities.extend(F.softmax(outputs, dim=1).cpu().numpy())

# Calculate precision, recall, and F1-score using sklearn
precision = precision_score(all_labels, all_predictions, average='macro')
recall = recall_score(all_labels, all_predictions, average='macro')
f1 = f1_score(all_labels, all_predictions, average='macro')
accuracy = correct / total

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')

# Calculate the confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

# Display and save the confusion matrix with letter labels
plt.figure(figsize=(15, 12))  # Increase the figure size
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=letters)
disp.plot(cmap=plt.cm.Blues, ax=plt.gca(), xticks_rotation='vertical')  # Rotate x-axis labels if needed
plt.title('Confusion Matrix')

# Adjust font sizes
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)

plt.savefig('confusion_matrix.jpg', bbox_inches='tight')
plt.show()

# Binarize the labels for ROC calculation
y_test_bin = label_binarize(all_labels, classes=range(num_classes))
y_score = np.array(all_probabilities)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curve with letter labels
plt.figure(figsize=(15, 12))  # Increase the figure size
plt.plot(fpr["micro"], tpr["micro"], label='Micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]), color='navy', lw=2)
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'darkgreen', 'red', 'purple', 'brown', 'pink', 'cyan', 'orange'])
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(letters[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Receiver Operating Characteristic (Multi-class)', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True)  # Add grid lines for better readability
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.savefig('roc_auc_curve.jpg', bbox_inches='tight')
plt.show()
