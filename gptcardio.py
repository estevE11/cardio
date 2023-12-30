from tqdm import trange, tqdm
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

device = torch.device("cpu")

# Load your data
data = pd.read_csv('dataset/train.csv')  # Replace with your actual file path

# Preprocess the data
X = data.iloc[:, :-1].values  # Features (all columns except the last one)
y = data.iloc[:, -1].values   # Labels (last column)

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
y_test = torch.tensor(y_test, dtype=torch.float32, device=device)

# Create data loaders
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 11, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.sigmoid(self.fc2(x))
        return x

model = CNN().to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

ts = time.monotonic()
# Training loop
for epoch in range(10):  # Number of epochs
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.squeeze(), target)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output.squeeze(), target).item()
            pred = output.round()
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'Epoch: {epoch}, Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.3f}%)')

print(f"Training took {time.monotonic() - ts:.2f} seconds")

def save_result(result, model):
    submission = pd.read_csv('dataset/sample.csv')
    submission['cardio'] = result

    i = 0
    while os.path.exists(f"submission_{i}.csv"):
        i += 1
    submission.to_csv(f'submission_{i}.csv', index = False)
    torch.save(model.state_dict(), f"cardio_{i}.pth")

# Save submittable model
sub_data = pd.read_csv('dataset/test.csv')
sub = scaler.fit_transform(sub_data.values)
sub = torch.tensor(sub, dtype=torch.float32)
y_fake = torch.tensor(np.random.randint(2, size=sub.shape[0]), dtype=torch.float32)
sub_loader = DataLoader(TensorDataset(sub, y_fake), batch_size=1, shuffle=False)

preds = []
model.eval()
for data, target in sub_loader:
    out = model(data)
    preds.append(int(out.round().item()))

preds = np.array(preds)
save_result(preds, model)