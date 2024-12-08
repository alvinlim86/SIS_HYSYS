import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Sample DataFrame
data = {
    'feature1': [1, 2, 3, 4],
    'feature2': [5, 6, 7, 8],
    'label': [0, 1, 0, 1]
}
df = pd.DataFrame(data)

# Convert DataFrame to PyTorch Tensor
# Excluding the label column for features tensor
features_tensor = torch.tensor(df[['feature1', 'feature2']].values, dtype=torch.float32)
labels_tensor = torch.tensor(df['label'].values, dtype=torch.float32)

print("Features Tensor:", features_tensor)
print("Labels Tensor:", labels_tensor)


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        #self.flatten = nn.Flatten()
        #self.linear_relu_stack = nn.Sequential(nn.Linear(64*64, 512), nn.ReLU(), nn.Linear(512, 512),nn.ReLU)
        self.fc1 = nn.Linear(2, 10)  # Two input features, 10 neurons in the first layer
        self.fc2 = nn.Linear(10, 1)  # One output neuron

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

model = SimpleNet()

criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent optimizer

# Assuming features_tensor and labels_tensor 
num_epochs = 100

for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(features_tensor)
    loss = criterion(outputs, labels_tensor.unsqueeze(1))
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Sample DataFrame
data2 = {
    'feature1': [5, 6, 7, 8],
    'feature2': [9, 10, 11, 12],
    'label': [0, 1, 0, 1]
}
df2 = pd.DataFrame(data2)
features_tensor2 = torch.tensor(df2[['feature1', 'feature2']].values, dtype=torch.float32)
with torch.no_grad():
    predictions2 = model(features_tensor2)
    predictions2 = torch.round(predictions2)  # Round predictions to 0 or 1 for binary classification

list_pred2 = [pred.item() for pred in predictions2.numpy()]

print("Predictions2:", list_pred2)


import pandas as pd
import numpy as np

# Generating a linear dataset
np.random.seed(0)  # For reproducibility
data = {
    'feature1': np.linspace(0, 10, 100),
    'feature2': np.linspace(0, 20, 100),
    'target': np.linspace(0, 10, 100) * 2 + np.random.normal(0, 1, 100)  # Linear relationship with some noise
}
df = pd.DataFrame(data)

print(df.head())


# Convert DataFrame to PyTorch Tensor
features_tensor = torch.tensor(df[['feature1', 'feature2']].values, dtype=torch.float32)
target_tensor = torch.tensor(df['target'].values, dtype=torch.float32)

class RegressionNet(nn.Module):
    def __init__(self):
        super(RegressionNet, self).__init__()
        self.fc1 = nn.Linear(2, 10)  # Two input features, 10 neurons in the first layer
        self.fc2 = nn.Linear(10, 1)  # One output neuron

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = RegressionNet()

criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent optimizer

num_epochs = 100

for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(features_tensor)
    loss = criterion(outputs, target_tensor.unsqueeze(1))
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    predictions = model(features_tensor)

print("Predictions_Linear:", predictions)
