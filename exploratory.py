import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

# Simple Neural Network Using Squared Log Returns to Predict Volatility
# Squard Log Returns are a ccnsistent (converges to theta) measure of volatility 

# Create a custom dataset for PyTorch
class VolatilityDataset(Dataset):
    def __init__(self, features, target):
        self.features = features
        self.target = target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]

#Neural Network
import torch.nn as nn
class VolatilityNN(nn.Module):
    def __init__(self, input_size):
        super(VolatilityNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Output log_volatility)
        )
    def forward(self, x):
        return self.network(x)

data = pd.read_csv('./processed_data/bitcoin_processed.csv')

data['log_return'] = np.log(data['close'] / data['close'].shift(1))
data['squared_log_return'] = data['log_return'] ** 2

# Thus we use rolling volatility?
data['rolling_volatility'] = np.sqrt(data['squared_log_return'].rolling(window=8).sum())
data['log_volatility'] = np.log(data['rolling_volatility'])

data.dropna(inplace=True)

#Features are predictor, Target is reponse
features = data[['squared_log_return']].values
target = data['log_volatility'].values

# Split data into training and testing sets
train_size = int(len(features) * 0.8)  # Use 80% for training, 20% for testing
X_train, X_test = features[:train_size], features[train_size:]
y_train, y_test = target[:train_size], target[train_size:]

#Standardize Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Convert to Tensors (Need to read up on why this is necessary)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = VolatilityDataset(X_train_tensor, y_train_tensor)
test_dataset = VolatilityDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

input_size = X_train.shape[1]
model = VolatilityNN(input_size)
# Loss Function we want to minimize
loss_f = nn.MSELoss()
# Need to read up on this
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#Training
epochs = 100
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for features, target in train_loader:
        optimizer.zero_grad()
        outputs = model(features).squeeze()  # Remove extra dimension
        # Use MSE Loss for Regression
        loss = loss_f(outputs,target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader)}")

#Evaluate Model
from sklearn.metrics import mean_squared_error
with torch.no_grad():
    predictions = []
    actuals = []
    for features, targets in test_loader:
        outputs = model(features).squeeze()
        predictions.append(outputs.numpy())
        actuals.append(targets.numpy())

# Flatten lists and compute metrics
predictions = np.concatenate(predictions)
actuals = np.concatenate(actuals)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(actuals, predictions)
results = pd.DataFrame({
    'actual': actuals,
    'prediction': predictions
})
results.to_csv('./results/first_try.csv')
print(f"Test MSE: {mse}")