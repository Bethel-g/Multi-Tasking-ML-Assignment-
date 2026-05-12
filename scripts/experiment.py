import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# 1. DATA GENERATION (Synthetic Student Performance)
def generate_student_data(n_samples=500):
    X = np.random.rand(n_samples, 10)
    y_reg = 100 * (0.4 * X[:,0] + 0.3 * X[:,1] + 0.2 * X[:,2] + 0.1 * X[:,3]) + np.random.normal(0, 5, n_samples)
    y_reg = np.clip(y_reg, 0, 100)
    y_cls = (y_reg > 50).astype(int)
    return X, y_reg, y_cls

X, y_reg, y_cls = generate_student_data()
X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
    X, y_reg, y_cls, train_size=40, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train_t = torch.FloatTensor(X_train)
X_test_t = torch.FloatTensor(X_test)
y_reg_train_t = torch.FloatTensor(y_reg_train).view(-1, 1)
y_reg_test_t = torch.FloatTensor(y_reg_test).view(-1, 1)
y_cls_train_t = torch.FloatTensor(y_cls_train).view(-1, 1)
y_cls_test_t = torch.FloatTensor(y_cls_test).view(-1, 1)

# 2. MODEL ARCHITECTURES (Matching user suggestion)

# A. STL Model (Overfitted Baseline)
class STLModel(nn.Module):
    def __init__(self):
        super(STLModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x): return self.net(x)

# B. MTL Model (Regularized)
class MTLModel(nn.Module):
    def __init__(self):
        super(MTLModel, self).__init__()
        # Shared Backbone
        self.shared = nn.Sequential(
            nn.Linear(10, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU()
        )
        # Regression Head
        self.reg_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        # Classification Head
        self.cls_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, x):
        shared_out = self.shared(x)
        return self.reg_head(shared_out), self.cls_head(shared_out)

# 3. TRAINING UTILS
def train_stl(model, epochs=300):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    history = {'train_loss': [], 'test_loss': []}
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_reg_train_t)
        loss.backward(); optimizer.step()
        history['train_loss'].append(loss.item())
        with torch.no_grad():
            model.eval()
            test_loss = criterion(model(X_test_t), y_reg_test_t)
            history['test_loss'].append(test_loss.item())
    return history

def train_mtl(model, epochs=300):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion_reg = nn.MSELoss()
    criterion_cls = nn.BCELoss()
    history = {'train_reg_loss': [], 'test_reg_loss': [], 'train_cls_acc': [], 'test_cls_acc': []}
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        reg_pred, cls_pred = model(X_train_t)
        loss = criterion_reg(reg_pred, y_reg_train_t) + 10.0 * criterion_cls(cls_pred, y_cls_train_t)
        loss.backward(); optimizer.step()
        
        history['train_reg_loss'].append(criterion_reg(reg_pred, y_reg_train_t).item())
        with torch.no_grad():
            model.eval()
            r, c = model(X_test_t)
            history['test_reg_loss'].append(criterion_reg(r, y_reg_test_t).item())
    return history

# 4. EXECUTION
print("--- Training STL ---")
stl = STLModel(); stl_hist = train_stl(stl)
print("--- Training MTL ---")
mtl = MTLModel(); mtl_hist = train_mtl(mtl)

# Save Results
with torch.no_grad():
    stl.eval(); mtl.eval()
    s_p = stl(X_test_t).numpy()
    m_r_p, m_c_p = mtl(X_test_t)
    m_r_p = m_r_p.numpy(); m_c_p = m_c_p.numpy()
    m_c_b = (m_c_p > 0.5).astype(int)

results = {
    'stl': {
        'metrics': {'mse': mean_squared_error(y_reg_test, s_p), 'r2': r2_score(y_reg_test, s_p)},
        'history': {'train_mse': stl_hist['train_loss'], 'test_mse': stl_hist['test_loss']}
    },
    'mtl': {
        'metrics': {
            'mse': mean_squared_error(y_reg_test, m_r_p), 
            'r2': r2_score(y_reg_test, m_r_p),
            'accuracy': accuracy_score(y_cls_test, m_c_b),
            'f1': f1_score(y_cls_test, m_c_b)
        },
        'history': {'train_mse': mtl_hist['train_reg_loss'], 'test_mse': mtl_hist['test_reg_loss']}
    }
}

# Fix for JSON serialization
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        return json.JSONEncoder.default(self, obj)

os.makedirs('report', exist_ok=True)
with open('report/results.json', 'w') as f:
    json.dump(results, f, indent=4, cls=NumpyEncoder)

# Save Models
os.makedirs('models', exist_ok=True)
torch.save(stl.state_dict(), 'models/stl_model.pth')
torch.save(mtl.state_dict(), 'models/mtl_model.pth')

# Visualizations
os.makedirs('plots', exist_ok=True)
plt.figure(figsize=(10, 5))
plt.plot(stl_hist['test_loss'], label='STL Test MSE (High Variance)', color='red')
plt.plot(mtl_hist['test_reg_loss'], label='MTL Test MSE (Regularized)', color='green')
plt.title('MTL Regularization vs STL Overfitting'); plt.yscale('log'); plt.legend(); plt.savefig('plots/comparison.png')

print("Experiment Complete. Results in report/results.json.")
