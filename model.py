import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

MODEL_PATH = "models/gesture_cnn.pth"
INPUT_SIZE = 63  # 21 landmarks * 3
NUM_CLASSES = 0  # will be set dynamically

class GestureCNN(nn.Module):
    def __init__(self, num_classes):
        super(GestureCNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(INPUT_SIZE, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

def train_model(samples, labels, num_epochs=50):
    """samples: list of lists (63 floats each), labels: list of ints"""
    if len(set(labels)) < 2:
        return None  # need at least 2 classes

    num_classes = len(set(labels))
    model = GestureCNN(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X = torch.tensor(samples, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    model.train()
    for epoch in range(num_epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    return model

def load_model(num_classes):
    model = GestureCNN(num_classes)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

def predict(landmarks, model, gesture_names):
    """landmarks: list of 63 floats, returns predicted gesture name"""
    with torch.no_grad():
        x = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0)
        outputs = model(x)
        pred_idx = torch.argmax(outputs, dim=1).item()
        return gesture_names[pred_idx]