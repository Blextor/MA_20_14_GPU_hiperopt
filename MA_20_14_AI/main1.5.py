import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 1. Egyedi PyTorch Dataset osztály az adatok kezelésére
class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# 2. Adatok betöltése fájlból
def load_data(file_path):
    data = pd.read_csv(file_path, delimiter=" ", header=None)
    print(f"Data loaded successfully. Shape: {data.shape}")
    if data.shape[1] < 2:
        raise ValueError("The dataset does not contain enough columns (features and target).")
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y

# 3. Adatok előfeldolgozása
def preprocess_data(X, y, test_size=0.2):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    dataset = CustomDataset(X_scaled, y)
    test_size = int(len(dataset) * test_size)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset, scaler

# 4. Neurális háló definiálása
class OverfittingFCNN(nn.Module):
    def __init__(self, input_dim):
        super(OverfittingFCNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.sigmoid(self.fc6(x))
        return x

# 5. Tanítási ciklus
def train_model(model, train_loader, test_loader, criterion, optimizer, scaler, epochs, device):
    model.to(device)
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            with torch.cuda.amp.autocast():
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(test_loader)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

    return history

# 6. Tanulási görbék plotolása
def plot_training(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

# 7. Modell kiértékelése (klasszifikációs riport és konfúziós mátrix)
def evaluate_model(model, test_loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze()
            y_true.extend(y_batch.numpy())
            y_pred.extend((outputs.cpu().numpy() > 0.5).astype(int))

    # Klasszifikációs riport
    report = classification_report(y_true, y_pred, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # Konfúziós mátrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Osztályonkénti darabszámok kiíratása
    tn, fp, fn, tp = cm.ravel()
    print(f"\nDetails:")
    print(f"True Negatives (0 -> 0): {tn}")
    print(f"False Positives (0 -> 1): {fp}")
    print(f"False Negatives (1 -> 0): {fn}")
    print(f"True Positives (1 -> 1): {tp}")

    # Összes helyes és helytelen predikció
    total_correct = tn + tp
    total_incorrect = fp + fn
    print(f"\nTotal Correct Predictions: {total_correct}")
    print(f"Total Incorrect Predictions: {total_incorrect}")

# --- Fő futtatás ---
if __name__ == "__main__":
    file_path = "ma20_14_ai_chk.txt"

    X, y = load_data(file_path)
    train_dataset, test_dataset, scaler = preprocess_data(X, y)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    input_dim = X.shape[1]
    model = OverfittingFCNN(input_dim)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.cuda.amp.GradScaler()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 50

    history = train_model(model, train_loader, test_loader, criterion, optimizer, scaler, epochs, device)

    plot_training(history)

    # Modell értékelése a plot után
    evaluate_model(model, test_loader, device)
