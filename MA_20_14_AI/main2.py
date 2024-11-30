import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
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
    try:
        data = pd.read_csv(file_path, delimiter=" ", header=None)
        print(f"Data loaded successfully. Shape: {data.shape}")
        print(data.head())  # Ellenőrizd az első néhány sort
        if data.shape[1] < 2:
            raise ValueError("The dataset does not contain enough columns (features and target).")
        X = data.iloc[:, :-1].values  # Az összes oszlop, kivéve az utolsót
        y = data.iloc[:, -1].values   # Az utolsó oszlop (célváltozó)
        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# 3. Adatok előfeldolgozása és oversampling
def preprocess_data(X, y, test_size=0.2):
    # Oversampling a kisebbségi osztályra
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    scaler = MinMaxScaler()  # Normalizálás [0,1] közé
    X_scaled = scaler.fit_transform(X_resampled)

    dataset = CustomDataset(X_scaled, y_resampled)
    test_size = int(len(dataset) * test_size)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset, scaler

# 4. Neurális háló definiálása
class SimpleFCNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleFCNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        return x

# 5. Tanítási ciklus korai megállítással
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs, patience, device):
    model.to(device)
    history = {'train_loss': [], 'val_loss': []}
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Edzési fázis
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Értékelési fázis
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        # Átlagos veszteség kiszámítása
        train_loss /= len(train_loader)
        val_loss /= len(test_loader)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

        # Korai megállítás
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

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

# 7. Modell mentése
def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)

# 8. Kiértékelés és osztályozási jelentés
def evaluate_model(model, test_loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze()
            y_true.extend(y_batch.numpy())
            y_pred.extend((outputs.cpu().numpy() > 0.5).astype(int))
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

# --- Fő futtatás ---
if __name__ == "__main__":
    # Fájl elérési útja
    file_path = "ma20_14_ai.txt"  # Cseréld ki a megfelelő fájl elérési úttal

    # Adatok betöltése és előfeldolgozása
    X, y = load_data(file_path)
    train_dataset, test_dataset, scaler = preprocess_data(X, y)

    # Adatbetöltők létrehozása
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Modell létrehozása
    input_dim = X.shape[1]
    model = SimpleFCNN(input_dim)

    # Súlyozott veszteségfüggvény az osztályok arányai alapján
    class_weights = torch.tensor([0.6, 1.0]).to('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.BCELoss()  # Súlyozott Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)

    # Tanítási paraméterek
    epochs = 50
    patience = 5  # Korai megállítási türelem
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Modell tanítása
    history = train_model(model, train_loader, test_loader, criterion, optimizer, epochs, patience, device)

    # Tanulási görbék ábrázolása
    plot_training(history)

    # Modell kiértékelése
    evaluate_model(model, test_loader, device)

    # Modell mentése
    save_model(model, "trained_model2.pth")
