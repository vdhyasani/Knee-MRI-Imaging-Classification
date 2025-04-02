import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import cv2

# Define the CNN Model first
class MRICNN(nn.Module):
    def __init__(self, input_channels=1):
        super(MRICNN, self).__init__()
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            
            # Second conv block
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            
            # Third conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
            # Fourth conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 2)  # 2 classes
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Custom Dataset class
class MRIDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_and_process_data(input_dataset, method='middle_slice'):
    data_list = []
    filenames = []
    
    # Path verification
    print(f"\nChecking paths...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Looking for data in: {input_dataset}")
    
    if not os.path.exists(input_dataset):
        raise FileNotFoundError(f"Data directory not found: {input_dataset}")
    
    print(f"\nDirectory contents: {os.listdir(input_dataset)}")
    
    # Sort the folders to process them in order
    folders = sorted([f for f in os.listdir(input_dataset) if f.lower().startswith('vol')])
    print(f"Found volume folders: {folders}")
    
    if not folders:
        raise ValueError(f"No volume folders found in {input_dataset}. "
                        f"Expected folders starting with 'vol'")
    
    for data_folder in folders:
        subfolder_path = os.path.join(input_dataset, data_folder)
        if not os.path.isdir(subfolder_path):
            print(f"Skipping {data_folder} - not a directory")
            continue
            
        print(f"\nProcessing {data_folder}...")
        
        # List contents of the volume folder
        folder_contents = os.listdir(subfolder_path)
        print(f"Contents of {data_folder}: {folder_contents}")
        
        pck_files = [f for f in folder_contents if f.endswith('.pck')]
        print(f"Found {len(pck_files)} .pck files in {data_folder}")
        
        for filename in pck_files:
            file_path = os.path.join(subfolder_path, filename)
            try:
                with open(file_path, 'rb') as file:
                    data = pickle.load(file)
                    print(f"Loaded {data_folder}/{filename} with shape: {np.shape(data)}")
                    
                    # Take the middle slice
                    middle_idx = data.shape[0] // 2
                    middle_slice = data[middle_idx]
                    
                    # Standardize size if needed
                    if middle_slice.shape != (320, 320):
                        middle_slice = cv2.resize(middle_slice, (320, 320))
                    
                    data_list.append(middle_slice)
                    filenames.append(f"{data_folder}/{filename}")
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
    
    if not data_list:
        raise ValueError("No data was loaded. Please check the data directory structure.")
    
    print("\nPre-processing data...")
    print(f"Number of samples loaded: {len(data_list)}")
    
    # Stack all slices
    data = np.stack(data_list)
    print(f"Shape after stacking: {data.shape}")
    
    # Normalize data
    data = (data - data.mean()) / (data.std() + 1e-8)
    
    # Reshape for CNN (add channel dimension)
    X = data.reshape(data.shape[0], 1, data.shape[1], data.shape[2])
    print(f"Final shape for CNN input: {X.shape}")
    
    # Create labels based on alcDiagnosis (replace this with actual logic)
    y = np.array([int(filename.split('-')[-1].split('.')[0]) % 2 for filename in filenames])
    print(f"Created labels with shape: {y.shape}")
    
    return X, y

# Training function
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=50):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Print epoch statistics
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}')
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            print(f'Test Accuracy: {accuracy:.2f}%')
            model.train()

# Main execution
# Main execution
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Use absolute path
    input_dataset = './datasets'
    print(f"Loading data from: {input_dataset}")
    X, y = load_and_process_data(input_dataset, method='middle_slice')
    
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create datasets
    train_dataset = MRIDataset(X_train, y_train)
    test_dataset = MRIDataset(X_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)
    
    # Initialize model, loss, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MRICNN(input_channels=1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    train_model(model, train_loader, test_loader, criterion, optimizer)
    
    # Final evaluation
    model.eval()
    y_pred = []
    y_true = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.numpy())
    
    # Print final metrics
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
