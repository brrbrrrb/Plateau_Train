"""
Plateau Classification with GPU Support
=======================================

Enhanced version with GPU acceleration for Apple Silicon (M1/M2) Macs
using PyTorch for neural networks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path

class PyTorchNeuralNet(nn.Module):
    """PyTorch neural network for GPU acceleration on Apple Silicon."""
    
    def __init__(self, input_size, hidden_sizes=[100, 50], n_classes=3):
        super(PyTorchNeuralNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, n_classes))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

def check_gpu_availability():
    """Check and report GPU availability."""
    print("\nGPU Availability Check:")
    print("-" * 40)
    
    # Check for CUDA (NVIDIA GPUs)
    if torch.cuda.is_available():
        print(f"CUDA available: Yes")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
    # Check for MPS (Apple Silicon)
    elif torch.backends.mps.is_available():
        print(f"Apple Silicon GPU (MPS) available: Yes")
        device = torch.device("mps")
    else:
        print(f"No GPU available, using CPU")
        device = torch.device("cpu")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"Selected device: {device}")
    print("-" * 40)
    
    return device

def train_pytorch_model(X_train, y_train, X_val, y_val, device, epochs=50):
    """Train PyTorch model with progress tracking."""
    input_size = X_train.shape[1]
    model = PyTorchNeuralNet(input_size).to(device)
    
    # Convert to tensors (ensure numpy arrays first)
    X_train_t = torch.FloatTensor(np.array(X_train)).to(device)
    y_train_t = torch.LongTensor(np.array(y_train)).to(device)
    X_val_t = torch.FloatTensor(np.array(X_val)).to(device)
    y_val_t = torch.LongTensor(np.array(y_val)).to(device)
    
    # Create data loader
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    # Training loop with progress bar
    from tqdm import tqdm
    
    best_val_acc = 0
    for epoch in tqdm(range(epochs), desc="Training Neural Network"):
        model.train()
        train_loss = 0
        
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t)
            _, val_preds = torch.max(val_outputs, 1)
            val_acc = (val_preds == y_val_t).float().mean().item()
        
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
        
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch+1}: Train Loss: {train_loss/len(loader):.4f}, Val Acc: {val_acc:.4f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model, best_val_acc

def run_gpu_demo():
    """Demo GPU-accelerated training."""
    device = check_gpu_availability()
    
    # Create models directory
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    
    # Load and prepare sample data
    print("\nPreparing sample data...")
    from plateau_classification import PlateauClassifier
    
    classifier = PlateauClassifier(models_dir="./models")
    classifier.load_data()
    classifier.prepare_features()
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        classifier.X_train_scaled, 
        classifier.y_train, 
        test_size=0.2, 
        random_state=42
    )
    
    print(f"\nTraining on device: {device}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Train model
    model, val_acc = train_pytorch_model(X_train, y_train, X_val, y_val, device)
    print(f"\nBest validation accuracy: {val_acc:.4f}")
    
    # Save PyTorch model
    pytorch_model_path = models_dir / 'pytorch_neural_net.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': X_train.shape[1],
        'validation_accuracy': val_acc,
        'device': str(device)
    }, pytorch_model_path)
    print(f"PyTorch model saved to: {pytorch_model_path}")
    
    # Save model metadata
    metadata = {
        'model_type': 'PyTorch Neural Network',
        'device': str(device),
        'validation_accuracy': val_acc,
        'input_features': X_train.shape[1],
        'training_samples': len(X_train),
        'validation_samples': len(X_val)
    }
    
    import json
    metadata_path = models_dir / 'pytorch_model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Model metadata saved to: {metadata_path}")
    
    return model, device

if __name__ == "__main__":
    model, device = run_gpu_demo()
    print(f"\nTraining completed on {device}!") 