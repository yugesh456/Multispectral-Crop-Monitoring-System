import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import AgriScan components
from core.classifier import AgriScanXception
from core.simulator import AgriScanDataset

def train_model(epochs=3, batch_size=32, dataset_size=1000, val_size=200, lr=0.001, save_path="agriscan_xception.pth"):
    print("=" * 60)
    print("         AgriScan AI - PyTorch Training Harness")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device.type.upper()}")
    
    # 1. Initialize Datasets and Loaders
    print(f"[*] Generating procedural multispectral datasets...")
    train_dataset = AgriScanDataset(size=dataset_size, train=True)
    val_dataset = AgriScanDataset(size=val_size, train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"    - Train size: {len(train_dataset)} samples")
    print(f"    - Val size: {len(val_dataset)} samples")
    
    # 2. Instantiate Model, Loss, Optimizer
    model = AgriScanXception(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 3. Training Loop
    print("\n[*] Starting optimization loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Epoch Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()
            
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{100.0 * correct_train / total_train:.2f}%"})
            
        scheduler.step()
        
        epoch_loss = running_loss / len(train_dataset)
        train_acc = 100.0 * correct_train / total_train
        
        # 4. Validation step
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += targets.size(0)
                correct_val += (predicted == targets).sum().item()
                
        epoch_val_loss = val_loss / len(val_dataset)
        val_acc = 100.0 * correct_val / total_val
        
        print(f" -> Summary Epoch {epoch}: Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
    # 5. Save Checkpoint
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\n[+] Model parameters saved successfully to: {os.path.abspath(save_path)}")
    print("=" * 60)
    
    return model, val_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgriScan AI PyTorch Model Trainer")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--train-size", type=int, default=300, help="Training dataset size")
    parser.add_argument("--val-size", type=int, default=100, help="Validation dataset size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output", type=str, default="weights/agriscan_xception.pth", help="Checkpoint save path")
    
    args = parser.parse_args()
    
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        dataset_size=args.train_size,
        val_size=args.val_size,
        lr=args.lr,
        save_path=args.output
    )
