import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from config import Config
from model import EfficientNetModel, CustomCNN
from dataset import create_dataloader

def train_model():
    # Set device to GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load training and testing datasets, and map class names to indices
    train_dataset, test_dataset, class_to_idx = create_dataloader()
    
    # Map class indices back to class names (for inference)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    print(f"Found {num_classes} classes")
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # Set up data loaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=Config.batchsize, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.batchsize, shuffle=False)

    # Initialize model based on configuration
    if Config.model_name == 'EfficientNetB0':
        model = EfficientNetModel(num_classes).to(device)
    elif Config.model_name == 'CustomCNN':
        model = CustomCNN(num_classes).to(device)
    else:
        raise NotImplementedError("Model not implemented")

    # Set loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    # Variables for tracking performance during training
    best_acc = 0
    no_improve = 0

    # Create directory for saving model checkpoints
    os.makedirs(os.path.dirname(Config.save_path), exist_ok=True)

    # Start training loop
    for epoch in range(Config.epochs):
        model.train()  # Set the model to training mode
        running_loss = 0
        correct = 0
        total = 0

        # Loop through training data
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.epochs}", leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)  # Move data to GPU/CPU
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update model parameters

            # Track loss and accuracy
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)  # Get the predicted class
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Update progress bar with loss and accuracy
            loop.set_postfix(loss=loss.item(), acc=100.*correct/total)

        # Calculate training accuracy for the epoch
        train_acc = 100 * correct / total

        # Start evaluation (validation) phase
        model.eval()  # Set the model to evaluation mode
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        # Disable gradient calculation for validation
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())  # Store predictions for final report
                all_labels.extend(labels.cpu().numpy())  # Store actual labels

        # Calculate validation accuracy
        val_acc = 100 * val_correct / val_total
        scheduler.step(val_acc)  # Adjust learning rate if needed

        # Print epoch results
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # Save the best model (if validation accuracy improves)
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), Config.save_path)  # Save the model
            print(f"Best model saved with val_acc: {val_acc:.2f}%")
        else:
            no_improve += 1
            if no_improve >= Config.patience:  # Early stopping condition
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load the best model after training
    model.load_state_dict(torch.load(Config.save_path))

    #print("\nClassification Report:")
    #target_names = [idx_to_class[i] for i in range(num_classes)]
    #print(classification_report(all_labels, all_preds, target_names=target_names))

    return model, idx_to_class
