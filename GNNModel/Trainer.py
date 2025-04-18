import wandb
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from GMCDataset import get_data_loaders
from GNNModel import get_gnn_model
import time
import torch.nn.functional as F
import csv


# Initialize wandb
wandb.init(project='gnn_malware')  # Set entity and project name as needed


# Focal Loss Implementation
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, num_classes=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Probability of the correct class
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batched_graph, labels, paths in train_loader:
        # Move graph and labels to GPU
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)

        features = batched_graph.ndata['feature']
        logits = model(batched_graph, features)

        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss


def evaluate_model(model, test_loader, device, csv_writer=None):
    """
    Evaluate the model's performance on the test set.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        test_loader (torch.utils.data.DataLoader): DataLoader for test data.
        device (torch.device): Device for computation (CPU or GPU).
        csv_writer (Optional[csv.DictWriter], optional): CSV writer for logging misclassified samples.

    Returns:
        Dict[str, Any]: Dictionary containing evaluation metrics and misclassified data.
    """
    model.eval()
    all_preds = []
    all_labels = []
    misclassified_data = []

    with torch.no_grad():
        for batched_graph, labels, paths in test_loader:
            # Move graph and labels to the specified device
            batched_graph = batched_graph.to(device)
            labels = labels.to(device)

            # Extract features and perform forward pass
            features = batched_graph.ndata['feature']
            logits = model(batched_graph, features)
        
            # Compute prediction probabilities (Softmax output)
            pred_probabilities = torch.softmax(logits, dim=1)

            # Use argmax to get predicted classes
            preds = torch.argmax(pred_probabilities, dim=1)

            # Collect predictions and true labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Identify and log misclassified samples
            misclassified_mask = preds != labels
            misclassified_indices = misclassified_mask.nonzero(as_tuple=False).squeeze()

            # Handle cases where there is only one misclassified sample
            if misclassified_indices.dim() == 0:
                misclassified_indices = misclassified_indices.unsqueeze(0)

            for idx in misclassified_indices:
                pred_label = preds[idx].item()
                true_label = labels[idx].item()
                prediction_probability = pred_probabilities[idx][pred_label].item()
                deviation = abs(prediction_probability - (1.0 if pred_label == 1 else 0.0))
                error_type = 'FN' if pred_label == 0 else 'FP'

                misclassified_data.append({
                    'path': paths[idx], 
                    'true_label': true_label, 
                    'pred_label': pred_label,
                    'prediction_probability': prediction_probability,
                    'deviation': deviation,
                    'error_type': error_type
                })

    # Calculate evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    # Calculate confusion matrix and extract FPR and FNR
    try:
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    except ValueError:
        # If class imbalance occurs, the confusion matrix may not be flattened
        tn = fp = fn = tp = 0
        fpr = fnr = 0

    # Write misclassified data to CSV file
    if csv_writer:
        for misclassified in misclassified_data:
            csv_writer.writerow(misclassified)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr,
        'fnr': fnr,
        'misclassified_data': misclassified_data
    }


def print_metrics(epoch, metrics):
    print(f"Test Metrics after Epoch {epoch}:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print(f"FPR: {metrics['fpr']:.4f}, FNR: {metrics['fnr']:.4f}")


def save_checkpoint(epoch, model, optimizer, loss, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}.")


def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from epoch {epoch}.")
    return epoch, loss


if __name__ == "__main__":
    # Configure model, data loader, loss function, etc.
    input_dim = 768
    hidden_dim = 128
    num_classes = 2
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    model_type = 'DotGAT'
    model = get_gnn_model(model_type=model_type, input_dim=768, hidden_dim=128, num_classes=2, num_heads=8)
    model = model.to(device)

    # Focal Loss with class weights
    # criterion = FocalLoss(gamma=2, alpha=0.25)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    parent_dir = "/home/wei/GMC_family_data"
    batch_size = 512
    train_ratio = 0.1
    train_loader, test_loader = get_data_loaders(batch_size=batch_size, train_ratio=train_ratio)

    checkpoint_path = '/home/wei/android-malware-detection-master/model_checkpoint_DotGAT.pth'
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path)

    # Create a CSV file to log misclassified data
    with open('/home/wei/android-malware-detection-master/notebooks/misclassified_data_gat.csv', mode='w', newline='') as file:
        fieldnames = ['path', 'true_label', 'pred_label', 'prediction_probability', 'deviation', 'error_type']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()  # Write header row

        # Training loop
        epochs = 300
        print("Training started")
        start_time = time.time()
        for epoch in range(start_epoch, epochs):
            total_loss = train_one_epoch(model, train_loader, criterion, optimizer, device=device)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

            # Evaluate and log metrics every 10 epochs
            if (epoch + 1) % 10 == 0:
                metrics = evaluate_model(model, test_loader, device, csv_writer=writer)
                print_metrics(epoch + 1, metrics)

                # Log training loss and evaluation metrics to wandb
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': total_loss,
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1'],
                    'fpr': metrics['fpr'],
                    'fnr': metrics['fnr'],
                    'model_type': model_type
                })

            # Save model checkpoint every epoch
            save_checkpoint(epoch + 1, model, optimizer, total_loss, checkpoint_path)

        end_time = time.time()
        print(f"Program execution time: {end_time - start_time}")
        torch.save(model.state_dict(), '/home/wei/android-malware-detection-master/Model_GraphDotGAT.pth')
        print("Model saved!")
        wandb.finish()  # Complete the current wandb experiment