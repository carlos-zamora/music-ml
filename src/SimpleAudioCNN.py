import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score
import time
import os
from PlaylistDataset import PlaylistDataset
from load_mel import load_mel

class SimpleAudioCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1))
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, num_classes))
    
    def forward(self,x):
        return self.head(self.conv(x))
    

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch_idx, (x, y) in enumerate(loader):
        # x shape: (batch_size, num_partitions, 1, height, width)
        # reshape to: (batch_size * num_partitions, 1, height, width)
        batch_size, num_partitions = x.shape[0], x.shape[1]
        x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])

        x, y = x.to(torch.device("cpu")), y.to(torch.device("cpu"))

        optimizer.zero_grad()
        logits = model(x) # Shape: (batch_size * num_partitions, num_classes)

        logits = logits.view(batch_size, num_partitions, -1).mean(dim=1) # Shape: (batch_size, num_classes)

        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Clean up memory and perform garbage collection every 10 batches
        del x, y, logits, loss
        if batch_idx % 10 == 0:
            import gc
            gc.collect()

    return total_loss / len(loader)

def evaluate(model ,loader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            # x shape: (batch_size, num_partitions, 1, height, width)
            # reshape to: (batch_size * num_partitions, 1, height, width)
            batch_size, num_partitions = x.shape[0], x.shape[1]
            x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])

            x, y = x.to(torch.device("cpu")), y.to(torch.device("cpu"))

            logits = model(x) # Shape: (batch_size * num_partitions, num_classes)

            # Reshape back and average across partitions
            logits = logits.view(batch_size, num_partitions, -1).mean(dim=1)  # Shape: (batch_size, num_classes)

            loss = criterion(logits, y)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    f1 = f1_score(all_labels, all_preds, average="micro")
    return total_loss / len(loader), f1

def predict(trackJson, model, labels):
    model.eval()
    mel = load_mel(trackJson)
    x = torch.tensor(mel).unsqueeze(0).unsqueeze(0).to(torch.device("cpu"))

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    
    results = [(labels[i], float(p)) for i, p in enumerate(probs)]
    return sorted(results, key=lambda x: -x[1])

def print_results_table(trackJSON, results):
    """Print prediction results in a simple table format"""
    print(f"\nPrediction Results for: {trackJSON['name']}")
    print(f"\tBy: {trackJSON['artist']}")
    print("=" * 50)
    print(f"{'Playlist':<30} {'Probability':<12}")
    print("-" * 50)
    for playlist, prob in results:
        print(f"{playlist:<30} {prob:<12.3f}")
    print("=" * 50)

# Save the model with automatic filename fallback - model.pth, model_1.pth, model2.pth, etc.
# Out:
# - the filename
def save(model, base_path='D:\\projects\\music-ml\\out', filename='model.pth'):
    # Extract name and extension
    name, ext = os.path.splitext(filename)
    
    # Try the original filename first
    full_path = os.path.join(base_path, filename)
    if not os.path.exists(full_path):
        torch.save(model, full_path)
        print(f"Model saved as: {filename}")
        return filename
    
    # Try numbered versions
    counter = 1
    while True:
        numbered_filename = f"{name}_{counter}{ext}"
        full_path = os.path.join(base_path, numbered_filename)
        
        if not os.path.exists(full_path):
            torch.save(model, full_path)
            print(f"Model saved as: {numbered_filename}")
            return numbered_filename
        counter += 1

def train_eval_test_save_model(ds:PlaylistDataset):
    # Train: 80%
    # Validate: 10%
    # Test: 10%
    train_size = int(0.8 * len(ds))
    val_size = int(0.1 * len(ds))
    test_size = len(ds) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(ds, [train_size, val_size, test_size])
    print(f"Train Size: {train_size} | "
          f"Val Size: {val_size} | "
          f"Test Size: {test_size}")

    batch_size = 16
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # prep model
    device = torch.device("cpu")
    model = SimpleAudioCNN(len(ds.playlists())).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # training
    EPOCHS = 20
    print("Starting training...")
    training_start_time = time.time()
    
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_f1 = evaluate(model, val_loader, criterion)
        
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val F1: {val_f1:.4f} | "
              f"Time: {epoch_time:.2f}s")
    
    total_training_time = time.time() - training_start_time
    print(f"\nTraining completed in {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
    
    print("\nEvaluating on test set...")
    test_start_time = time.time()
    test_loss, test_f1 = evaluate(model, test_loader, criterion)
    test_time = time.time() - test_start_time
    print(f"Test Loss: {test_loss:.4f}, Test F1: {test_f1:.4f} | Test Time: {test_time:.2f}s")
    
    # Save the trained model
    saved_filename = save(model)
    print(f"Training complete! Model saved as: {saved_filename}")

def predict_tracks(model_path:str, ds:PlaylistDataset, track_filter:str):
    tracksDB = ds.find_tracks(track_filter)
    model = torch.load(model_path, weights_only=False)
    for trackJSON in tracksDB:
        result = predict(trackJSON, model, ds.playlists())
        print_results_table(trackJSON, result)

if __name__ == "__main__":
    ds = PlaylistDataset.from_json('D:\\projects\\music-ml\\out\\tracks.json')
    ds.setPartitionConfig(num_of_partitions=5,
                          partition_length=10)
    
    # Use this to predict a set of tracks using 
    #   model_path = 'D:\\projects\\music-ml\\out\\model.pth'
    #   track_filter = "Riddim"
    #   predict_tracks(model_path, ds, track_filter)

    # Use this to train a new model
    train_eval_test_save_model(ds)
    
    