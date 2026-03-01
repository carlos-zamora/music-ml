from PlaylistDataset import PlaylistDataset
from evaluation import (
    EvalConfig,
    evaluate,
    get_metric_definitions,
    run_kfold_evaluation,
    train_epoch,
    write_eval_schema_json,
    write_eval_summary_json,
    write_per_playlist_csv,
)
from torch.utils.data import DataLoader, random_split
from datetime import datetime
from pathlib import Path
import os
import time
import torch
import torch.nn as nn
from load_mel import load_mel

class SimpleAudioCNN(nn.Module):
    # Defines a simple CNN for mel-spectrogram multi-label classification.
    # In:
    # - num_classes: number of playlist labels to predict.
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1))
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, num_classes))
    
    # Runs a forward pass through the network.
    # In:
    # - x: tensor shaped (batch, 1, mel_height, mel_width).
    # Out:
    # - logits shaped (batch, num_classes).
    def forward(self,x):
        return self.head(self.conv(x))
    

# Predicts playlist probabilities for one track.
# In:
# - trackJson: single track JSON object from tracks.json.
# - model: trained model instance.
# - labels: ordered playlist labels.
# Out:
# - sorted list of (playlist, probability), descending by probability.
def predict(trackJson, model, labels):
    model.eval()
    mel = load_mel(trackJson)
    x = torch.tensor(mel).unsqueeze(0).unsqueeze(0).to(torch.device("cpu"))

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    
    results = [(labels[i], float(p)) for i, p in enumerate(probs)]
    return sorted(results, key=lambda x: -x[1])

# Prints prediction scores in a simple table.
# In:
# - trackJSON: track metadata used for title/artist.
# - results: list of (playlist, probability) tuples.
def print_results_table(trackJSON, results):
    print(f"\nPrediction Results for: {trackJSON['name']}")
    print(f"\tBy: {trackJSON['artist']}")
    print("=" * 50)
    print(f"{'Playlist':<30} {'Probability':<12}")
    print("-" * 50)
    for playlist, prob in results:
        print(f"{playlist:<30} {prob:<12.3f}")
    print("=" * 50)

# Save the model with automatic filename fallback - model.pth, model_1.pth, model2.pth, etc.
# In:
# - model: trained model to save.
# - base_path: output directory.
# - filename: preferred output filename.
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

# Writes evaluation artifacts for a single split run.
# In:
# - labels: ordered playlist names.
# - val_details: validation detailed evaluation payload.
# - test_details: test detailed evaluation payload.
# - out_root: output root folder path.
# Out:
# - path to report directory.
def write_single_split_reports(labels, val_details, test_details, out_root='D:\\projects\\music-ml\\out\\eval'):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_single")
    out_dir = Path(out_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    write_eval_schema_json(out_dir / "summary.schema.json")

    summary = {
        "$schema": "./summary.schema.json",
        "mode": "single_split",
        "labels": labels,
        "metric_definitions": get_metric_definitions(),
        "validation": {
            "loss": val_details["loss"],
            "metrics": val_details["metrics"],
        },
        "test": {
            "loss": test_details["loss"],
            "metrics": test_details["metrics"],
        },
    }
    write_eval_summary_json(out_dir / "summary.json", summary)

    rows = []
    for row in val_details["per_playlist"]:
        csv_row = dict(row)
        csv_row["fold"] = "validation"
        rows.append(csv_row)
    for row in test_details["per_playlist"]:
        csv_row = dict(row)
        csv_row["fold"] = "test"
        rows.append(csv_row)
    write_per_playlist_csv(out_dir / "per_playlist_metrics.csv", rows)
    return str(out_dir)


# Splits dataset, trains model, evaluates test set, then saves model.
# In:
# - ds: dataset of track features and playlist labels.
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
    labels = ds.playlists()
    model = SimpleAudioCNN(len(labels)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # training
    EPOCHS = 20
    print("Starting training...")
    training_start_time = time.time()
    val_details = None
    
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device=device)
        val_details = evaluate(
            model,
            val_loader,
            criterion,
            labels=labels,
            thresholds=0.5,
            return_details=True,
            device=device,
        )
        val_loss = val_details["loss"]
        val_f1 = val_details["metrics"]["micro_f1"]
        val_macro_f1 = val_details["metrics"]["macro_f1"]
        
        epoch_time = time.time() - epoch_start_time
        ds.resetCount()
        
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Micro F1: {val_f1:.4f} | "
              f"Val Macro F1: {val_macro_f1:.4f} | "
              f"Time: {epoch_time:.2f}s")
    
    total_training_time = time.time() - training_start_time
    print(f"\nTraining completed in {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
    
    print("\nEvaluating on test set...")
    test_start_time = time.time()
    test_details = evaluate(
        model,
        test_loader,
        criterion,
        labels=labels,
        thresholds=0.5,
        return_details=True,
        device=device,
    )
    test_loss = test_details["loss"]
    test_f1 = test_details["metrics"]["micro_f1"]
    test_macro_f1 = test_details["metrics"]["macro_f1"]
    test_time = time.time() - test_start_time
    print(f"Test Loss: {test_loss:.4f}, Test Micro F1: {test_f1:.4f}, Test Macro F1: {test_macro_f1:.4f} | Test Time: {test_time:.2f}s")

    reports_dir = write_single_split_reports(labels, val_details, test_details)
    print(f"Evaluation reports saved to: {reports_dir}")
    
    # Save the trained model
    saved_filename = save(model)
    print(f"Training complete! Model saved as: {saved_filename}")

# Runs prediction for tracks matching a regex filter.
# In:
# - model_path: path to a saved model file.
# - ds: dataset for track search and labels.
# - track_filter: regex filter for track names.
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

    # Use this to train a new model with single train/val/test split
    # train_eval_test_save_model(ds)

    # Use this to run k-fold cross-validation with threshold tuning + recall guard.
    kfold_summary = run_kfold_evaluation(
        model_factory=lambda num_classes: SimpleAudioCNN(num_classes),
        ds=ds,
        config=EvalConfig(),
        device=torch.device("cpu"),
    )
    
    
