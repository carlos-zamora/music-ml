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
import time
import torch
import torch.nn as nn
from load_mel import load_mel

class SimpleAudioCNN(nn.Module):
    # Defines a CNN for mel-spectrogram multi-label classification with artist embedding.
    # In:
    # - num_classes: number of playlist labels to predict.
    # - num_artists: artist vocab size (including OOV at index 0).
    # - artist_embed_dim: dimensionality of the artist embedding.
    def __init__(self, num_classes, num_artists, artist_embed_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1))
        )
        self.artist_embed = nn.Embedding(num_artists, artist_embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(128 + artist_embed_dim, num_classes)

    # Encodes audio through the CNN backbone.
    # In:
    # - x: tensor shaped (batch, 1, mel_height, mel_width).
    # Out:
    # - audio features shaped (batch, 128).
    def encode_audio(self, x):
        return self.conv(x).flatten(1)

    # Fuses audio features with artist embedding(s) and returns logits.
    # In:
    # - audio_feat: tensor (batch, 128).
    # - artist_idx: tensor shaped (batch, max_artists); use (batch, 1) for single-artist.
    # - artist_mask: bool tensor (batch, max_artists) or None when artist_idx is (batch, 1).
    # Out:
    # - logits shaped (batch, num_classes).
    def classify(self, audio_feat, artist_idx, artist_mask=None):
        emb = self.artist_embed(artist_idx)          # (batch, max_artists, 32) or (batch, 32)
        if artist_mask is not None:
            emb = emb * artist_mask.unsqueeze(-1).float()
            counts = artist_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
            emb = emb.sum(dim=1) / counts            # masked mean → (batch, 32)
        else:
            emb = emb.squeeze(1)                     # (batch, 1, 32) → (batch, 32)
        fused = torch.cat([audio_feat, emb], dim=1)
        return self.classifier(self.dropout(fused))

    # Runs a forward pass for the single-partition predict path.
    # In:
    # - x: tensor shaped (batch, 1, mel_height, mel_width).
    # - artist_idx: tensor shaped (batch,).
    # Out:
    # - logits shaped (batch, num_classes).
    def forward(self, x, artist_idx):
        return self.classify(self.encode_audio(x), artist_idx.unsqueeze(1))


# Predicts playlist probabilities for one track.
# In:
# - track: Track ORM object.
# - model: trained model instance.
# - labels: ordered playlist labels.
# - vocab: ArtistVocab instance for artist encoding.
# Out:
# - sorted list of (playlist, probability), descending by probability.
def predict(track, model, labels, vocab):
    model.eval()
    mel = load_mel(track)
    x = torch.tensor(mel).unsqueeze(0).unsqueeze(0).to(torch.device("cpu"))
    artist_idx = torch.tensor([vocab.encode(track.artist)], dtype=torch.long)

    with torch.no_grad():
        logits = model(x, artist_idx)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    results = [(labels[i], float(p)) for i, p in enumerate(probs)]
    return sorted(results, key=lambda x: -x[1])

# Prints prediction scores in a simple table.
# In:
# - track: Track ORM object used for title/artist.
# - results: list of (playlist, probability) tuples.
def print_results_table(track, results):
    print(f"\nPrediction Results for: {track.title}")
    print(f"\tBy: {track.artist}")
    print("=" * 50)
    print(f"{'Playlist':<30} {'Probability':<12}")
    print("-" * 50)
    for playlist, prob in results:
        print(f"{playlist:<30} {prob:<12.3f}")
    print("=" * 50)

# Saves the model to out_dir/model.pth.
# In:
# - model: trained model to save.
# - out_dir: directory to save the model into (must already exist).
def save(model, out_dir: Path):
    path = out_dir / "model.pth"
    torch.save(model, path)
    print(f"Model saved as: {path}")

# Writes evaluation artifacts for a single split run.
# In:
# - labels: ordered playlist names.
# - val_details: validation detailed evaluation payload.
# - test_details: test detailed evaluation payload.
# - out_dir: pre-created output directory.
def write_single_split_reports(labels, val_details, test_details, out_dir: Path):
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


# Splits dataset, trains model, evaluates test set, then saves model.
# In:
# - ds: dataset of track features and playlist labels.
# - vocab: ArtistVocab instance for artist encoding.
# - epochs: number of training epochs.
# - batch_size: dataloader batch size.
# - report_dir: output directory for evaluation artifacts.
def train_eval_test_save_model(ds:PlaylistDataset, vocab, epochs:int=20, batch_size:int=16, report_dir:str='out/runs'):
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

    collate_fn = vocab.make_collate_fn()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, collate_fn=collate_fn)

    # prep model
    device = torch.device("cpu")
    labels = ds.playlists()
    model = SimpleAudioCNN(len(labels), vocab.size()).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # training
    print("Starting training...")
    training_start_time = time.time()
    val_details = None

    for epoch in range(epochs):
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

        print(f"Epoch {epoch+1}/{epochs} | "
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

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_single")
    out_dir = Path(report_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    write_single_split_reports(labels, val_details, test_details, out_dir)
    print(f"Evaluation reports saved to: {out_dir}")

    save(model, out_dir)
    print(f"Model saved to: {out_dir / 'model.pth'}")

# Runs prediction for tracks matching a regex filter.
# In:
# - model_path: path to a saved model file.
# - ds: dataset for track search and labels.
# - track_filter: regex filter for track names.
def predict_tracks(model_path:str, ds:PlaylistDataset, track_filter:str):
    tracksDB = ds.find_tracks(track_filter)
    model = torch.load(model_path, weights_only=False)
    for track in tracksDB:
        result = predict(track, model, ds.playlists(), ds.vocab)
        print_results_table(track, result)
