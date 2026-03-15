from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from terminal import TrainingMonitor, print_fold_epoch_row, print_metric_legend, print_section
import csv
import json
import numpy as np
import random
import torch
import torch.nn as nn

METRIC_DEFINITIONS = {
    "loss": {
        "description": "Model error from the loss function.",
        "range": ">= 0",
        "direction": "lower_is_better",
    },
    "micro_f1": {
        "description": "F1 across all label decisions pooled together.",
        "range": "0..1",
        "direction": "higher_is_better",
    },
    "macro_f1": {
        "description": "Average F1 across playlists, weighting each playlist equally.",
        "range": "0..1",
        "direction": "higher_is_better",
    },
    "micro_precision": {
        "description": "Fraction of predicted positives that were correct across all labels pooled together.",
        "range": "0..1",
        "direction": "higher_is_better",
    },
    "macro_precision": {
        "description": "Average precision across playlists.",
        "range": "0..1",
        "direction": "higher_is_better",
    },
    "micro_recall": {
        "description": "Fraction of true positives found across all labels pooled together.",
        "range": "0..1",
        "direction": "higher_is_better",
    },
    "macro_recall": {
        "description": "Average recall across playlists.",
        "range": "0..1",
        "direction": "higher_is_better",
    },
    "threshold": {
        "description": "Decision threshold used for a playlist. Lower predicts more often; higher predicts more conservatively.",
        "range": "typically 0..1",
        "direction": "context_dependent",
    },
    "support": {
        "description": "Number of ground-truth positive tracks for a playlist in the evaluated split.",
        "range": "integer >= 0",
        "direction": "informational",
    },
    "tp": {
        "description": "True positives: correct positive predictions.",
        "range": "integer >= 0",
        "direction": "higher_is_better",
    },
    "fp": {
        "description": "False positives: songs incorrectly predicted into the playlist.",
        "range": "integer >= 0",
        "direction": "lower_is_better",
    },
    "tn": {
        "description": "True negatives: correct negative predictions.",
        "range": "integer >= 0",
        "direction": "higher_is_better_but_less_important_for_imbalanced_classes",
    },
    "fn": {
        "description": "False negatives: songs that belonged in the playlist but were missed.",
        "range": "integer >= 0",
        "direction": "lower_is_better",
    },
}


@dataclass
class EvalConfig:
    # Configuration for evaluation and validation flows.
    # In:
    # - n_splits: number of cross-validation folds.
    # - random_seed: seed used for deterministic split behavior.
    # - batch_size: dataloader batch size.
    # - epochs: number of training epochs per fold.
    # - learning_rate: optimizer learning rate.
    # - val_fraction_within_fold: fraction of train_val used for validation.
    # - default_threshold: fallback threshold when tuning is not possible.
    # - threshold_min: minimum candidate threshold for tuning.
    # - threshold_max: maximum candidate threshold for tuning.
    # - threshold_step: step size for threshold tuning grid.
    # - recall_guard_min: minimum recall required for guarded playlists.
    # - report_dir: output directory for evaluation artifacts.
    n_splits: int = 3
    random_seed: int = 42
    batch_size: int = 16
    epochs: int = 5
    learning_rate: float = 1e-3
    val_fraction_within_fold: float = 0.1
    default_threshold: float = 0.5
    threshold_min: float = 0.05
    threshold_max: float = 0.95
    threshold_step: float = 0.01
    recall_guard_min: float = 0.65
    report_dir: str = "out/runs"


# Converts threshold input into an array of shape (num_classes,).
# In:
# - thresholds: either a scalar threshold or list of per-class thresholds.
# - num_classes: number of label classes.
# Out:
# - numpy array of thresholds per class.
def _to_threshold_array(thresholds, num_classes: int):
    if np.isscalar(thresholds):
        return np.full(num_classes, float(thresholds), dtype=np.float32)
    arr = np.asarray(thresholds, dtype=np.float32)
    if arr.shape[0] != num_classes:
        raise ValueError("Threshold length must match number of classes")
    return arr


# Flattens partitioned audio tensor batch into model input batch.
# In:
# - x: tensor with shape (batch_size, num_partitions, 1, mel_h, mel_w).
# Out:
# - tuple(flattened_x, batch_size, num_partitions).
def _reshape_partition_batch(x):
    batch_size, num_partitions = x.shape[0], x.shape[1]
    x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
    return x, batch_size, num_partitions


# Trains the model for one epoch.
# In:
# - model: model to train.
# - loader: training DataLoader.
# - optimizer: optimizer for parameter updates.
# - criterion: loss function.
# - device: torch device for execution.
# - on_batch: optional callable invoked after each batch (e.g. a progress bar advance fn).
# Out:
# - average training loss over all batches.
def train_epoch(model, loader, optimizer, criterion, device=None, on_batch=None):
    if device is None:
        device = torch.device("cpu")

    model.train()
    total_loss = 0.0
    for x, artist_idx, artist_mask, y in loader:
        x, batch_size, num_partitions = _reshape_partition_batch(x)
        x, artist_idx, artist_mask, y = x.to(device), artist_idx.to(device), artist_mask.to(device), y.to(device)

        optimizer.zero_grad()
        audio_feat = model.encode_audio(x)                                       # (batch*parts, 128)
        audio_feat = audio_feat.view(batch_size, num_partitions, -1).mean(dim=1) # (batch, 128)
        logits = model.classify(audio_feat, artist_idx, artist_mask)

        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if on_batch:
            on_batch()

    return total_loss / len(loader)


# Collects loss, true labels, and predicted probabilities from a loader.
# In:
# - model: model to evaluate.
# - loader: evaluation DataLoader.
# - criterion: loss function.
# - device: torch device for execution.
# Out:
# - tuple(avg_loss, y_true_np, y_prob_np).
def collect_outputs(model, loader, criterion, device=None):
    if device is None:
        device = torch.device("cpu")

    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, artist_idx, artist_mask, y in loader:
            x, batch_size, num_partitions = _reshape_partition_batch(x)
            x, artist_idx, artist_mask, y = x.to(device), artist_idx.to(device), artist_mask.to(device), y.to(device)

            audio_feat = model.encode_audio(x)                                       # (batch*parts, 128)
            audio_feat = audio_feat.view(batch_size, num_partitions, -1).mean(dim=1) # (batch, 128)
            logits = model.classify(audio_feat, artist_idx, artist_mask)
            loss = criterion(logits, y)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu())
            all_labels.append(y.cpu())

    y_prob = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_labels).numpy()
    avg_loss = total_loss / len(loader)
    return avg_loss, y_true, y_prob


# Converts predicted probabilities into binary decisions.
# In:
# - y_prob: array of predicted probabilities with shape (n_samples, n_classes).
# - thresholds: scalar or per-class thresholds.
# Out:
# - binary predictions array.
def predict_labels(y_prob, thresholds=0.5):
    threshold_array = _to_threshold_array(thresholds, y_prob.shape[1])
    return (y_prob >= threshold_array).astype(np.float32)


# Computes micro/macro metrics for multi-label predictions.
# In:
# - y_true: binary ground-truth labels.
# - y_pred: binary predictions.
# Out:
# - dictionary with micro/macro precision, recall, and f1.
def compute_multilabel_metrics(y_true, y_pred):
    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


# Computes per-playlist confusion counts and derived metrics.
# In:
# - y_true: binary ground-truth labels.
# - y_pred: binary predictions.
# - labels: ordered playlist names.
# - thresholds: scalar or per-class thresholds used for prediction.
# Out:
# - list of dictionaries with per-playlist metrics.
def compute_per_playlist_confusion(y_true, y_pred, labels, thresholds=0.5):
    threshold_array = _to_threshold_array(thresholds, y_true.shape[1])
    rows = []
    for i, label in enumerate(labels):
        true_col = y_true[:, i].astype(np.int32)
        pred_col = y_pred[:, i].astype(np.int32)

        tp = int(np.sum((true_col == 1) & (pred_col == 1)))
        fp = int(np.sum((true_col == 0) & (pred_col == 1)))
        tn = int(np.sum((true_col == 0) & (pred_col == 0)))
        fn = int(np.sum((true_col == 1) & (pred_col == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        rows.append({
            "playlist": label,
            "support": int(np.sum(true_col)),
            "threshold": float(threshold_array[i]),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        })
    return rows


# Tunes per-playlist thresholds using validation set class-wise F1.
# In:
# - y_true: validation binary labels.
# - y_prob: validation probabilities.
# - default_threshold: fallback threshold for classes without positives.
# - t_min: minimum threshold candidate.
# - t_max: maximum threshold candidate.
# - t_step: threshold step.
# Out:
# - list of per-playlist thresholds.
def tune_thresholds(y_true, y_prob, default_threshold=0.5, t_min=0.05, t_max=0.95, t_step=0.01):
    thresholds = []
    candidates = np.arange(t_min, t_max + 1e-9, t_step)
    for i in range(y_true.shape[1]):
        true_col = y_true[:, i]
        prob_col = y_prob[:, i]

        if np.sum(true_col) == 0:
            thresholds.append(float(default_threshold))
            continue

        best_threshold = float(default_threshold)
        best_f1 = -1.0
        best_distance = float("inf")
        for threshold in candidates:
            pred_col = (prob_col >= threshold).astype(np.float32)
            score = f1_score(true_col, pred_col, average="binary", zero_division=0)
            distance = abs(float(threshold) - 0.5)

            if score > best_f1 or (score == best_f1 and distance < best_distance):
                best_f1 = score
                best_threshold = float(threshold)
                best_distance = distance
        thresholds.append(best_threshold)
    return thresholds


# Evaluates a model on a loader and returns legacy or detailed metrics.
# In:
# - model: model to evaluate.
# - loader: evaluation DataLoader.
# - criterion: loss function.
# - labels: optional ordered playlist labels.
# - thresholds: scalar or per-class thresholds.
# - return_details: whether to return detailed metrics instead of tuple.
# - device: torch device for execution.
# Out:
# - when return_details=False: tuple(loss, micro_f1)
# - when return_details=True: dictionary with detailed metrics.
def evaluate(
    model,
    loader,
    criterion,
    labels=None,
    thresholds=0.5,
    return_details=False,
    device=None,
):
    loss, y_true, y_prob = collect_outputs(model, loader, criterion, device=device)
    y_pred = predict_labels(y_prob, thresholds=thresholds)
    metrics = compute_multilabel_metrics(y_true, y_pred)

    if not return_details:
        return loss, metrics["micro_f1"]

    per_playlist = []
    if labels is not None:
        per_playlist = compute_per_playlist_confusion(
            y_true, y_pred, labels, thresholds=thresholds
        )

    return {
        "loss": float(loss),
        "metrics": metrics,
        "per_playlist": per_playlist,
        "y_true": y_true,
        "y_prob": y_prob,
        "thresholds": list(_to_threshold_array(thresholds, y_true.shape[1])),
    }


# Writes summary JSON for one evaluation run.
# In:
# - path: output JSON file path.
# - summary_dict: summary payload to serialize.
def write_eval_summary_json(path, summary_dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)


# Returns metric definitions used in JSON output for human-readable browsing.
# Out:
# - dictionary of metric metadata.
def get_metric_definitions():
    return dict(METRIC_DEFINITIONS)


# Writes a JSON Schema file describing summary.json fields.
# In:
# - path: output schema file path.
def write_eval_schema_json(path):
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Music ML Evaluation Summary",
        "type": "object",
        "description": "Schema for summary.json generated by this project.",
        "properties": {
            "$schema": {
                "type": "string",
                "description": "Relative path to this schema file.",
            },
            "mode": {
                "type": "string",
                "description": "Run mode, such as single_split.",
            },
            "labels": {
                "type": "array",
                "description": "Ordered playlist labels used by the model.",
                "items": {"type": "string"},
            },
            "metric_definitions": {
                "type": "object",
                "description": "Human-readable explanations for metric fields.",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "range": {"type": "string"},
                        "direction": {"type": "string"},
                    },
                    "required": ["description", "range", "direction"],
                    "additionalProperties": False,
                },
            },
            "validation": {
                "type": "object",
                "description": "Validation summary for single-split or per-fold selected epoch.",
                "properties": {
                    "loss": {
                        "type": "number",
                        "description": "Model loss. Lower is better.",
                    },
                    "metrics": {
                        "$ref": "#/$defs/metricBundle",
                    },
                },
                "additionalProperties": True,
            },
            "test": {
                "type": "object",
                "description": "Test summary.",
                "properties": {
                    "loss": {
                        "type": "number",
                        "description": "Model loss. Lower is better.",
                    },
                    "metrics": {
                        "$ref": "#/$defs/metricBundle",
                    },
                },
                "additionalProperties": True,
            },
            "config": {
                "type": "object",
                "description": "Evaluation configuration for k-fold runs.",
                "additionalProperties": True,
            },
            "guarded_playlists": {
                "type": "array",
                "description": "Playlists protected by the recall guard.",
                "items": {"type": "string"},
            },
            "folds": {
                "type": "array",
                "description": "Per-fold k-fold evaluation summaries.",
                "items": {
                    "type": "object",
                    "additionalProperties": True,
                },
            },
            "aggregate_metrics": {
                "type": "object",
                "description": "Aggregate mean/std metrics across folds.",
                "additionalProperties": {
                    "type": "number",
                },
            },
        },
        "$defs": {
            "metricBundle": {
                "type": "object",
                "properties": {
                    "micro_f1": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": METRIC_DEFINITIONS["micro_f1"]["description"],
                    },
                    "macro_f1": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": METRIC_DEFINITIONS["macro_f1"]["description"],
                    },
                    "micro_precision": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": METRIC_DEFINITIONS["micro_precision"]["description"],
                    },
                    "macro_precision": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": METRIC_DEFINITIONS["macro_precision"]["description"],
                    },
                    "micro_recall": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": METRIC_DEFINITIONS["micro_recall"]["description"],
                    },
                    "macro_recall": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": METRIC_DEFINITIONS["macro_recall"]["description"],
                    },
                },
                "additionalProperties": True,
            },
        },
        "additionalProperties": True,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)


# Writes per-playlist rows to CSV.
# In:
# - path: output CSV file path.
# - per_playlist_rows: list of per-playlist metric dictionaries.
def write_per_playlist_csv(path, per_playlist_rows):
    if len(per_playlist_rows) == 0:
        headers = ["playlist", "fold", "support", "threshold", "precision", "recall", "f1", "tp", "fp", "tn", "fn"]
    else:
        headers = list(per_playlist_rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in per_playlist_rows:
            writer.writerow(row)


# Builds deterministic train and validation index lists from a source index list.
# In:
# - indices: source indices to split.
# - val_fraction: fraction assigned to validation.
# - seed: random seed for deterministic shuffle.
# Out:
# - tuple(train_indices, val_indices).
def split_train_val_indices(indices, val_fraction, seed):
    indices = list(indices)
    rng = random.Random(seed)
    rng.shuffle(indices)

    val_size = max(1, int(len(indices) * val_fraction))
    if val_size >= len(indices):
        val_size = max(1, len(indices) - 1)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices


# Gets top-K playlists by positive support counts.
# In:
# - ds: PlaylistDataset instance.
# - top_k: number of playlists to return.
# Out:
# - list of playlist names sorted by descending support.
def get_top_playlists_by_support(ds, top_k=5):
    counts = {}
    for playlist in ds.playlists():
        counts[playlist] = 0
    for track in ds.trackList:
        for playlist in (p.name for p in track.playlists):
            if playlist in counts:
                counts[playlist] += 1
    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in sorted_items[:top_k]]


# Checks if recall guard is satisfied for target playlists.
# In:
# - per_playlist_rows: per-playlist metrics rows.
# - guarded_playlists: playlist names that must satisfy recall floor.
# - recall_min: minimum recall threshold.
# Out:
# - tuple(guard_satisfied, recall_by_playlist).
def check_recall_guard(per_playlist_rows, guarded_playlists, recall_min):
    recall_by_playlist = {}
    for row in per_playlist_rows:
        if row["playlist"] in guarded_playlists:
            recall_by_playlist[row["playlist"]] = float(row["recall"])

    guard_satisfied = True
    for playlist in guarded_playlists:
        recall = recall_by_playlist.get(playlist, 0.0)
        if recall < recall_min:
            guard_satisfied = False
    return guard_satisfied, recall_by_playlist


# Trains one fold and returns best model state and validation diagnostics.
# In:
# - model_factory: callable(num_classes) -> torch model.
# - ds: PlaylistDataset instance.
# - train_indices: fold training indices.
# - val_indices: fold validation indices.
# - labels: ordered playlist names.
# - guarded_playlists: playlists used for recall guard checks.
# - config: EvalConfig object.
# - device: torch device for execution.
# - collate_fn: optional DataLoader collate function (e.g. from ArtistVocab.make_collate_fn).
# - fold_index: 1-based fold number used for display output.
# Out:
# - dictionary containing best state, thresholds, and validation metrics.
def train_one_fold(
    model_factory,
    ds,
    train_indices,
    val_indices,
    labels,
    guarded_playlists,
    config,
    device=None,
    collate_fn=None,
    fold_index: int = 1,
):
    if device is None:
        device = torch.device("cpu")

    train_loader = DataLoader(Subset(ds, train_indices), batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(Subset(ds, val_indices), batch_size=config.batch_size, collate_fn=collate_fn)

    model = model_factory(len(labels)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_state_dict = None
    best_thresholds = [config.default_threshold for _ in labels]
    best_result = None
    monitor = TrainingMonitor(config.epochs)

    for epoch in range(config.epochs):
        with monitor.epoch(epoch + 1, len(train_loader), f"Fold {fold_index} · Epoch {epoch + 1}/{config.epochs}") as advance:
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device=device, on_batch=advance)

        val_loss, val_true, val_prob = collect_outputs(
            model, val_loader, criterion, device=device
        )
        tuned_thresholds = tune_thresholds(
            val_true,
            val_prob,
            default_threshold=config.default_threshold,
            t_min=config.threshold_min,
            t_max=config.threshold_max,
            t_step=config.threshold_step,
        )
        val_pred = predict_labels(val_prob, thresholds=tuned_thresholds)
        val_metrics = compute_multilabel_metrics(val_true, val_pred)
        val_per_playlist = compute_per_playlist_confusion(
            val_true, val_pred, labels, thresholds=tuned_thresholds
        )
        guard_satisfied, guard_recalls = check_recall_guard(
            val_per_playlist, guarded_playlists, config.recall_guard_min
        )


        ds.resetCount()
        candidate = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "metrics": val_metrics,
            "per_playlist": val_per_playlist,
            "guard_satisfied": guard_satisfied,
            "guard_recalls": guard_recalls,
        }
        monitor.log_fold_epoch(
            fold_index, epoch + 1,
            float(train_loss), float(val_loss),
            val_metrics["micro_f1"], val_metrics["macro_f1"],
            guard_satisfied,
        )

        # Selection: guard-satisfying candidates first, then macro F1.
        if best_result is None:
            choose = True
        else:
            if candidate["guard_satisfied"] and not best_result["guard_satisfied"]:
                choose = True
            elif candidate["guard_satisfied"] == best_result["guard_satisfied"]:
                choose = candidate["metrics"]["macro_f1"] > best_result["metrics"]["macro_f1"]
            else:
                choose = False

        if choose:
            best_result = candidate
            best_thresholds = tuned_thresholds
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    model.load_state_dict(best_state_dict)
    return {
        "model": model,
        "thresholds": best_thresholds,
        "best_validation": best_result,
    }


# Runs K-Fold evaluation with per-playlist thresholds and recall guard.
# In:
# - model_factory: callable(num_classes) -> torch model.
# - ds: PlaylistDataset instance.
# - config: EvalConfig with CV and metric settings.
# - device: torch device for execution.
# - collate_fn: optional DataLoader collate function; auto-built from ds.vocab if omitted.
# Out:
# - summary dictionary containing fold and aggregate metrics.
def run_kfold_evaluation(model_factory, ds, config=None, device=None, collate_fn=None):
    if config is None:
        config = EvalConfig()
    if device is None:
        device = torch.device("cpu")

    if collate_fn is None and hasattr(ds, 'vocab') and ds.vocab is not None:
        collate_fn = ds.vocab.make_collate_fn()

    labels = ds.playlists()
    indices = np.arange(len(ds))
    kfold = KFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.random_seed,
    )

    guarded_playlists = get_top_playlists_by_support(ds, top_k=5)
    fold_summaries = []
    per_playlist_rows = []

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(config.report_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print_metric_legend(include_guard=True)

    for fold_index, (train_val_idx, test_idx) in enumerate(kfold.split(indices), start=1):
        train_idx, val_idx = split_train_val_indices(
            train_val_idx,
            val_fraction=config.val_fraction_within_fold,
            seed=config.random_seed + fold_index,
        )

        print_section(f"Fold {fold_index} / {config.n_splits}")
        fold_train = train_one_fold(
            model_factory=model_factory,
            ds=ds,
            train_indices=train_idx,
            val_indices=val_idx,
            labels=labels,
            guarded_playlists=guarded_playlists,
            config=config,
            device=device,
            collate_fn=collate_fn,
            fold_index=fold_index,
        )

        criterion = nn.BCEWithLogitsLoss()
        test_loader = DataLoader(Subset(ds, test_idx), batch_size=config.batch_size, collate_fn=collate_fn)
        test_eval = evaluate(
            fold_train["model"],
            test_loader,
            criterion,
            labels=labels,
            thresholds=fold_train["thresholds"],
            return_details=True,
            device=device,
        )

        fold_summary = {
            "fold": fold_index,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
            "thresholds": fold_train["thresholds"],
            "validation": fold_train["best_validation"],
            "test": {
                "loss": test_eval["loss"],
                "metrics": test_eval["metrics"],
            },
        }
        fold_summaries.append(fold_summary)

        for row in test_eval["per_playlist"]:
            csv_row = dict(row)
            csv_row["fold"] = fold_index
            per_playlist_rows.append(csv_row)

    aggregate = {}
    for key in ["micro_f1", "macro_f1", "micro_precision", "macro_precision", "micro_recall", "macro_recall"]:
        values = [fold["test"]["metrics"][key] for fold in fold_summaries]
        aggregate[f"{key}_mean"] = float(np.mean(values))
        aggregate[f"{key}_std"] = float(np.std(values))

    aggregate_rows = []
    for label in labels:
        label_rows = [row for row in per_playlist_rows if row["playlist"] == label]
        tp = sum(row["tp"] for row in label_rows)
        fp = sum(row["fp"] for row in label_rows)
        tn = sum(row["tn"] for row in label_rows)
        fn = sum(row["fn"] for row in label_rows)
        support = sum(row["support"] for row in label_rows)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        aggregate_rows.append({
            "playlist": label,
            "fold": "aggregate",
            "support": int(support),
            "threshold": float(np.mean([row["threshold"] for row in label_rows])) if len(label_rows) > 0 else config.default_threshold,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        })

    summary = {
        "$schema": "./summary.schema.json",
        "config": {
            "n_splits": config.n_splits,
            "random_seed": config.random_seed,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "val_fraction_within_fold": config.val_fraction_within_fold,
            "default_threshold": config.default_threshold,
            "threshold_min": config.threshold_min,
            "threshold_max": config.threshold_max,
            "threshold_step": config.threshold_step,
            "recall_guard_min": config.recall_guard_min,
            "report_dir": str(out_dir),
        },
        "metric_definitions": get_metric_definitions(),
        "guarded_playlists": guarded_playlists,
        "folds": fold_summaries,
        "aggregate_metrics": aggregate,
    }

    write_eval_schema_json(out_dir / "summary.schema.json")
    write_eval_summary_json(out_dir / "summary.json", summary)
    write_per_playlist_csv(out_dir / "per_playlist_metrics.csv", per_playlist_rows + aggregate_rows)
    return summary
