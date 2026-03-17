# scripts/run.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse
import torch

from PlaylistDataset import PlaylistDataset
from SimpleAudioCNN import SimpleAudioCNN, train_eval_test_save_model, predict_tracks
from evaluation import EvalConfig, run_kfold_evaluation
from artist_vocab import ArtistVocab


def _load_dataset(args):
    ds = PlaylistDataset.from_db(args.db)
    ds.setPartitionConfig(num_of_partitions=args.parts, partition_length=args.part_len)
    ds.cache_dir = args.mel_cache or None
    vocab = ArtistVocab(ds.trackList)
    ds.vocab = vocab
    return ds, vocab


def cmd_train(args) -> None:
    ds, vocab = _load_dataset(args)
    train_eval_test_save_model(ds, vocab, epochs=args.epochs, batch_size=args.batch_size, report_dir=args.report_dir)


def cmd_kfold(args) -> None:
    ds, vocab = _load_dataset(args)
    config = EvalConfig(
        epochs=args.epochs,
        n_splits=args.folds,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        random_seed=args.seed,
        recall_guard_min=args.recall_min,
        report_dir=args.report_dir,
    )
    run_kfold_evaluation(
        model_factory=lambda num_classes: SimpleAudioCNN(num_classes, vocab.size()),
        ds=ds,
        config=config,
        device=torch.device("cpu"),
    )


def cmd_predict(args) -> None:
    ds, vocab = _load_dataset(args)
    predict_tracks(args.model, ds, args.filter)


def _build_parser() -> argparse.ArgumentParser:
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--db",        default="./data/music.db")
    shared.add_argument("--parts",     type=int, default=5)
    shared.add_argument("--part-len",  type=int, default=10)
    shared.add_argument("--mel-cache", default="./data/mel_cache",
                        help="Directory for persisted mel spectrogram .npy files. Pass empty string to disable.")

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Music ML pipeline",
        parents=[shared],
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_train = sub.add_parser("train", parents=[shared])
    p_train.add_argument("--epochs",     type=int,   default=20)
    p_train.add_argument("--batch-size", type=int,   default=16)
    p_train.add_argument("--report-dir", default="out")
    p_train.set_defaults(func=cmd_train)

    p_kfold = sub.add_parser("kfold", parents=[shared])
    p_kfold.add_argument("--epochs",     type=int,   default=5)
    p_kfold.add_argument("--folds",      type=int,   default=3)
    p_kfold.add_argument("--batch-size", type=int,   default=16)
    p_kfold.add_argument("--lr",         type=float, default=1e-3)
    p_kfold.add_argument("--seed",       type=int,   default=42)
    p_kfold.add_argument("--recall-min", type=float, default=0.65)
    p_kfold.add_argument("--report-dir", default="out")
    p_kfold.set_defaults(func=cmd_kfold)

    p_pred = sub.add_parser("predict", parents=[shared])
    p_pred.add_argument("--model",  required=True)
    p_pred.add_argument("--filter", required=True)
    p_pred.set_defaults(func=cmd_predict)

    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    args.func(args)
