from contextlib import contextmanager
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
import time

_console = Console()


# Emits an OSC 1337 SetMark escape sequence so the terminal (e.g. Windows Terminal,
# iTerm2) inserts a navigable mark at this point in the scrollback buffer.
# Only writes when stdout is a real TTY; silently skips when output is redirected.
def _set_mark() -> None:
    if _console.is_terminal:
        _console.file.write("\x1b]1337;SetMark\x07")
        _console.file.flush()


# Formats a duration as a human-readable string.
# In:
# - seconds: duration in seconds.
# Out:
# - string like "~45s", "~1m 23s", or "~1h 05m".
def _fmt_eta(seconds: float) -> str:
    if seconds < 60:
        return f"~{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"~{m}m {s:02d}s"
    else:
        h, rem = divmod(int(seconds), 3600)
        return f"~{h}h {rem // 60:02d}m"


# Prints a section header rule and sets a terminal mark.
# In:
# - title: section title to display.
def print_section(title: str) -> None:
    _set_mark()
    _console.rule(f"[bold]{title}[/bold]")


# Prints dataset split sizes.
# In:
# - train, val, test: split sizes.
def print_split_sizes(train: int, val: int, test: int) -> None:
    _console.print(f"Split  train [cyan]{train}[/cyan]  val [cyan]{val}[/cyan]  test [cyan]{test}[/cyan]")


# Prints batch counts and batch size for the training run.
# In:
# - train_batches, val_batches: number of batches per loader.
# - batch_size: samples per batch.
def print_batch_info(train_batches: int, val_batches: int, batch_size: int) -> None:
    _console.print(
        f"Batches  train [cyan]{train_batches}[/cyan]"
        f"  val [cyan]{val_batches}[/cyan]"
        f"  batch size [cyan]{batch_size}[/cyan]"
    )


# Context manager that displays a transient per-epoch batch progress bar.
# Yields an advance() callable; call it once per completed batch.
# The bar disappears when the context exits, leaving the scrollback clean.
# In:
# - total_batches: total number of batches in the epoch.
# - label: description shown in the progress bar (e.g. "Epoch 3/20").
@contextmanager
def epoch_progress(total_batches: int, label: str = "Training"):
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        transient=True,
        console=_console,
    ) as progress:
        task = progress.add_task(label, total=total_batches)
        yield lambda: progress.advance(task)


# Prints a single track-loading row with fixed-width alignment.
# Elapsed time is suppressed ("--") for cached tracks since it is sub-millisecond noise.
# In:
# - count: current item count (1-based).
# - total: total items in the dataset.
# - title: track title (truncated to 32 chars with ellipsis if longer).
# - artist: track artist (truncated to 18 chars with ellipsis if longer).
# - bpm: track BPM (float or None).
# - key: Camelot key string (e.g. "7A") or None.
# - elapsed: load time in seconds; ignored when cached is True.
# - cached: whether the mel was served from the in-memory cache.
def print_track_row(count: int, total: int, title: str, artist: str,
                    bpm: float | None, key: str | None,
                    elapsed: float, cached: bool) -> None:
    title_str  = (title[:31]  + "…") if len(title)  > 32 else title
    artist_str = (artist[:17] + "…") if len(artist) > 18 else artist
    bpm_str    = f"{int(round(bpm)):>3}" if bpm else "---"
    key_str    = f"{key:<3}" if key else " --"
    time_str   = "    --" if cached else f"{elapsed:>6.2f}s"
    cache_str  = "  [dim]cached[/dim]" if cached else ""
    _console.print(
        f"[dim][{count:>{len(str(total))}}/{total}][/dim]"
        f"  ♪ {title_str:<32}"
        f"  [cyan]{artist_str:<18}[/cyan]"
        f"  ♩ [yellow]{bpm_str}[/yellow]"
        f"  [magenta]{key_str}[/magenta]"
        f"  {time_str}{cache_str}"
    )


# Prints one training epoch summary row and sets a terminal mark.
# In:
# - epoch: current epoch (1-based).
# - total_epochs: total number of epochs.
# - train_loss, val_loss: training and validation BCE losses.
# - micro_f1, macro_f1: validation F1 scores.
# - elapsed: wall time for this epoch in seconds.
# - total_elapsed: optional cumulative time since training started.
# - eta_seconds: optional estimated time remaining for all remaining epochs.
def print_epoch_row(epoch: int, total_epochs: int, train_loss: float,
                    val_loss: float, micro_f1: float, macro_f1: float,
                    elapsed: float, total_elapsed: float = None,
                    eta_seconds: float = None) -> None:
    total_str = f"  [dim]∑ {total_elapsed:.1f}s[/dim]" if total_elapsed is not None else ""
    eta_str = f"  [dim]ETA {_fmt_eta(eta_seconds)}[/dim]" if eta_seconds is not None else ""
    _set_mark()
    _console.print(
        f"Epoch {epoch:>2}/{total_epochs}"
        f"  train [yellow]{train_loss:.4f}[/yellow]"
        f"  val [yellow]{val_loss:.4f}[/yellow]"
        f"  \u00b5F1 [green]{micro_f1:.4f}[/green]"
        f"  MF1 [green]{macro_f1:.4f}[/green]"
        f"  {elapsed:.1f}s{total_str}{eta_str}"
    )


# Prints one k-fold epoch summary row and sets a terminal mark.
# In:
# - fold: current fold index (1-based).
# - epoch: current epoch (1-based).
# - total_epochs: total number of epochs per fold.
# - train_loss, val_loss: training and validation BCE losses.
# - micro_f1, macro_f1: validation F1 scores.
# - guard_ok: whether the per-playlist recall guard was satisfied this epoch.
# - elapsed: optional wall time for this epoch in seconds.
# - eta_seconds: optional estimated time remaining for all remaining epochs in this fold.
def print_fold_epoch_row(fold: int, epoch: int, total_epochs: int,
                         train_loss: float, val_loss: float,
                         micro_f1: float, macro_f1: float, guard_ok: bool,
                         elapsed: float = None, eta_seconds: float = None) -> None:
    guard_str = "[green]\u2713[/green]" if guard_ok else "[red]\u2717[/red]"
    time_str = f"  {elapsed:.1f}s" if elapsed is not None else ""
    eta_str = f"  [dim]ETA {_fmt_eta(eta_seconds)}[/dim]" if eta_seconds is not None else ""
    _set_mark()
    _console.print(
        f"Fold {fold}  Epoch {epoch:>2}/{total_epochs}"
        f"  train [yellow]{train_loss:.4f}[/yellow]"
        f"  val [yellow]{val_loss:.4f}[/yellow]"
        f"  \u00b5F1 [green]{micro_f1:.4f}[/green]"
        f"  MF1 [green]{macro_f1:.4f}[/green]"
        f"  guard {guard_str}{time_str}{eta_str}"
    )


# Prints training completion summary and sets a terminal mark.
# In:
# - elapsed_seconds: total wall time from training start to finish.
def print_training_complete(elapsed_seconds: float) -> None:
    _set_mark()
    _console.print(
        f"Training complete  [bold]{elapsed_seconds:.2f}s[/bold] ({elapsed_seconds / 60:.2f} min)"
    )


# Prints test evaluation results.
# In:
# - loss: test BCE loss.
# - micro_f1, macro_f1: test F1 scores.
# - elapsed: evaluation wall time in seconds.
def print_test_results(loss: float, micro_f1: float, macro_f1: float, elapsed: float) -> None:
    _console.print(
        f"Test  loss [yellow]{loss:.4f}[/yellow]"
        f"  \u00b5F1 [green]{micro_f1:.4f}[/green]"
        f"  MF1 [green]{macro_f1:.4f}[/green]"
        f"  {elapsed:.1f}s"
    )


# Prints a labeled artifact path in dim styling.
# In:
# - label: display label (e.g. "Reports", "Model").
# - path: file or directory path to display.
def print_artifact_path(label: str, path) -> None:
    _console.print(f"{label}  [dim]{path}[/dim]")


# Prints prediction results as a Rich table with borders.
# In:
# - title: track title.
# - artist: track artist.
# - results: list of (playlist, probability) tuples, sorted descending by probability.
def print_prediction_table(title: str, artist: str, results: list) -> None:
    table = Table(title=f"{title}\nBy: {artist}", show_header=True, header_style="bold")
    table.add_column("Playlist", min_width=30)
    table.add_column("Probability", justify="right", min_width=12)
    for playlist, prob in results:
        table.add_row(playlist, f"{prob:.3f}")
    _console.print(table)


# Prints a compact legend of metrics with descriptions and target goals.
# In:
# - include_guard: whether to include the recall guard row (k-fold only).
def print_metric_legend(include_guard: bool = False) -> None:
    rows = [
        ("\u00b5F1",   "micro F1 — F1 pooled across all label decisions",       "target \u2265 0.70"),
        ("MF1",   "macro F1 — avg F1 per playlist, equal weight per class", "target \u2265 0.65"),
        ("loss",  "BCE loss — binary cross-entropy over all labels",         "lower is better"),
    ]
    if include_guard:
        rows.append(("guard", "recall guard — all top-5 playlists meet recall floor", "\u2713 = all satisfied"))

    table = Table(show_header=False, box=None, padding=(0, 2), show_edge=False)
    table.add_column(style="dim bold", no_wrap=True)
    table.add_column(style="dim", no_wrap=True)
    table.add_column(style="dim italic", no_wrap=True)
    for abbr, desc, goal in rows:
        table.add_row(abbr, desc, goal)
    _console.print(table)


# Prints a rekordbox import completion summary.
# In:
# - track_count: number of tracks imported.
# - playlist_count: number of playlists imported.
def print_import_summary(track_count: int, playlist_count: int) -> None:
    _console.print(
        f"Imported [cyan]{track_count}[/cyan] tracks, [cyan]{playlist_count}[/cyan] playlists."
    )


# Tracks timing state across epochs and exposes helpers to print epoch rows
# without requiring callers to manually manage start times or epoch time lists.
class TrainingMonitor:
    # Initializes monitor state for a new training run.
    # In:
    # - total_epochs: total number of epochs that will be run.
    def __init__(self, total_epochs: int):
        self._total_epochs = total_epochs
        self._start_time = time.time()
        self._epoch_times: list = []

    # Context manager that wraps one training epoch.
    # Shows a transient progress bar during batch iteration, then records elapsed time.
    # Yields an advance() callable; pass it as on_batch to train_epoch.
    # In:
    # - epoch: current epoch number (1-based).
    # - n_batches: total number of batches in the epoch (used for the progress bar).
    # - label: optional progress bar label; defaults to "Epoch N/total".
    @contextmanager
    def epoch(self, epoch: int, n_batches: int, label: str = None):
        _label = label or f"Epoch {epoch}/{self._total_epochs}"
        t0 = time.time()
        with epoch_progress(n_batches, _label) as advance:
            yield advance
        self._epoch_times.append(time.time() - t0)

    # Prints the epoch summary row for a single-split training run.
    # Automatically computes elapsed, cumulative, and ETA from recorded epoch times.
    # In:
    # - epoch: current epoch number (1-based).
    # - train_loss, val_loss: training and validation BCE losses.
    # - micro_f1, macro_f1: validation F1 scores.
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float,
                  micro_f1: float, macro_f1: float) -> None:
        elapsed = self._epoch_times[-1]
        total_elapsed = time.time() - self._start_time
        remaining = self._total_epochs - epoch
        avg = sum(self._epoch_times) / len(self._epoch_times)
        print_epoch_row(
            epoch, self._total_epochs, train_loss, val_loss, micro_f1, macro_f1,
            elapsed, total_elapsed,
            eta_seconds=avg * remaining if remaining > 0 else None,
        )

    # Prints the epoch summary row for a k-fold training run.
    # Automatically computes elapsed and ETA from recorded epoch times.
    # In:
    # - fold: current fold index (1-based).
    # - epoch: current epoch number (1-based).
    # - train_loss, val_loss: training and validation BCE losses.
    # - micro_f1, macro_f1: validation F1 scores.
    # - guard_ok: whether the per-playlist recall guard was satisfied this epoch.
    def log_fold_epoch(self, fold: int, epoch: int, train_loss: float, val_loss: float,
                       micro_f1: float, macro_f1: float, guard_ok: bool) -> None:
        elapsed = self._epoch_times[-1]
        remaining = self._total_epochs - epoch
        avg = sum(self._epoch_times) / len(self._epoch_times)
        print_fold_epoch_row(
            fold, epoch, self._total_epochs, train_loss, val_loss, micro_f1, macro_f1,
            guard_ok, elapsed,
            eta_seconds=avg * remaining if remaining > 0 else None,
        )

    # Prints the training completion summary with total elapsed time.
    def complete(self) -> None:
        print_training_complete(time.time() - self._start_time)
