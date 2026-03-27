import librosa
import librosa.display
from intervaltree import IntervalTree
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from IPython.display import Audio, display
import soundfile as sf
import urllib.request
import tarfile
import shutil
from pathlib import Path
import torch


# --- Download -----------------------------------------------------------
def _reporthook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct = min(100.0, 100 * downloaded / total_size) if total_size > 0 else 0
    mb  = downloaded / 1e6
    total_mb = total_size / 1e6
    print(f"\r  {pct:5.1f}%  {mb:.0f} / {total_mb:.0f} MB", end="", flush=True)

def download(url: Path, archive: Path, output_dir: Path, dataset_name: str = "dataset"):
    if not archive.exists():
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, archive, reporthook=_reporthook)
        print("\nDone.")
    else:
        print("Archive already present, skipping download.")

    # --- Extract --------------------------------------------------------
    if not (output_dir / dataset_name).exists():
        print("Extracting archive (this may take a few minutes) ...")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(output_dir)
        print("Extraction complete.")
    else:
        print(f"{dataset_name} already extracted.")


# ---- Helpers -----------------------------------------------------------
def load_audio(path: str, target_sr: int = 16000, duration=1000):
    """Load and resample to target_sr. Returns (waveform np array, sr)."""
    wav, sr = librosa.load(path)
    wav = torch.from_numpy(wav)
    print(wav)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0).numpy(), target_sr


def shade_breaths(ax, tree: IntervalTree, ymin=0, ymax=1, alpha=0.25):
    """Draw translucent red spans over detected breath regions."""
    for iv in sorted(tree):
        ax.axvspan(iv.begin, iv.end, ymin=ymin, ymax=ymax,
                   color="tomato", alpha=alpha, linewidth=0)


def plot_segment(breath_preds, wave: np.array = [], sr: int = 16000,
                 title: str = ""):
    """
    Run breath detection and produce a 2-panel figure:
      top    — waveform
      bottom — log-mel spectrogram
    Breath regions are highlighted in both panels.
    """
    # --- Breath detection -----------------------------------------------

    breath_intervals = sorted(breath_preds)

    # --- Audio ----------------------------------------------------------
    #wave, sr = load_audio(audio_path) if audio_path != "" else audio
    duration  = len(wave) / sr
    times     = np.linspace(0, duration, len(wave))

    # --- Mel spectrogram ------------------------------------------------
    n_fft    = 600   # 25 ms @ 24 kHz
    hop      = 240   # 10 ms @ 24 kHz
    mel      = librosa.feature.melspectrogram(y=wave, sr=sr,
                                               n_fft=n_fft, hop_length=hop,
                                               n_mels=80, fmax=8000)
    log_mel  = librosa.power_to_db(mel, ref=np.max)

    # --- Figure ---------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(13, 5),
                              gridspec_kw={"height_ratios": [1, 2]})
    fig.suptitle(title, fontsize=11, y=1.01)

    # Waveform
    ax0 = axes[0]
    ax0.plot(times, wave, color="#4a90d9", linewidth=0.4, rasterized=True)
    ax0.set_xlim(0, duration)
    ax0.set_ylabel("Amplitude")
    ax0.set_xlabel("")
    ax0.tick_params(labelbottom=False)
    shade_breaths(ax0, breath_preds)

    # Mel spectrogram
    ax1 = axes[1]
    img = librosa.display.specshow(log_mel, sr=sr, hop_length=hop,
                                    x_axis="time", y_axis="mel",
                                    fmax=8000, ax=ax1, cmap="magma")
    fig.colorbar(img, ax=ax1, format="%+2.0f dB", pad=0.01)
    ax1.set_xlabel("Time (s)")
    shade_breaths(ax1, breath_preds)

    # Legend proxy
    from matplotlib.patches import Patch
    legend_handle = Patch(color="tomato", alpha=0.5, label="Breath")
    ax0.legend(handles=[legend_handle], loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()

    # Breath summary
    total_breath = sum(iv.end - iv.begin for iv in breath_intervals)
    print(f"  Duration: {duration:.2f}s  |  "
          f"Breaths detected: {len(breath_intervals)}  |  "
          f"Total breath time: {total_breath:.2f}s ({100*total_breath/duration:.1f}%)")
    if breath_intervals:
        print("  Intervals (s):", [(round(iv.begin,2), round(iv.end,2)) for iv in breath_intervals])

    # Audio widget
    display(Audio(wave, rate=sr))
    print()
