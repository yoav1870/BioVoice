import os.path
from typing import List, Tuple, Optional,Any, Iterable


import librosa.display
from matplotlib import pyplot as plt
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from PreprocessParams import HOP_LENGTH, SAMPLE_RATE

""" 
for extracting meta-data for organized plot savings
"""
import re
from pathlib import Path
from typing import Any
from HyperParams import HPARAM_ALIASES

""" 
for extracting meta-data for organized plot savings
"""
def hparams_to_str(
    hparams: dict[str, Any],
    *,
    keys: list[str] | None = None,
    alias_map: dict[str, str] | None = None,
) -> str:
    """
    Convert selected hyper-parameters to a filesystem-safe tag.
    If `alias_map` is given, the key is replaced by alias_map[key].

    Example:
        >>> hparams = {"learning_rate": 1e-3, "batch_size": 32, "epochs": 50}
        >>> hparams_to_str(hparams)
        'lr0_001_bs32_ep50'
    """
    alias_map = alias_map or {}
    keys = keys or list(hparams.keys())

    parts: list[str] = []

    for k in keys:
        if k not in hparams:
            continue                     # silent skip if key not present
        v = hparams[k]

        # replace long key by short alias if available
        alias = alias_map.get(k, k)

        # canonicalise the value → string safe for filenames
        if isinstance(v, bool):
            v = int(v)
        safe_v = re.sub(r"[^\w\-]", "_", str(v))

        parts.append(f"{alias}{safe_v}")

    return "_".join(parts)

def plot_waveform(waveform, sample_rate):
    # Plot each channel separately
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    plt.figure(figsize=(12, 6))
    for i in range(num_channels):
        plt.subplot(num_channels, 1, i + 1)
        plt.plot(time_axis, waveform[i].numpy(), label=f'Channel {i + 1}')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title(f'Waveform for Channel {i + 1}')

    plt.tight_layout()
    plt.show()

def plot_mel_spectrogram(mel_spec, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, block=True):
    """
    Plot a mel spectrogram on a given matplotlib Axes.
    """
    if isinstance(mel_spec, torch.Tensor):
        mel_spec = mel_spec.squeeze().numpy()  # Remove extra dimensions and convert to NumPy
    
    # if the first dimension is 3, assume it's channels and the first dim to last dim
    if mel_spec.ndim == 3 and mel_spec.shape[0] == 3:
        mel_spec = np.moveaxis(mel_spec, 0, -1)  # CHW -> HWC

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', cmap='inferno', fmax=sr//2)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show(block=block)

def plot_loss_per_epoch(
                        file_save_name: str,
                        dir_path: Path,
                        hparams: dict[str, Any] | None = None,
                        keys: Iterable[str] | None = None, # you don’t have to supply keys at all. It’s there only if you want to override the default order or filter the list.
                        **kwargs
                        ):    # NEW → controls order
    training_loss = kwargs.get("training_loss", None)
    validation_loss = kwargs.get("validation_loss", None)

    if training_loss is None and validation_loss is None:
        print("No data given for plotting")
    else:
        # Plot
        plt.figure(figsize=(10, 6))

        if training_loss is not None:
            plt.plot(training_loss, marker='o', linestyle='-', color='b', label='Training Loss')

        if validation_loss is not None:
            plt.plot(validation_loss, marker='o', linestyle='--', color='r', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss per Epoch')
        plt.legend()
        plt.grid(True)
        os.makedirs(dir_path, exist_ok=True)
        # ---------- NEW saving logic ---------- #
        tag = hparams_to_str(hparams or {}, keys=keys,alias_map=HPARAM_ALIASES)
        fname = f"{file_save_name}_{tag}.png"
        plt.savefig(os.path.join(dir_path, fname), bbox_inches="tight")
        # -------------------------------------- #

def plot_accuracy_per_epoch(
    file_save_name: str,
    dir_path: Path,
    hparams: dict[str, Any] | None = None,
    keys: Iterable[str] | None = None,
    **kwargs):
    training_accuracy = kwargs.get("training_accuracy", None)
    validation_accuracy = kwargs.get("validation_accuracy", None)
    test_accuracy = kwargs.get("test_accuracy", None)

    if training_accuracy is None and validation_accuracy is None and test_accuracy is None:
        print("No data given for plotting")
    else:
        # Plot
        plt.figure(figsize=(10, 6))

        if training_accuracy is not None:
            plt.plot(training_accuracy, marker='o', linestyle='-', color='b', label='Training Accuracy')

        if validation_accuracy is not None:
            plt.plot(validation_accuracy, marker='o', linestyle='--', color='r', label='Validation Accuracy')

        if test_accuracy is not None:
            test_accuracy *= 100
            plt.axhline(y=test_accuracy, color='g', linestyle=':', linewidth=2, label="Test Accuracy")
            # Add text label near the line
            plt.text(
                x=len(training_accuracy) + 1 if training_accuracy else 0,  # position to the right
                y=test_accuracy + 0.6,  # slight offset above the line
                s=f"{test_accuracy:.2f}%",
                color="g",
                fontsize=10,
                fontweight="bold"
            )

        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy per Epoch')
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(True)
        os.makedirs(dir_path, exist_ok=True)
        # ---------- NEW saving logic ---------- #
        tag = hparams_to_str(hparams or {}, keys=keys, alias_map=HPARAM_ALIASES)
        fname = f"{file_save_name}_{tag}.png"
        plt.savefig(os.path.join(dir_path, fname), bbox_inches="tight")
        # -------------------------------------- #

def plot_confusion_matrix(
    file_save_name: str,
    dir_path: Path,
    hparams: dict[str, Any] | None = None,
    keys: Iterable[str] | None = None,
    **kwargs):
    train_values_data: Optional[Tuple[list, list, list]]= kwargs.get("train_label_data")
    val_values_data: Optional[Tuple[list, list, list]] = kwargs.get("val_label_data")
    test_values_data: Optional[Tuple[list, list, list]] = kwargs.get("test_label_data")

    if not train_values_data and not val_values_data:
        print("No data provided for confusion matrices")

    conf_matrices = []
    titles = []
    data_classes = []

    if train_values_data:
        train_truth_labels, train_pred_labels, train_classes = train_values_data
        assert len(train_truth_labels) > 0 and len(train_pred_labels) > 0, "train_truth_labels or train_pred_labels is empty!"
        conf_matrices.append(confusion_matrix(np.array(train_truth_labels).flatten(), np.array(train_pred_labels).flatten()))
        titles.append("Train")
        data_classes.append(train_classes)

    if val_values_data:
        val_truth_labels, val_pred_labels, val_classes = val_values_data
        assert len(val_truth_labels) > 0 and len(val_truth_labels) > 0, "val_truth_labels or val_pred_labels is empty!"

        conf_matrices.append(confusion_matrix(np.array(val_truth_labels).flatten(), np.array(val_pred_labels).flatten()))
        titles.append("Validation")
        data_classes.append(val_classes)

    if test_values_data:
        test_truth_labels, test_pred_labels, test_classes = test_values_data
        assert len(test_truth_labels) > 0 and len(test_pred_labels) > 0, "val_truth_labels or val_pred_labels is empty!"

        conf_matrices.append(confusion_matrix(np.array(test_truth_labels).flatten(), np.array(test_pred_labels).flatten()))
        titles.append("Test")
        data_classes.append(test_classes)

    fig, axes = plt.subplots(1, len(conf_matrices), figsize=(7 * len(conf_matrices), 10))

    # If only one confusion matrix, turn variable to iterable.
    if len(conf_matrices) == 1:
        axes = [axes]

    split_cmaps = {
        "Train": "Blues",
        "Validation": "Oranges",
        "Test": "Greens"
    }

    for ax, conf_matrix, title, class_names in zip(axes, conf_matrices, titles, data_classes):
        sns.heatmap(conf_matrix,
                    annot=True,
                    fmt="d",
                    cmap=split_cmaps[title],
                    xticklabels=class_names,
                    yticklabels=class_names,
                    ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    plt.tight_layout()
    
    os.makedirs(dir_path, exist_ok=True)
    # ---------- NEW saving logic ---------- #
    tag = hparams_to_str(hparams or {}, keys=keys, alias_map=HPARAM_ALIASES)
    fname = f"{file_save_name}_{tag}.png"
    plt.savefig(os.path.join(dir_path, fname), bbox_inches="tight")
  
def save_mel_spectrogram(
    mel_spec, 
    file_save_path: Path,
    hparams: dict[str, Any] | None = None,
    keys: List[str] | None = None,
    sr=SAMPLE_RATE, 
    hop_length=HOP_LENGTH
):
    """
    Create and save a mel spectrogram visualization to file.
    
    Args:
        mel_spec: The mel spectrogram data
        file_save_path: Full path where to save the image file
        hparams: Optional hyperparameters to include in filename
        keys: Optional list of hyperparameter keys to include
        sr: Sample rate
        hop_length: Hop length for the spectrogram
    """
    if isinstance(mel_spec, torch.Tensor):
        mel_spec = mel_spec.squeeze().numpy()  # Remove extra dimensions and convert to NumPy

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', cmap='inferno', fmax=sr//2)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    
    # Extract directory and make sure it exists
    save_dir = os.path.dirname(file_save_path)
    if save_dir != '':
        os.makedirs(save_dir, exist_ok=True)
    
    # Add hyperparameter tag to filename if provided
    if hparams:
        base_path = os.path.splitext(file_save_path)[0]
        ext = os.path.splitext(file_save_path)[1]
        tag = hparams_to_str(hparams, keys=keys, alias_map=HPARAM_ALIASES)
        final_path = f"{base_path}_{tag}{ext}"
    else:
        final_path = file_save_path
        
    plt.savefig(final_path, bbox_inches="tight")
    plt.close()  # Close the figure to free memory