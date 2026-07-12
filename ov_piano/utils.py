#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
This generic module contains different bits code that can be re-used in
various modules and applications.
"""


import os
import json
import random
from collections import OrderedDict
from typing import Optional
#
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import numpy as np
import h5py
#
from .custom_logging import make_timestamp


# ##############################################################################
# # HDF5 DATABASES
# ##############################################################################
class IncrementalHDF5:
    """
    Incrementally concatenate matrices of same height. This can be useful to
    e.g. join all spectrograms of a database into a single file, for faster
    processing.
    """
    DATA_NAME = "data"
    METADATA_NAME = "metadata"
    IDXS_NAME = "data_idxs"

    def __init__(self, out_path, height, dtype=np.float32, compression="lzf",
                 data_chunk_length=500, metadata_chunk_length=500,
                 err_if_exists=True):
        """
        :param height: This class incrementally stores a matrix of shape
          ``(height, w++)``, where ``height`` is always fixed.
        :param compression: ``lzf`` is fast, ``gzip`` slower but provides
          better compression
        :param data_chunk_length: Every I/O operation goes by chunks. A too
          small chunk size will cause many syscalls (slow), and with a too
          large chunk size we will be loading too much information in a single
          syscall (also slow, and bloats the RAM). Ideally, the chunk length is
          a bit larger than what is usually needed (e.g. if we expect to read
          between 10 and 50 rows at a time, we can choose chunk=60).
        """
        self.out_path = out_path
        self.height = height
        self.dtype = dtype
        self.compression = compression
        #
        if err_if_exists:
            if os.path.isfile(out_path):
                raise FileExistsError(f"File already exists! {out_path}")
        #
        self.h5f = h5py.File(out_path, "w")
        self.data_ds = self.h5f.create_dataset(
            self.DATA_NAME, shape=(height, 0), maxshape=(height, None),
            dtype=dtype, compression=compression,
            chunks=(height, data_chunk_length))
        self.metadata_ds = self.h5f.create_dataset(
            self.METADATA_NAME, shape=(0,), maxshape=(None,),
            compression=compression, dtype=h5py.string_dtype(),
            chunks=(metadata_chunk_length,))
        self.data_idxs_ds = self.h5f.create_dataset(
            self.IDXS_NAME, shape=(2, 0), maxshape=(2, None), dtype=np.int64,
            compression=compression, chunks=(2, metadata_chunk_length))
        self._current_data_width = 0
        self._num_entries = 0

    def __enter__(self):
        """
        """
        return self

    def __exit__(self, type, value, traceback):
        """
        """
        self.close()

    def close(self):
        """
        """
        self.h5f.close()

    def append(self, matrix, metadata_str):
        """
        :param matrix: dtype array of shape ``(fix_height, width)``
        """
        n = self._num_entries
        h, w = matrix.shape
        assert h == self.height, \
            f"Shape was {(h, w)} but should be ({self.height}, ...). "
        # update arr size and add data
        new_data_w = self._current_data_width + w
        self.data_ds.resize((self.height, new_data_w))
        self.data_ds[:, self._current_data_width:new_data_w] = matrix
        # # update meta-arr size and add metadata
        self.metadata_ds.resize((n + 1,))
        self.metadata_ds[n] = metadata_str
        # update data-idx size and add entry
        self.data_idxs_ds.resize((2, n + 1))
        self.data_idxs_ds[:, n] = (self._current_data_width, new_data_w)
        #
        self.h5f.flush()
        self._current_data_width = new_data_w
        self._num_entries += 1

    @classmethod
    def get_element(cls, h5file, elt_idx):
        """
        :param int elt_idx: Index of the appended element, e.g. first element
          has index 0, second has index 1...
        :returns: the ``(data, metadata_str)`` corresponding to that index,
          as they were appended.
        """
        data_beg, data_end = h5file[cls.IDXS_NAME][:, elt_idx]
        data = h5file[cls.DATA_NAME][:, data_beg:data_end]
        metadata = h5file[cls.METADATA_NAME][elt_idx].decode("utf-8")
        return data, metadata

    @classmethod
    def get_num_elements(cls, h5file):
        """
        :returns: The number of elements that have been added to the file via
          append.
        """
        num_elements = len(h5file[cls.METADATA_NAME])
        return num_elements


# ##############################################################################
# # AUDIO PREPROCESSING
# ##############################################################################
def torch_resample_audio(wave: torch.Tensor, sr_in: int, target_sr: int, mono: bool = True, device: str = "cpu") -> torch.Tensor:
    """
    Resamples a wave tensor.
    :param wave: The input wave tensor
    :param sr_in: The input sample rate
    :param target_sr: The target sample rate
    :param mono: If true, returned wavfile will be averaged down to mono.
    :param device: Returned wavfile will be on the specified device.
    """
    wave = wave.to(device)

    # Convert to mono if requested and needed
    if mono and wave.dim() > 1 and wave.shape[0] > 1:
        wave = wave.mean(dim=0, keepdim=True)

    # Resample if necessary
    if sr_in != target_sr:
        resampler = torchaudio.transforms.Resample(sr_in, target_sr).to(device)
        wave = resampler(wave)
    
    return wave

def torch_load_resample_audio(path, target_sr=16000, mono=True,
                              device="cpu"):
    """
    Analogously to ``librosa.load``, this function loads and resamples a wav
    file. The resampling operation from torchaudio is much faster.
    :param path: Absolute path to the wav file to be loaded
    :param target_sr: Returned wavfile will have this sample rate
    :param mono: If true, returned wavfile will be averaged down to mono.
    :param device: Returned wavfile will be on the specified device.
    """
    wave, sr_in = torchaudio.load(path)
    return torch_resample_audio(wave, sr_in, target_sr, mono, device)


class TorchWavToLogmel(torch.nn.Module):
    """
    Analogous to ``librosa``, torchaudio implementation to convert wav arrays
    to log-mel spectrograms. Much faster, results differ only slightly.
    Since this is a torch Module, can be sent ``.to("cuda")`` in order
    to admit CUDA tensors.
    """
    def __init__(self, samplerate, winsize, hopsize, n_mels,
                 mel_fmin=50, mel_fmax=8_000, window_fn=torch.hann_window):
        """
        :param samplerate: Expected audio input samplerate.
        :param winsize: Window size for the STFT (and mel).
        :param hopsize: Hop size for the STFT (and mel).
        :param stft_window: Windowing function for the STFT.
        :param n_mels: Number of mel bins.
        :param mel_fmin: Lowest mel bin, in Hz.
        :param mel_fmax: Highest mel bin, in Hz.
        """
        super().__init__()
        self.melspec = MelSpectrogram(
            samplerate, winsize, hop_length=hopsize,
            f_min=mel_fmin, f_max=mel_fmax, n_mels=n_mels,
            power=2, window_fn=window_fn)
        self.to_db = AmplitudeToDB(stype="power", top_db=80.0)
        # run melspec once, otherwise produces NaNs!
        self._warm_up_melspec(winsize)

    def _module_device(self) -> torch.device:
        """Return the current device of this module's tensors."""
        return self.melspec.spectrogram.window.device

    def _warm_up_melspec(self, winsize: int) -> None:
        """Initialize torchaudio internals without changing module device semantics."""
        self.melspec(torch.rand(winsize * 10, device=self._module_device()))

    def forward(self, wav_arr):
        """
        :param wav_arr: Float tensor array of either 1D or ``(chans, time)``
        :returns: log-mel spectrogram of shape ``(n_mels, t)``
        """
        mel = self.melspec(wav_arr)
        log_mel = self.to_db(mel)

        # If input was batched but batch size was 1, squeeze the batch dim
        if log_mel.dim() == 3 and log_mel.shape[0] == 1:
            log_mel = log_mel.squeeze(0)

        return log_mel


# ##############################################################################
# # DL MODEL SERIALIZATION
# ##############################################################################
def save_model(model, path):
    """
    """
    torch.save(model.state_dict(), path)


def _shape_tuple(value):
    """Return a printable tensor-like shape, or ``None`` for non-shaped values."""
    return tuple(value.shape) if hasattr(value, "shape") else None


def format_load_model_warnings(
    report,
    ignored_missing_keys=None,
    ignored_missing_key_prefixes=None,
):
    """Format non-strict checkpoint loading diagnostics as warning strings.

    ``ignored_missing_keys`` and ``ignored_missing_key_prefixes`` are intended
    for explicitly expected migration paths, such as loading an older note-only
    checkpoint into a newer model that initializes a newly-added head from
    scratch. They only affect the *missing key* diagnostic; unexpected and
    shape-mismatched checkpoint entries are always reported.
    """
    warnings = []
    if not report or report.get("strict", True):
        return warnings

    path = report.get("path", "<unknown checkpoint>")
    ignored_missing_keys = set(ignored_missing_keys or ())
    ignored_missing_key_prefixes = tuple(ignored_missing_key_prefixes or ())
    missing_keys = [
        key for key in report.get("missing_keys", [])
        if key not in ignored_missing_keys
        and not any(key.startswith(prefix) for prefix in ignored_missing_key_prefixes)
    ]
    unexpected_keys = report.get("unexpected_keys", [])
    shape_mismatched_keys = report.get("shape_mismatched_keys", [])

    if missing_keys:
        warnings.append(
            f"Checkpoint '{path}' is missing {len(missing_keys)} model key(s): "
            + ", ".join(missing_keys)
        )
    if unexpected_keys:
        warnings.append(
            f"Checkpoint '{path}' has {len(unexpected_keys)} unexpected key(s): "
            + ", ".join(unexpected_keys)
        )
    if shape_mismatched_keys:
        details = []
        for item in shape_mismatched_keys:
            details.append(
                f"{item['key']} checkpoint_shape={item['checkpoint_shape']} "
                f"model_shape={item['model_shape']}"
            )
        warnings.append(
            f"Checkpoint '{path}' has {len(shape_mismatched_keys)} shape-mismatched key(s) skipped: "
            + "; ".join(details)
        )
    return warnings


def load_model(model, path, eval_phase=True, strict=True, to_cpu=False):
    """
    Load a PyTorch state dict into ``model``.

    When ``strict=False``, shape-incompatible checkpoint entries are skipped so
    partially compatible checkpoints can still be loaded. A diagnostics report is
    returned with missing model keys, unexpected checkpoint keys, and skipped
    shape-mismatched checkpoint keys. Existing callers may ignore the return
    value, but callers that perform non-strict loading should log it.
    """
    try:
        state_dict = torch.load(path, map_location="cpu" if to_cpu else None)
    except RuntimeError as exc:
        raise RuntimeError(f"Failed to load PyTorch checkpoint '{path}': {exc}") from exc

    report = {
        "path": path,
        "strict": strict,
        "missing_keys": [],
        "unexpected_keys": [],
        "shape_mismatched_keys": [],
    }

    if not strict:
        migration_fn = getattr(model, "migrate_checkpoint_state_dict", None)
        if callable(migration_fn):
            state_dict = migration_fn(state_dict)

        model_state = model.state_dict()
        filtered_state_dict = OrderedDict()
        if hasattr(state_dict, "_metadata"):
            filtered_state_dict._metadata = state_dict._metadata
        unexpected_keys = []
        shape_mismatched_keys = []
        for key, value in state_dict.items():
            if key not in model_state:
                unexpected_keys.append(key)
                continue
            checkpoint_shape = _shape_tuple(value)
            model_shape = _shape_tuple(model_state[key])
            if checkpoint_shape != model_shape:
                shape_mismatched_keys.append(
                    {
                        "key": key,
                        "checkpoint_shape": checkpoint_shape,
                        "model_shape": model_shape,
                    }
                )
                continue
            filtered_state_dict[key] = value
        state_dict = filtered_state_dict
        report["unexpected_keys"] = unexpected_keys
        report["shape_mismatched_keys"] = shape_mismatched_keys

    incompatible = model.load_state_dict(state_dict, strict=strict)
    if not strict:
        shape_mismatched_key_names = {
            item["key"] for item in report["shape_mismatched_keys"]
        }
        report["missing_keys"] = [
            key for key in incompatible.missing_keys
            if key not in shape_mismatched_key_names
        ]
        # Preserve unexpected keys found before filtering. PyTorch will not see
        # them after filtering, but they are important diagnostics for callers.
        report["unexpected_keys"].extend(
            key for key in incompatible.unexpected_keys
            if key not in report["unexpected_keys"]
        )
    if eval_phase:
        model.eval()
    else:
        model.train()
    return report


class ModelSaver:
    """
    Convenience functor to save model at specific times, can be used as a
    parameterless hook e.g. at the end of each SGDR cycle.
    """
    def __init__(self, model, out_folder, log_fn=None):
        """
        """
        self.model = model
        self.model_name = model.__class__.__name__
        self.out_folder = out_folder
        self.log_fn = log_fn

    def __call__(self, suffix=None):
        """
        :param suffix: If given, string added after the output basename.
        """
        basename = f"{self.model_name}_{make_timestamp(with_tz_output=False)}"
        if suffix is not None:
            basename += suffix
        out_path = os.path.join(self.out_folder, basename + ".torch")
        save_model(self.model, out_path)
        if self.log_fn is not None:
            msg = f"Saved model to {out_path}"
            self.log_fn(msg)
        return out_path


# ##############################################################################
# # TRAINING UTILS
# ##############################################################################
def set_seed(seed=0):
    """
    Set randomness seed for Python, NumPy and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_resume_state(epoch, batch_idx, global_step, checkpoint_dir):
    """
    Save training resume state (epoch, batch index, global step) to JSON file.
    
    This enables resumption from exact position within an epoch.
    
    :param epoch: Current epoch number (1-indexed)
    :param batch_idx: Current batch index within epoch (0-indexed)
    :param global_step: Global training step counter
    :param checkpoint_dir: Directory where resume state file will be saved
    """
    resume_state = {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "global_step": global_step
    }
    state_path = os.path.join(checkpoint_dir, ".resume_state.json")
    with open(state_path, "w") as f:
        json.dump(resume_state, f)
    return state_path


def load_resume_state(checkpoint_dir):
    """
    Load training resume state from JSON file.
    
    Returns resume state if exists, otherwise returns None.
    
    :param checkpoint_dir: Directory to search for resume state file
    :return: dict with keys 'epoch', 'batch_idx', 'global_step' or None
    """
    state_path = os.path.join(checkpoint_dir, ".resume_state.json")
    if os.path.isfile(state_path):
        try:
            with open(state_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None


def breakpoint_json(path="breakpoint.json", step=None):
    """
    This function can be used to trigger a breakpoint during training, by
    altering the contents of a given JSON file, by:
    * Setting ``inconditional`` to true
    * Setting ``step_gt`` to a number and then passing a
      ``step`` that is larger than that number
    * Setting ``step_every`` to a number and then passing
      a ``step`` that is divided by that number.

    If any of the above conditions (checked in that order) is
    met, the function returns ``True``. Otherwise False.
    """
    try:
        with open(path, "r") as f:
            j = json.load(f)
        #
        incond = j["inconditional"]
        step_gt = j["step_gt"]
        step_every = j["step_every"]
        #
        if incond:
            return True
        elif ((step is not None) and (step_gt is not None) and
              (step >= step_gt)):
            return True
        elif ((step is not None) and (step_every is not None) and
              ((step % step_every) == 0)):
            return True
        else:
            return False
    except Exception as e:
        print("Exception in breakpoint_json! returning False.", e)
        return False


class MaskedBCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    """
    This module extends ``torch.nn.BCEWithlogitsloss`` with the possibility
    to multiply each scalar loss by a mask number between 0 and 1, before
    aggregating via average.
    """

    def __init__(self, *args, **kwargs):
        """
        """
        super().__init__(*args, **kwargs, reduction="none")

    def forward(self, pred, target, mask=None):
        """
        """
        eltwise_loss = super().forward(pred, target)
        if mask is not None:
            assert mask.min() >= 0, "Mask must be in [0, 1]!"
            assert mask.max() <= 1, "Mask must be in [0, 1]!"
            eltwise_loss = eltwise_loss * mask
        result = eltwise_loss.mean()
        #
        return result


def init_weights(module, init_fn=torch.nn.init.kaiming_normal,
                 bias_val=0.0, verbose=False):
    """
    Custom, layer-aware initializer for PyTorch modules.

    :param init_fn: initialization function, such that ``init_fn(weight)``
      modifies in-place the weight values. If ``None``, found weights won't be
      altered
    :param float bias_val: Any module with biases will initialize them to this
      constant value

    Usage example, inside of any ``torch.nn.Module.__init__`` method:

    if init_fn is not None:
            self.apply(lambda module: init_weights(module, init_fn, 0.0))

    Apply is applied recursively to any submodule inside, so this works.
    """
    if isinstance(module, (torch.nn.Linear,
                           torch.nn.Conv1d,
                           torch.nn.Conv2d)):
        if init_fn is not None:
            init_fn(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(bias_val)
    elif isinstance(module, (torch.nn.GRU, torch.nn.LSTM)):
        raise NotImplementedError("No RNNs supported at the moment :)")
    else:
        if verbose:
            print("init_weights: ignored module:", module.__class__.__name__)
