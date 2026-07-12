#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
This module hosts re-usable evaluation code for automated piano transcription,
including:
* Convenience classes to load MIDI files from datasets, and convert them to
  Pandas dataframes storing events with onset, offset... this helps in two
  ways: first, multiple things can be processed in parallel; second, the full
  dataset is processed at once, avoiding redundancies and speeding up training.
* Event-base evaluation that makes use of official ``mir_eval`` implementations
  to ensure rigor, reproducibility and compatibility with previous literature.
"""


import hashlib
import json
import os
import numpy as np
import pandas as pd
import torch
from mir_eval.transcription import precision_recall_f1_overlap as prf1o
from mir_eval.transcription_velocity import precision_recall_f1_overlap \
    as prf1o_v
from .data.key_model import KeyboardStateMachine
from .data.midi import MidiToPianoRoll, MaestroMidiParser, SingletrackMidiParser
from .inference import PedalDecoder


# ##############################################################################
# # EVALUATION CHECKPOINTS
# ##############################################################################
EVAL_CHECKPOINT_VERSION = 1


def _json_default(value):
    """Convert common scientific-Python objects to stable JSON values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (tuple, set)):
        return list(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def evaluation_fingerprint(config):
    """Return a short stable hash for an evaluation checkpoint configuration.

    The fingerprint is used to avoid accidentally resuming cached inference or
    metrics from a different model, dataset, split, decoder setup, or threshold
    search.  ``config`` should contain only values that affect the cached stage.
    """
    payload = json.dumps(config, sort_keys=True, default=_json_default)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def metadata_to_file_id(metadata):
    """Return a stable per-piece identifier from MAESTRO/MAPS metadata."""
    if isinstance(metadata, (list, tuple)) and len(metadata) > 0:
        return str(metadata[0])
    return str(metadata)


def evaluation_checkpoint_path(checkpoint_dir, script_name, stage, fingerprint):
    """Build a readable checkpoint path for a script/stage/fingerprint tuple."""
    safe_script = "".join(
        ch if ch.isalnum() or ch in ("-", "_", ".") else "_"
        for ch in os.path.basename(str(script_name))
    )
    safe_stage = "".join(
        ch if ch.isalnum() or ch in ("-", "_", ".") else "_"
        for ch in str(stage)
    )
    return os.path.join(
        checkpoint_dir,
        f"{safe_script}.{safe_stage}.{fingerprint}.pt",
    )


def _torch_load_checkpoint(path):
    """Load a checkpoint payload across PyTorch versions.

    PyTorch 2.6 changed ``torch.load`` defaults toward ``weights_only=True``.
    Evaluation checkpoints intentionally store Pandas dataframes and other Python
    objects, so they must be loaded as trusted local artifacts.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class EvaluationCheckpointStore:
    """Small append/update store for resumable evaluation stages.

    Each stage stores a dictionary of per-file entries under one atomic ``.pt``
    file.  Entries can contain Pandas dataframes and CPU tensors, which makes the
    helper suitable for persisting expensive full-file inference outputs or final
    per-file metrics after each successfully processed piece.
    """

    def __init__(self, path, fingerprint, stage, enabled=True, reset=False,
                 logger=None):
        self.path = os.fspath(path)
        self.fingerprint = str(fingerprint)
        self.stage = str(stage)
        self.enabled = bool(enabled)
        self.logger = logger
        self.payload = {
            "version": EVAL_CHECKPOINT_VERSION,
            "stage": self.stage,
            "fingerprint": self.fingerprint,
            "items": {},
        }

        if not self.enabled:
            return

        out_dir = os.path.dirname(self.path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        if reset and os.path.isfile(self.path):
            os.remove(self.path)
            self._log(f"Reset evaluation checkpoint: {self.path}")

        self._load_existing()

    def _log(self, message):
        if self.logger is not None:
            self.logger(message)

    def _load_existing(self):
        if not os.path.isfile(self.path):
            return
        try:
            payload = _torch_load_checkpoint(self.path)
        except Exception as exc:
            self._log(
                f"Ignoring unreadable evaluation checkpoint {self.path}: {exc}"
            )
            return

        if not isinstance(payload, dict):
            self._log(f"Ignoring malformed evaluation checkpoint {self.path}")
            return
        if payload.get("version") != EVAL_CHECKPOINT_VERSION:
            self._log(
                f"Ignoring evaluation checkpoint with incompatible version: "
                f"{self.path}"
            )
            return
        if payload.get("stage") != self.stage:
            self._log(f"Ignoring evaluation checkpoint for another stage: {self.path}")
            return
        if payload.get("fingerprint") != self.fingerprint:
            self._log(
                f"Ignoring stale evaluation checkpoint with different fingerprint: "
                f"{self.path}"
            )
            return

        items = payload.get("items", {})
        if not isinstance(items, dict):
            self._log(f"Ignoring evaluation checkpoint with invalid items: {self.path}")
            return
        self.payload = payload
        self._log(
            f"Loaded evaluation checkpoint {self.path} "
            f"({len(self.payload['items'])} item(s))"
        )

    def __len__(self):
        return len(self.payload["items"])

    def get(self, file_id, default=None):
        return self.payload["items"].get(str(file_id), default)

    def contains(self, file_id):
        return str(file_id) in self.payload["items"]

    def items(self):
        return self.payload["items"].items()

    def upsert(self, file_id, entry):
        """Persist ``entry`` for ``file_id`` and atomically flush to disk."""
        if not self.enabled:
            return None
        self.payload["items"][str(file_id)] = entry
        tmp_path = self.path + ".tmp"
        torch.save(self.payload, tmp_path)
        os.replace(tmp_path, self.path)
        return self.path


# ##############################################################################
# # GROUND TRUTH CONVENIENCE LOADERS
# ##############################################################################
class GtLoaderMaps:
    """
    Auxiliary class to the main eval class (in order to avoid multiple forward
    passes during evaluation and to leverage optimal F1 threshold from roll
    evaluation).
    During construction, it parses and stores the MIDI files (also to avoid
    doing this multiple times). Then, it offers a series of convenience methods
    that can be used by the main eval class.
    """
    PARSER = SingletrackMidiParser
    MIDI_EXT = ".mid"
    MIN_NOTE_DUR = 0.001  # in seconds

    @classmethod
    def get_metadata_path(cls, data_md, meta_dataset):
        """
        :param dataset_md: along with the logmels and rolls, datasets provide
          metadata. This method reconstructs the complete path from this
          given metadata, such that it can be found in the meta_dataset.
        """
        basename, instr, cat = data_md
        path = os.path.join(meta_dataset.rootpath, instr, cat, basename)
        return path + cls.MIDI_EXT

    @classmethod
    def get_midi_eventdata(cls, abspath):
        """Get MIDI event data from a file."""
        # load and check midi
        mid = cls.PARSER.load_midi(abspath)
        msgs, meta_msgs = cls.PARSER.parse_midi(mid)
        MidiToPianoRoll._check_midi(msgs, meta_msgs)
        # convert midi to events with onset and offset
        (key_events, sus_states, ten_states, soft_states,
         largest_ts) = cls.PARSER.ksm_parse_midi_messages(
             msgs, KeyboardStateMachine(
                 MidiToPianoRoll.SUS_PEDAL_THRESH,
                 MidiToPianoRoll.TEN_PEDAL_THRESH,
                 ignore_redundant_keypress=True,
                 ignore_redundant_keylift=True))
        #
        return (key_events, sus_states, ten_states, soft_states, largest_ts)

    def __init__(self, dataset, meta_dataset):
        """Initialize GtLoaderMaps."""
        self.dataset, self.meta_dataset = dataset, meta_dataset
        self.midi_abspaths = [self.get_metadata_path(md, meta_dataset)
                              for _, _, md in dataset]
        # Disable ProcessPoolExecutor to avoid memory issues on Windows
        # Use sequential processing instead
        midi_eventdata = [self.get_midi_eventdata(ap) for ap in self.midi_abspaths]
        self.midi_eventdata = dict(zip(self.midi_abspaths, midi_eventdata))
        # all onset-offset intervals must be >0, so add epsilon if needed:
        for (key_evts, _, _, _, _) in self.midi_eventdata.values():
            diffs = key_evts["offset"] - key_evts["onset"]
            key_evts.loc[diffs == 0, "offset"] += self.MIN_NOTE_DUR

    def __call__(self, data_md):
        """
        :param data_md: The metadata output of the dataset. It is also the
          input to ``get_metadata_path``.
        """
        md_path = self.get_metadata_path(data_md, self.meta_dataset)
        result = self.midi_eventdata[md_path]
        return result


class GtLoaderMaestro(GtLoaderMaps):
    """
    Extension of ``GtLoaderMaps`` for MAESTRO.
    """
    PARSER = MaestroMidiParser
    MIDI_EXT = ".midi"

    @classmethod
    def get_metadata_path(cls, data_md, meta_dataset):
        """
        :param dataset_md: along with the logmels and rolls, datasets provide
          metadata. This method reconstructs the complete path from this
          given metadata, such that it can be found in the meta_dataset.
        """
        basename, year, _, _, _, _ = data_md
        path = os.path.join(meta_dataset.rootpath, str(year), basename)
        return path + cls.MIDI_EXT

    def get_sus_pedal_events(self, data_md, secs_per_frame):
        """
        Get sustain pedal events for the given metadata.
        
        :param data_md: The metadata output of the dataset
        :param secs_per_frame: Time per frame in seconds for pedal event conversion
        :returns: DataFrame with sustain pedal onset/offset events
        """
        md_path = self.get_metadata_path(data_md, self.meta_dataset)
        _, sus_states, _, _, _ = self.midi_eventdata[md_path]

        # Convert sustain pedal states to events
        sus_pedal_events = sus_states_to_events(sus_states)
        if not sus_pedal_events.empty:
            sus_pedal_events = sus_pedal_events.copy()
            sus_pedal_events["pedal_idx"] = 0
        return sus_pedal_events


# ##############################################################################
# # SUSTAIN PEDAL EVENT CONVERSION
# ##############################################################################
def sus_states_to_events(sus_states_df):
    """
    Convert sustain pedal state DataFrame to onset/offset events.

    :param sus_states_df: DataFrame with columns ["ts", "val"] from MIDI parsing
    :returns: DataFrame with columns ["onset", "event_type"] where event_type is "onset" or "offset"
    """
    if sus_states_df.empty:
        return pd.DataFrame(columns=["onset", "event_type"])

    events = []
    prev_state = 0
    thresholds_to_try = [7, 1, 0]  # Try progressively lower thresholds

    for threshold in thresholds_to_try:
        events = []
        prev_state = 0
        for _, row in sus_states_df.iterrows():
            current_state = 1 if row["val"] > threshold else 0

            # Detect state transitions
            if prev_state == 0 and current_state == 1:
                events.append({"onset": row["ts"], "event_type": "onset"})
            elif prev_state == 1 and current_state == 0:
                events.append({"onset": row["ts"], "event_type": "offset"})

            prev_state = current_state
        if len(events) > 0:
            break

    return pd.DataFrame(events)


# ##############################################################################
# # EVENT-BASED EVALUATION
# ##############################################################################
def _prepare_data(gt_onsets, pred_onsets, pred_keys, pred_key_shift,
                  pred_onset_mul, pred_shift):
    if pred_key_shift != 0:
        pred_keys = pred_keys + pred_key_shift
    if pred_onset_mul != 1.0:
        pred_onsets = pred_onsets * pred_onset_mul
    if pred_shift != 0:
        pred_onsets = pred_onsets + pred_shift
    gt_offsets = gt_onsets + 1
    pred_offsets = pred_onsets + 1
    return pred_keys, pred_onsets, gt_offsets, pred_offsets

def _calculate_scores(gt_onsets, gt_offsets, gt_keys, gt_vels, pred_onsets, pred_offsets, pred_keys, pred_vels, tol_secs, pitch_tolerance, velocity_tolerance):
    if (gt_vels is not None) and (pred_vels is not None):
        prec, rec, f1, _ = prf1o_v(
            np.stack((gt_onsets, gt_offsets)).T, gt_keys, gt_vels,
            np.stack((pred_onsets, pred_offsets)).T, pred_keys, pred_vels,
            onset_tolerance=tol_secs, pitch_tolerance=pitch_tolerance,
            velocity_tolerance=velocity_tolerance,
            offset_ratio=None,
            offset_min_tolerance=tol_secs)
    else:
        prec, rec, f1, _ = prf1o(
            np.stack((gt_onsets, gt_offsets)).T, gt_keys,
            np.stack((pred_onsets, pred_offsets)).T, pred_keys,
            onset_tolerance=tol_secs, pitch_tolerance=pitch_tolerance,
            offset_ratio=None,
            offset_min_tolerance=tol_secs)
    return prec, rec, f1

def eval_note_events(gt_onsets, gt_keys, pred_onsets, pred_keys, gt_vels=None, pred_vels=None, tol_secs=0.05, pitch_tolerance=0.1, velocity_tolerance=0.1, pred_key_shift=0, pred_onset_mul=1.0, pred_shift=0):
    if len(gt_onsets) == 0:
        return (1, 1, 1) if len(pred_onsets) == 0 else (0, 1, 0)
    if len(pred_onsets) == 0:
        return 0, 0, 0

    pred_keys, pred_onsets, gt_offsets, pred_offsets = _prepare_data(
        gt_onsets, pred_onsets, pred_keys, pred_key_shift,
        pred_onset_mul, pred_shift)
    prec, rec, f1 = _calculate_scores(gt_onsets, gt_offsets, gt_keys, gt_vels, pred_onsets, pred_offsets, pred_keys, pred_vels, tol_secs, pitch_tolerance, velocity_tolerance)
    return prec, rec, f1


def threshold_eval_single_file(
        gt_df, pred_df, secs_per_frame, pred_key_offset,
        thresh=0.5, shift_preds=0, tol_secs=0.05, tol_vel=0.1):
    """
    Given a set of ground truths, predictions and tolerances, the imported
    function ``eval_note_events`` returns their precision, recall and F1 score
    for the given time/velocity tolerances.

    This wrapper function receives ground truths and predictions as Pandas
    dataframes, and runs ``eval_note_events`` twice; once for the onsets alone
    and once for onsets+velocities. Also, it thresholds the predictions first
    by their onset probability, and shifts+scales the onset time by a constant.

    See the evaluation script for an usage example.

    :returns: A tuple ``((p, r, f1), (p_v, r_v, f1_v))``, where the first
      triple contains the onsets-only evaluation, and the second
      onsets+velocities.
    """
    pred_df = pred_df[pred_df["prob"] >= thresh]
    pred_t = (pred_df["t_idx"].to_numpy() *
              float(secs_per_frame)) + shift_preds
    pred_k = pred_df["key"].to_numpy() + float(pred_key_offset)
    pred_v = pred_df["vel"].to_numpy()
    #
    gt_t = gt_df["onset"].to_numpy()
    gt_k = gt_df["key"].to_numpy()
    gt_v = gt_df["vel"].to_numpy()
    # without velocity
    prec, rec, f1 = eval_note_events(
        gt_t, gt_k, pred_t, pred_k,
        tol_secs=tol_secs, pitch_tolerance=0.1)
    # with velocity
    prec_v, rec_v, f1_v = eval_note_events(
        gt_t, gt_k, pred_t, pred_k,
        gt_vels=gt_v, pred_vels=pred_v,
        tol_secs=tol_secs, pitch_tolerance=0.1,
        velocity_tolerance=tol_vel)
    #
    return (prec, rec, f1), (prec_v, rec_v, f1_v)

# ##############################################################################
# # PEDAL EVALUATION
# ##############################################################################
def eval_sus_pedal_simple(gt_events_df, pred_events_df, tol_secs=0.05):
    """Score sustain-pedal onset/offset events with one-to-one matching."""
    if len(gt_events_df) == 0 and len(pred_events_df) == 0:
        return 1.0, 1.0, 1.0
    if len(gt_events_df) == 0:
        return 0.0, 1.0, 0.0
    if len(pred_events_df) == 0:
        return 0.0, 0.0, 0.0

    tp = 0
    matched_gt = set()
    for _, pred_row in pred_events_df.iterrows():
        candidates = gt_events_df[
            (gt_events_df["event_type"] == pred_row["event_type"])
            & (~gt_events_df.index.isin(matched_gt))
        ]
        if candidates.empty:
            continue
        distances = (candidates["onset"] - pred_row["onset"]).abs()
        best_idx = distances.idxmin()
        if distances.loc[best_idx] <= tol_secs:
            matched_gt.add(best_idx)
            tp += 1

    fp = len(pred_events_df) - tp
    fn = len(gt_events_df) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _infer_logical_pedal_count(num_channels):
    """Infer logical pedals from prediction channels.

    A single sustain pedal can now be represented by three channels
    ``[state, onset, offset]``.  Legacy state-only predictions still use one
    channel per pedal.
    """
    if num_channels > 1 and (num_channels % 3) == 0:
        return num_channels // 3
    return num_channels


def _empty_pedal_results(num_pedals=1):
    pedal_names = ["sustain", "soft", "tenuto"][:num_pedals]
    results = {
        name: {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        for name in pedal_names
    }
    results["macro_avg"] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return results


def _prepare_pedal_data(pred_pedal_probs):
    if not isinstance(pred_pedal_probs, torch.Tensor):
        pred_pedal_probs = torch.tensor(pred_pedal_probs)

    if pred_pedal_probs.dim() > 3:
        shape = pred_pedal_probs.shape
        batch_size = int(np.prod(shape[:-2]))
        pred_pedal_probs = pred_pedal_probs.reshape(batch_size, shape[-2], shape[-1])
    elif pred_pedal_probs.dim() == 2:
        if pred_pedal_probs.shape[0] == 1:
            pred_pedal_probs = pred_pedal_probs.unsqueeze(1)
        else:
            pred_pedal_probs = pred_pedal_probs.unsqueeze(0)
    elif pred_pedal_probs.dim() == 1:
        pred_pedal_probs = pred_pedal_probs.unsqueeze(0).unsqueeze(0)
    elif pred_pedal_probs.dim() == 0:
        pred_pedal_probs = pred_pedal_probs.view(1, 1, 1)

    if pred_pedal_probs.dim() != 3:
        raise ValueError(f"Unexpected pred_pedal_probs shape after normalization: {pred_pedal_probs.shape}")

    return pred_pedal_probs

def _decode_pedal_predictions(pred_pedal_probs, thresh, hysteresis=0.1,
                              min_hold_steps=2, smoothing_window=3):
    num_pedals = _infer_logical_pedal_count(pred_pedal_probs.shape[1])
    if num_pedals < 1:
        return None

    try:
        decoder = PedalDecoder(
            num_pedals=num_pedals,
            threshold=thresh,
            hysteresis=hysteresis,
            min_hold_steps=min_hold_steps,
            smoothing_window=smoothing_window,
        )
        events_df, _, _ = decoder(pred_pedal_probs)
        return events_df
    except ValueError:
        return None

def threshold_eval_pedals(
        gt_pedal_events, pred_pedal_probs, secs_per_frame, thresh=0.5,
        shift_preds=0, tol_secs=0.05, hysteresis=0.1,
        min_hold_steps=2, smoothing_window=3):
    pred_pedal_probs = _prepare_pedal_data(pred_pedal_probs)
    logical_pedals = _infer_logical_pedal_count(pred_pedal_probs.shape[1])
    events_df = _decode_pedal_predictions(
        pred_pedal_probs,
        thresh,
        hysteresis=hysteresis,
        min_hold_steps=min_hold_steps,
        smoothing_window=smoothing_window,
    )

    if events_df is None:
        return _empty_pedal_results(logical_pedals)

    if "t_idx" in events_df.columns:
        events_df["onset"] = (events_df["t_idx"].astype(float) * float(secs_per_frame)) + float(shift_preds)

    pedal_names = ["sustain", "soft", "tenuto"][:logical_pedals]
    results = {}

    for pedal_idx, pedal_name in enumerate(pedal_names):
        gt_subset = gt_pedal_events[gt_pedal_events["pedal_idx"] == pedal_idx] if "pedal_idx" in gt_pedal_events.columns else gt_pedal_events
        pred_subset = events_df[events_df["pedal_idx"] == pedal_idx]

        gt_onsets = gt_subset["onset"].to_numpy() if len(gt_subset) > 0 else np.array([])
        gt_types = gt_subset["event_type"].to_numpy() if len(gt_subset) > 0 else np.array([])

        pred_onsets = pred_subset["onset"].to_numpy() if len(pred_subset) > 0 else np.array([])
        pred_types = pred_subset["event_type"].to_numpy() if len(pred_subset) > 0 else np.array([])

        gt_events_df = pd.DataFrame({"onset": gt_onsets, "event_type": gt_types})
        pred_events_df = pd.DataFrame({"onset": pred_onsets, "event_type": pred_types})

        onset_prec, onset_rec, onset_f1 = eval_sus_pedal_simple(gt_events_df[gt_events_df["event_type"] == "onset"], pred_events_df[pred_events_df["event_type"] == "onset"], tol_secs)
        offset_prec, offset_rec, offset_f1 = eval_sus_pedal_simple(gt_events_df[gt_events_df["event_type"] == "offset"], pred_events_df[pred_events_df["event_type"] == "offset"], tol_secs)

        prec, rec, f1 = eval_sus_pedal_simple(gt_events_df, pred_events_df, tol_secs)

        results[pedal_name] = {"precision": prec, "recall": rec, "f1": f1, "onset_precision": onset_prec, "onset_recall": onset_rec, "onset_f1": onset_f1, "offset_precision": offset_prec, "offset_recall": offset_rec, "offset_f1": offset_f1}

    avg_prec = np.mean([v["precision"] for v in results.values()])
    avg_rec = np.mean([v["recall"] for v in results.values()])
    avg_f1 = np.mean([v["f1"] for v in results.values()])
    results["macro_avg"] = {"precision": avg_prec, "recall": avg_rec, "f1": avg_f1}

    return results


def pedal_grid_search(
        pedal_eval_items, secs_per_frame, thresholds, hysteresis_values,
        smoothing_windows, min_hold_steps_values, shifts, tol_secs=0.05,
        logger=None, log_prefix="XV pedal", max_logged_items=None,
        checkpoint_store=None):
    """Grid-search sustain-pedal decoder hyperparameters.

    :param pedal_eval_items: Iterable of ``(gt_pedal_events, pred_pedal_probs)``
      pairs.  Predictions are usually cached tensors returned by model inference.
    :param secs_per_frame: Seconds represented by one prediction frame.
    :param thresholds: Activation/event thresholds to evaluate.
    :param hysteresis_values: Hysteresis margins to evaluate.
    :param smoothing_windows: Moving-average smoothing windows to evaluate.
    :param min_hold_steps_values: Minimum hold durations, in frames, to evaluate.
    :param shifts: Prediction time shifts, in seconds, to evaluate.
    :param tol_secs: Event matching tolerance, in seconds.
    :param logger: Optional callable receiving progress strings.
    :param log_prefix: Prefix for optional progress messages.
    :param max_logged_items: If set, only log the first N files per combo to
      avoid very large logs during full validation searches.
    :param checkpoint_store: Optional ``EvaluationCheckpointStore`` used to
      persist one summary metric vector per hyperparameter combination.
    :returns: ``(summary, best_params, best_metrics)`` where ``summary`` maps
      ``(threshold, hysteresis, smoothing_window, min_hold_steps, shift)`` to
      ``np.array([precision, recall, f1])``.
    """
    pedal_eval_items = list(pedal_eval_items)
    summary = {}

    for thresh in thresholds:
        for hysteresis in hysteresis_values:
            for smoothing_window in smoothing_windows:
                for min_hold_steps in min_hold_steps_values:
                    for shift in shifts:
                        key = (
                            float(thresh),
                            float(hysteresis),
                            int(smoothing_window),
                            int(min_hold_steps),
                            float(shift),
                        )
                        checkpoint_key = json.dumps(key)
                        cached = (
                            checkpoint_store.get(checkpoint_key)
                            if checkpoint_store is not None else None
                        )
                        if isinstance(cached, dict) and cached.get("status") == "ok":
                            summary[key] = np.asarray(cached["metrics"], dtype=float)
                            if logger is not None:
                                logger(f"{log_prefix} checkpoint hit for {key}")
                            continue

                        metrics = []
                        for idx, (gt_pedal_df, pedal_pred) in enumerate(pedal_eval_items, 1):
                            if logger is not None and (
                                max_logged_items is None or idx <= max_logged_items
                            ):
                                logger(
                                    f"[{idx}/{len(pedal_eval_items)} {log_prefix}]: "
                                    f"threshold={key[0]}, hysteresis={key[1]}, "
                                    f"smoothing_window={key[2]}, min_hold_steps={key[3]}, "
                                    f"shift={key[4]}"
                                )
                            try:
                                pedal_results = threshold_eval_pedals(
                                    gt_pedal_df,
                                    pedal_pred,
                                    secs_per_frame,
                                    thresh=key[0],
                                    shift_preds=key[4],
                                    tol_secs=tol_secs,
                                    hysteresis=key[1],
                                    smoothing_window=key[2],
                                    min_hold_steps=key[3],
                                )
                                sustain = pedal_results.get("sustain")
                                if sustain is None:
                                    pedal_prf1 = (0.0, 0.0, 0.0)
                                else:
                                    pedal_prf1 = (
                                        sustain["precision"],
                                        sustain["recall"],
                                        sustain["f1"],
                                    )
                            except Exception as exc:
                                if logger is not None:
                                    logger(f"{log_prefix} eval failed for {key}: {exc}")
                                pedal_prf1 = (0.0, 0.0, 0.0)
                            metrics.append(pedal_prf1)

                        if len(metrics) == 0:
                            summary[key] = np.array([0.0, 0.0, 0.0])
                        else:
                            arr = np.asarray(metrics, dtype=float)
                            summary[key] = arr.mean(axis=0)

                        if checkpoint_store is not None:
                            checkpoint_store.upsert(
                                checkpoint_key,
                                {
                                    "status": "ok",
                                    "params": key,
                                    "metrics": summary[key].tolist(),
                                },
                            )

    if len(summary) == 0:
        return summary, None, np.array([0.0, 0.0, 0.0])

    best_params, best_metrics = max(summary.items(), key=lambda elt: elt[1][2])
    return summary, best_params, best_metrics
