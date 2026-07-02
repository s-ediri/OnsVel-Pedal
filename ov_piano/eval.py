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


import os
#
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
#
from .data.key_model import KeyboardStateMachine
from .data.midi import SingletrackMidiParser, MaestroMidiParser
from .data.midi import MidiToPianoRoll
from mir_eval.transcription import precision_recall_f1_overlap as prf1o
from mir_eval.transcription_velocity import precision_recall_f1_overlap \
    as prf1o_v


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
        """
        """
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
        """
        """
        self.dataset, self.meta_dataset = dataset, meta_dataset
        self.midi_abspaths = [self.get_metadata_path(md, meta_dataset)
                              for _, _, md in dataset]
        # Disable ProcessPoolExecutor to avoid memory issues on Windows
        # Use sequential processing instead
        midi_eventdata = [self.get_midi_eventdata(ap) for ap in self.midi_abspaths]
        self.midi_eventdata = {ap: data for ap, data
                               in zip(self.midi_abspaths, midi_eventdata)}
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
        key_events, sus_states, ten_states, soft_states, largest_ts = self.midi_eventdata[md_path]
        
        # Convert sustain pedal states to events
        sus_pedal_events = sus_states_to_events(sus_states, secs_per_frame)
        return sus_pedal_events


# ##############################################################################
# # SUSTAIN PEDAL EVENT CONVERSION
# ##############################################################################
def sus_states_to_events(sus_states_df, secs_per_frame):
    """
    Convert sustain pedal state DataFrame to onset/offset events.
    
    :param sus_states_df: DataFrame with columns ['ts', 'val'] from MIDI parsing
    :param secs_per_frame: Time per frame in seconds
    :returns: DataFrame with columns ['onset', 'event_type'] where event_type is 'onset' or 'offset'
    """
    if sus_states_df.empty:
        return pd.DataFrame(columns=['onset', 'event_type'])
    
    events = []
    prev_state = 0
    thresholds_to_try = [7, 1, 0]  # Try progressively lower thresholds
    
    for threshold in thresholds_to_try:
        events = []
        prev_state = 0
        for _, row in sus_states_df.iterrows():
            current_state = 1 if row['val'] > threshold else 0
            
            # Detect state transitions
            if prev_state == 0 and current_state == 1:
                events.append({'onset': row['ts'], 'event_type': 'onset'})
            elif prev_state == 1 and current_state == 0:
                events.append({'onset': row['ts'], 'event_type': 'offset'})
            
            prev_state = current_state
        if len(events) > 0:
            break
    
    return pd.DataFrame(events)


# ##############################################################################
# # EVENT-BASED EVALUATION
# ##############################################################################
def eval_note_events(gt_onsets, gt_keys,
                     pred_onsets, pred_keys,
                     gt_vels=None, pred_vels=None,
                     tol_secs=0.05, pitch_tolerance=0.1,
                     velocity_tolerance=0.1,
                     pred_key_shift=0, pred_onset_mul=1.0,
                     pred_shift=0):
    """
    Given sets of ground truth and predicted note events (with their onsets,
    keys, and optionally velocities), as well as the potential shift of onset
    keys and scale+shift of onset times, computes+returns the precision, recall
    and F1 score. Predictions are considered correct if they are within given
    onset time and pitch tolerances (and also velocity if given). Check the
    ``precision_recall_f1_overlap`` functions from ``mir_eval.transcription``
    and ``mir_eval.transcription_velocity`` for more details.

    :param gt_onsets: Numpy 1D array with onset timestamps in seconds.
    :param gt_keys: Numpy 1D array with same shape as gt_onsets designing the
      corresponding keys.
    :param pred_onsets: Predicted onsets, they can be of different length
      than the ground truth but needs to be a numpy array.
    :param pred_keys: see gt_keys
    :param gt_vels: Numpy 1D array with GT velocities (usually MIDI 0-127, but
      will be rescaled to 0-1 during evaluation).
    :param pred_vels: Numpy 1D array with predicted velocities. During
      evaluation, it will be scaled+shifted to the gt_vels, so scale and shift
      are not important.
    :param velocity_tolerance: Once ``pred_vels`` are scaled+shifted to best
      fit the GT, a prediction is considered true if within this tolerance.
    :param tol_secs: Tolerance for considering a true prediction, in seconds
    :param pred_onset_mul: Given pred_onsets will be multiplied by this
    :param pred_shift: **After** onsets are multiplied by ``pred_onset_mul``,
      they will be added this shift.
    """
    if len(pred_onsets) == 0:
        # if model didn't predict any onsets gather 0 for all metrics
        prec, rec, f1 = 0, 0, 0
    else:
        #
        if pred_key_shift != 0:
            pred_keys = pred_keys + pred_key_shift
        if pred_onset_mul != 1.0:
            pred_onsets = pred_onsets * pred_onset_mul
        if pred_shift != 0:
            pred_onsets = pred_onsets + pred_shift
        # mir_eval code needs offsets, even when ignored
        gt_offsets = gt_onsets + 1
        pred_offsets = pred_onsets + 1
        # eval predictions using the mir_eval lib
        if (gt_vels is not None) and (pred_vels is not None):
            prec, rec, f1, _ = prf1o_v(
                np.stack((gt_onsets, gt_offsets)).T, gt_keys, gt_vels,
                np.stack((pred_onsets, pred_offsets)).T, pred_keys, pred_vels,
                onset_tolerance=tol_secs, pitch_tolerance=pitch_tolerance,
                velocity_tolerance=velocity_tolerance,
                offset_ratio=None,  # ignore offsets
                offset_min_tolerance=tol_secs)
        else:
            prec, rec, f1, _ = prf1o(
                np.stack((gt_onsets, gt_offsets)).T, gt_keys,
                np.stack((pred_onsets, pred_offsets)).T, pred_keys,
                onset_tolerance=tol_secs, pitch_tolerance=pitch_tolerance,
                offset_ratio=None,  # ignore offsets
                offset_min_tolerance=tol_secs)
    #
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
# # SUSTAIN PEDAL EVALUATION
# ##############################################################################
def eval_sus_pedal_simple(gt_events_df, pred_events_df, tol_secs=0.05):
    """
    Simple sustain pedal evaluation comparing onset/offset events.
    
    :param gt_events_df: Ground truth pedal events DataFrame with ['onset', 'event_type']
    :param pred_events_df: Predicted pedal events DataFrame with ['onset', 'event_type']  
    :param tol_secs: Time tolerance in seconds
    :returns: (precision, recall, f1) for sustain pedal events
    """
    if len(gt_events_df) == 0:
        if len(pred_events_df) == 0:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 1.0, 0.0
    
    if len(pred_events_df) == 0:
        return 0.0, 0.0, 0.0
    
    tp = 0  # True positives
    fp = 0  # False positives  
    fn = 0  # False negatives
    
    matched_gt = set()
    
    for _, pred_row in pred_events_df.iterrows():
        pred_time = pred_row['onset']
        pred_type = pred_row['event_type']
        
        # Find matching GT event
        best_match = None
        best_dist = tol_secs
        
        for gt_idx, gt_row in gt_events_df.iterrows():
            if gt_idx in matched_gt:
                continue
            
            if gt_row['event_type'] != pred_type:
                continue
                
            dist = abs(pred_time - gt_row['onset'])
            if dist < best_dist:
                best_match = gt_idx
                best_dist = dist
        
        if best_match is not None:
            tp += 1
            matched_gt.add(best_match)
        else:
            fp += 1
    
    fn = len(gt_events_df) - len(matched_gt)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


# ##############################################################################
# # PEDAL EVALUATION
# ##############################################################################
def eval_pedal_events(gt_onsets, gt_pedals, pred_onsets, pred_pedals, 
                      tol_secs=0.05):
    """
    Evaluate pedal event detection accuracy.
    
    Similar to eval_note_events but for binary pedal state detection.
    Compares ground truth vs predicted pedal onset/offset times.
    
    :param gt_onsets: Ground truth pedal event times (seconds)
    :param gt_pedals: Ground truth pedal states (0=off, 1=on)
    :param pred_onsets: Predicted pedal event times (seconds)
    :param pred_pedals: Predicted pedal states (0=off, 1=on)
    :param tol_secs: Time tolerance in seconds (default 50ms)
    :returns: Tuple (precision, recall, f1_score)
    """
    # If no events, return perfect score for empty predictions, 0 otherwise
    if len(gt_onsets) == 0:
        if len(pred_onsets) == 0:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 1.0, 0.0
    
    if len(pred_onsets) == 0:
        return 0.0, 0.0, 0.0
    
    # Match predictions to ground truth within tolerance window
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives
    
    matched_gt = set()
    
    for pred_t, pred_state in zip(pred_onsets, pred_pedals):
        # Find closest GT event within tolerance
        best_match = None
        best_dist = tol_secs
        
        for gt_idx, (gt_t, gt_state) in enumerate(zip(gt_onsets, gt_pedals)):
            if gt_idx in matched_gt:
                continue
            
            dist = abs(pred_t - gt_t)
            if dist < best_dist and pred_state == gt_state:
                best_match = gt_idx
                best_dist = dist
        
        if best_match is not None:
            tp += 1
            matched_gt.add(best_match)
        else:
            fp += 1
    
    # Remaining unmatched GTs are false negatives
    fn = len(gt_onsets) - len(matched_gt)
    
    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) \
        if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def _score_event_type(gt_events_df, pred_events_df, tol_secs):
    if len(gt_events_df) == 0 and len(pred_events_df) == 0:
        return 1.0, 1.0, 1.0
    if len(gt_events_df) == 0:
        return 0.0, 1.0, 0.0
    if len(pred_events_df) == 0:
        return 0.0, 0.0, 0.0
    return eval_sus_pedal_simple(gt_events_df, pred_events_df,
                                  tol_secs=tol_secs)


def threshold_eval_pedals(gt_pedal_events, pred_pedal_probs, secs_per_frame,
                          thresh=0.5, shift_preds=0, tol_secs=0.05):
    """
    Evaluate pedal event detection with thresholding.

    :param gt_pedal_events: Ground truth pedal events dataframe with columns
      [onset, pedal_idx, event_type]
    :param pred_pedal_probs: Predicted pedal probabilities (b, num_pedals, t)
    :param secs_per_frame: Conversion factor from frame index to seconds
    :param thresh: Probability threshold for pedal activation
    :param shift_preds: Time shift to apply to predictions (seconds)
    :param tol_secs: Time tolerance for matching
    :returns: Dictionary with per-pedal precision, recall, f1 scores
    """
    from .inference import PedalDecoder
    import torch

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
        raise ValueError(
            f"Unexpected pred_pedal_probs shape after normalization: {pred_pedal_probs.shape}"
        )

    num_pedals = pred_pedal_probs.shape[1]
    if num_pedals < 1:
        return {"sustain": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "soft": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "tenuto": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "macro_avg": {"precision": 0.0, "recall": 0.0, "f1": 0.0}}

    try:
        decoder = PedalDecoder(num_pedals=num_pedals, threshold=thresh)
        events_df, probs, states = decoder(pred_pedal_probs)
    except Exception:
        result = {
            "sustain": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "soft": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "tenuto": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "macro_avg": {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        }
        if num_pedals == 1:
            return {"sustain": result["sustain"], "macro_avg": result["macro_avg"]}
        return result

    if "t_idx" in events_df.columns:
        events_df["onset"] = (
            events_df["t_idx"].astype(float) * float(secs_per_frame)
            + float(shift_preds)
        )

    pedal_names = ["sustain", "soft", "tenuto"][:num_pedals]
    results = {}

    for pedal_idx, pedal_name in enumerate(pedal_names):
        gt_subset = gt_pedal_events[gt_pedal_events["pedal_idx"] == pedal_idx] if "pedal_idx" in gt_pedal_events.columns else gt_pedal_events
        pred_subset = events_df[events_df["pedal_idx"] == pedal_idx]

        gt_onsets = gt_subset["onset"].to_numpy() if len(gt_subset) > 0 else np.array([])
        gt_types = gt_subset["event_type"].to_numpy() if len(gt_subset) > 0 else np.array([])

        pred_onsets = pred_subset["onset"].to_numpy() if len(pred_subset) > 0 else np.array([])
        pred_types = pred_subset["event_type"].to_numpy() if len(pred_subset) > 0 else np.array([])

        gt_events_df = pd.DataFrame({'onset': gt_onsets, 'event_type': gt_types})
        pred_events_df = pd.DataFrame({'onset': pred_onsets, 'event_type': pred_types})

        onset_prec, onset_rec, onset_f1 = _score_event_type(
            gt_events_df[gt_events_df['event_type'] == 'onset'],
            pred_events_df[pred_events_df['event_type'] == 'onset'],
            tol_secs)
        offset_prec, offset_rec, offset_f1 = _score_event_type(
            gt_events_df[gt_events_df['event_type'] == 'offset'],
            pred_events_df[pred_events_df['event_type'] == 'offset'],
            tol_secs)

        prec, rec, f1 = _score_event_type(gt_events_df, pred_events_df,
                                          tol_secs)

        results[pedal_name] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "onset_precision": onset_prec,
            "onset_recall": onset_rec,
            "onset_f1": onset_f1,
            "offset_precision": offset_prec,
            "offset_recall": offset_rec,
            "offset_f1": offset_f1,
        }

    avg_prec = np.mean([v["precision"] for v in results.values()])
    avg_rec = np.mean([v["recall"] for v in results.values()])
    avg_f1 = np.mean([v["f1"] for v in results.values()])
    results["macro_avg"] = {"precision": avg_prec, "recall": avg_rec, "f1": avg_f1}

    return results