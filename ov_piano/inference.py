#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
This module contains re-usable functionality for inference:
* Convenience functionality to perform strided inference
* Decoders to convert piano roll predictions into events
"""


import pandas as pd
import torch
import torch.nn.functional as F
#
from .models.building_blocks import Nms1d, GaussianBlur1d


# ##############################################################################
# # STRIDED INFERENCE
# ##############################################################################
def strided_inference(model, x, chunk_size=10000, chunk_overlap=0):
    """
    This function is designed to allow the inference of very large signals that
    don't fit on the resources at once, by processing strided, windowed chunks
    with given window size and overlap.
    The chunks are then connected together by removing half of the overlap from
    each side.

    :param model: Functor that accepts a tensor of shape ``(b, h, t)`` and
      returns multiple outputs of shapes ``(b, h_i, t)`` (e.g. onsets and
      velocities), where ``b, t`` are identical between input and output.
    :param x: Tensor of shape ``(b, h, t)``, input to the model.
    :returns: List of tensors of shape ``(b, h_i, t)`` on CPU.
    """
    # sanity checks
    assert chunk_overlap >= 2, "overlap must be >=2!"
    assert (chunk_overlap % 2) == 0, "chunk_overlap must be even!"
    half_overlap = chunk_overlap // 2
    #
    in_b, in_h, in_w = x.shape
    stride = chunk_size - chunk_overlap
    assert stride > 0, "chunk_overlap must be smaller than chunk_size!"
    if in_w <= chunk_size:
        stride = chunk_size  # in this case only 1 chunk needed
    # compute strided inference
    # results is in the form [(out1a, out1b, ...), (out2a, out2b...)]
    results = []
    result_lengths = []
    for beg in range(0, in_w, stride):
        chunk = x[..., beg:beg+chunk_size]
        outputs = model(chunk)
        # Guard against invalid or unexpected model returns
        if outputs is None:
            outputs = []
        try:
            outputs = [o.cpu().detach() for o in outputs]
        except Exception:
            # Try to coerce single-tensor return
            if isinstance(outputs, torch.Tensor):
                try:
                    outputs = [outputs.cpu().detach()]
                except Exception:
                    outputs = []
            else:
                outputs = []

        # If outputs is empty, skip this chunk
        if len(outputs) == 0:
            result_lengths.append(chunk.shape[-1])
            del chunk
            # ensure outputs removed
            try:
                del outputs
            except Exception:
                pass
            continue

        # Handle variable number of model outputs (for compatibility)
        assert len(outputs) >= 2, "Model must return at least 2 outputs (probs, vels)"

        # Validate all outputs have correct batch size and time dimension
        assert all(o.shape[0] == chunk.shape[0] for o in outputs), \
            "all b_outputs must equal b_in!"
        assert all(o.shape[-1] == chunk.shape[-1] for o in outputs), \
            "all t_outputs must equal t_in!"

        results.append(outputs)
        result_lengths.append(chunk.shape[-1])
        del chunk
        del outputs

    # For >1 chunks, at most 1 partial-length chunk at the end is allowed
    valid_chunks = sum(x == chunk_size for x in result_lengths)
    extra_chunks = int(sum(x != chunk_size for x in result_lengths) > 0)
    results = results[:(valid_chunks + extra_chunks)]

    # gather concatenated results
    t_results = []
    for result in map(list, zip(*results)):
        # If we have 1 chunk, return as-is, no cuts needed
        if len(result) > 1:
            result[0] = result[0][..., :-half_overlap]
            result[-1] = result[-1][..., half_overlap:]
            for i in range(1, len(result) - 1):
                result[i] = result[i][..., half_overlap:-half_overlap]
        result = torch.cat(result, dim=-1)

        assert x.shape[0] == result.shape[0], \
            f"Result b_out must equal b_in! {(x.shape, result.shape)}"
        assert x.shape[-1] == result.shape[-1], \
            f"Result t_out must equal t_in! {(x.shape, result.shape)}"
        t_results.append(result)
    #
    return t_results


# ##############################################################################
# # ONSET DECODERS
# ##############################################################################
class OnsetNmsDecoder(torch.nn.Module):
    """
    Simple pianoroll to onsets decoder. Given a pianoroll with detected onset
    probabilites:
    1. Optionally applies Gaussian smoothening across time dimension
    2. Removes non-maxima
    3. Extracts indexes of maxima as the onsets
    """

    def __init__(self, num_keys, nms_pool_ksize=3, gauss_conv_stddev=None,
                 gauss_conv_ksize=None):
        """
        :param num_keys: Expected input to forward is ``(b, num_keys, t)``.
        :param gauss_conv_stddev: If given
        :param gauss_conv_ksize: Unused if stddev is not given. If given, a
          default ksize of ``7*stddev`` will be taken, but here we can provide
          a custom ksize (sometimes needed since odd ksize is required).
        """
        super().__init__()
        self.num_keys = num_keys
        self.nms1d = Nms1d(nms_pool_ksize)
        #
        self.blur = gauss_conv_stddev is not None
        if self.blur:
            if gauss_conv_ksize is None:
                gauss_conv_ksize = round(gauss_conv_stddev * 7)
            self.gauss1d = GaussianBlur1d(
                num_keys, gauss_conv_ksize, gauss_conv_stddev)

    @staticmethod
    def idxs_to_df(batch_idxs, key_idxs, time_idxs, values):
        """
        Inputs are flat tensors of same length.
        """
        result = pd.DataFrame(
            {"batch_idx": batch_idxs.cpu(), "key": key_idxs.cpu(),
             "t_idx": time_idxs.cpu(), "value": values.cpu()})
        return result

    def refine_t(self, xmap, ymap, bbb, hhh, ttt, vvv):
        """
        Extend this method for more complex behaviour.
        """
        return ttt

    def forward(self, x):
        """
        :param x: Tensor of shape ``(b, keys, t)`` expected to contain onset
          probabilities
        :param thresholds: Activations above threshold will be considered
          predictions. Multiple thresholds can be given
        :param as_df: If true, onsets are given as pandas dataframe. Otherwise
          filtered versions of ``x`` are returned.
        :returns: One pandas dataframe per given threshold, with columns
          containing the onsets in the form ``b_idx, key, t_idx, value``
        """
        assert 0 <= x.min() <= x.max() <= 1, \
            "Input is expected to contain probabilities in range [0, 1]!"
        norm_factor = 1  # useful to re-calibrate threshold
        with torch.no_grad():
            # optional blur
            y = x
            if self.blur:
                prev_max = x.max()
                if prev_max > 0:
                    y = self.gauss1d(y)
                    norm_factor = (prev_max / x.max()).item()
                    if norm_factor != 1:
                        y = y * norm_factor
            # nms
            y = self.nms1d(y)
        # extract NMS indexes and perform refinement
        bbb, hhh, ttt = y.nonzero(as_tuple=True)
        vvv = y[bbb, hhh, ttt]
        refined_t = self.refine_t(x, y, bbb, hhh, ttt, vvv)
        df = self.idxs_to_df(bbb, hhh, refined_t, vvv)
        return df, norm_factor


# ##############################################################################
# # ONSET+VELOCITY DECODERS
# ##############################################################################
class OnsetVelocityNmsDecoder(torch.nn.Module):
    """
    Modification of ``OnsetNmsdecoder``, that also processes velocities. Given
    a pianoroll with detected onset probabilites, and an analogous roll with
    predicted velocities:
    1. Detects onsets in the same way as ``OnsetNmsdecoder``
    2. Reads the velocity at the detected onsets from the given velocity maps
    3. Returns onset positions, probabilities and velocities
    """

    def __init__(self, num_keys, nms_pool_ksize=3, gauss_conv_stddev=None,
                 gauss_conv_ksize=None, vel_pad_left=1, vel_pad_right=1):
        """
        :param num_keys: Expected input to forward is ``(b, num_keys, t)``.
        :param gauss_conv_stddev: If given
        :param gauss_conv_ksize: Unused if stddev is not given. If given, a
          default ksize of ``7*stddev`` will be taken, but here we can provide
          a custom ksize (sometimes needed since odd ksize is required).
        :param vel_pad_left: When checking the predicted velocity, how many
         indexes to the left to the peak are regarded (average of all regarded
         entries is computed).
        :param vel_pad_right: See ``vel_pad_left``.
        """
        super().__init__()
        self.num_keys = num_keys
        self.nms1d = Nms1d(nms_pool_ksize)
        #
        self.blur = gauss_conv_stddev is not None
        if self.blur:
            if gauss_conv_ksize is None:
                gauss_conv_ksize = round(gauss_conv_stddev * 7)
            self.gauss1d = GaussianBlur1d(
                num_keys, gauss_conv_ksize, gauss_conv_stddev)
        #
        self.vel_pad_left = vel_pad_left
        self.vel_pad_right = vel_pad_right

    @staticmethod
    def read_velocities(velmap, batch_idxs, key_idxs, t_idxs,
                        pad_l=0, pad_r=0):
        """
        Given:
        1. A tensor of shape ``(b, k, t)``
        2. Indexes corresponding to points in the tensor
        3. Potential span to the left and right of points across the t dim.
        This method reads and returns the corresponding points in the tensor.
        If spans are given, the results are averaged for each span.
        """
        assert pad_l >= 0, "Negative padding not allowed!"
        assert pad_r >= 0, "Negative padding not allowed!"
        # if we read extra l/r, pad to avoid OOB (reflect to retain averages)
        if (pad_l > 0) or (pad_r > 0):
            velmap = F.pad(velmap, (pad_l, pad_r), mode="reflect")
        #
        total_readings = pad_l + pad_r + 1
        result = velmap[batch_idxs, key_idxs, t_idxs]
        for delta in range(1, total_readings):
            result += velmap[batch_idxs, key_idxs, t_idxs + delta]
        result /= total_readings
        return result

    def forward(self, onset_probs, velmap, pthresh=None):
        """
        :param onset_probs: Tensor of shape ``(b, keys, t)`` expected to
          contain onset probabilities
        :param velmap: Velocity map of same shape as onset_probs, containing
          the predicted velocity for each given entry.
        :param pthresh: Any probs below this value won't be regarded.

        """
        assert 0 <= onset_probs.min() <= onset_probs.max() <= 1, \
            "Onset probs expected to contain probabilities in range [0, 1]!"
        assert onset_probs.shape == velmap.shape, \
            "Onset probs and velmap must have same shape!"
        # perform NMS on onset probs
        with torch.no_grad():
            # optional blur
            if self.blur:
                prev_max = onset_probs.max()
                if prev_max > 0:
                    onset_probs = self.gauss1d(onset_probs)
            onset_probs = self.nms1d(onset_probs, pthresh)
        # extract NMS indexes and prob values
        bbb, kkk, ttt = onset_probs.nonzero(as_tuple=True)
        ppp = onset_probs[bbb, kkk, ttt]
        # extract velocity readings. Reflect pad to avoid OOB and retain avgs
        vvv = self.read_velocities(velmap, bbb, kkk, ttt,
                                   self.vel_pad_left, self.vel_pad_right)
        # create dataframe and return
        df = pd.DataFrame(
            {"batch_idx": bbb.cpu(), "key": kkk.cpu(), "t_idx": ttt.cpu(),
             "prob": ppp.cpu(), "vel": vvv.cpu()})
        return df

# ##############################################################################
# # PEDAL DECODERS
# ##############################################################################
class PedalDecoder(torch.nn.Module):
    """Decode pedal predictions into onset/offset events.

    The decoder converts raw logits to probabilities, applies a light temporal
    smoothing pass, and then uses hysteresis plus a minimum hold-time rule to
    suppress chatter around threshold crossings.
    """

    def __init__(self, num_pedals=3, threshold=0.5, hysteresis=0.1,
                 min_hold_steps=2, smoothing_window=3):
        """
        :param num_pedals: Number of pedal types (default: 3 for sustain, soft, tenuto)
        :param threshold: Probability threshold for pedal activation
        :param hysteresis: Margin used to avoid rapid chatter around the threshold
        :param min_hold_steps: Minimum number of frames a state must persist before changing
        :param smoothing_window: Small moving-average window for the probability sequence
        """
        super().__init__()
        self.num_pedals = num_pedals
        self.threshold = threshold
        self.hysteresis = hysteresis
        self.min_hold_steps = max(1, int(min_hold_steps))
        self.smoothing_window = max(1, int(smoothing_window))

    @staticmethod
    def logits_to_probs(logits):
        """Convert raw logits or already-normalized probabilities to probabilities."""
        if isinstance(logits, torch.Tensor):
            min_val = float(logits.min())
            max_val = float(logits.max())
            if 0.0 <= min_val and max_val <= 1.0:
                return logits
        return torch.sigmoid(logits)

    def _smooth_probs(self, probs):
        """Apply a small moving-average smoothing over time."""
        if self.smoothing_window <= 1 or probs.shape[-1] <= 1:
            return probs
        batch_size, num_pedals, num_steps = probs.shape
        flat = probs.reshape(-1, 1, num_steps)
        kernel = torch.ones(1, 1, self.smoothing_window,
                            device=probs.device, dtype=probs.dtype)
        kernel /= self.smoothing_window
        padded = F.pad(flat, (self.smoothing_window // 2,
                              self.smoothing_window // 2),
                       mode="replicate")
        smoothed = F.conv1d(padded, kernel)
        return smoothed.reshape(batch_size, num_pedals, num_steps)

    def detect_transitions(self, probs):
        """
        Detect state transitions in pedal activation using hysteresis.
        Returns onset and offset indices for each pedal.

        :param probs: Tensor of shape (b, num_pedals, t) with values in [0, 1]
        :returns: Dictionary with "onsets", "offsets", and "states" tensors
        """
        smoothed_probs = self._smooth_probs(probs)
        probs = 0.7 * probs + 0.3 * smoothed_probs
        batch_size, num_pedals, num_steps = probs.shape
        states = torch.zeros((batch_size, num_pedals, num_steps),
                             device=probs.device, dtype=torch.float32)
        onsets = torch.zeros((batch_size, num_pedals, num_steps),
                             device=probs.device, dtype=torch.bool)
        offsets = torch.zeros((batch_size, num_pedals, num_steps),
                              device=probs.device, dtype=torch.bool)

        prev_states = torch.zeros((batch_size, num_pedals),
                                  device=probs.device, dtype=torch.float32)
        last_change = torch.full((batch_size, num_pedals), -self.min_hold_steps,
                                 device=probs.device, dtype=torch.long)
        upper = self.threshold + self.hysteresis
        lower = self.threshold - self.hysteresis

        for step in range(num_steps):
            current_probs = probs[..., step]
            next_states = prev_states.clone()
            for batch_idx in range(batch_size):
                for pedal_idx in range(num_pedals):
                    prob = float(current_probs[batch_idx, pedal_idx])
                    prev_state = int(prev_states[batch_idx, pedal_idx])
                    time_since_change = step - int(last_change[batch_idx, pedal_idx])
                    if prev_state == 0 and prob >= upper:
                        if time_since_change >= self.min_hold_steps:
                            next_states[batch_idx, pedal_idx] = 1.0
                            onsets[batch_idx, pedal_idx, step] = True
                            last_change[batch_idx, pedal_idx] = step
                    elif prev_state == 1 and prob <= lower:
                        if time_since_change >= self.min_hold_steps:
                            next_states[batch_idx, pedal_idx] = 0.0
                            offsets[batch_idx, pedal_idx, step] = True
                            last_change[batch_idx, pedal_idx] = step
            states[..., step] = next_states
            prev_states = next_states

        return {"onsets": onsets, "offsets": offsets, "states": states}

    def forward(self, pedal_logits):
        """
        :param pedal_logits: Tensor of shape (b, num_pedals, t) with raw logits
        :returns: Dictionary with pedal events for each batch and pedal type
        """
        b, p, t = pedal_logits.shape
        assert p == self.num_pedals, \
            f"Expected {self.num_pedals} pedals, got {p}"

        with torch.no_grad():
            probs = self.logits_to_probs(pedal_logits)
            transitions = self.detect_transitions(probs)

        batch_indices, pedal_indices, time_indices = transitions["onsets"].nonzero(as_tuple=True)
        onset_df = pd.DataFrame({
            "batch_idx": batch_indices.cpu(),
            "pedal_idx": pedal_indices.cpu(),
            "t_idx": time_indices.cpu(),
            "event_type": "onset"
        })

        batch_indices, pedal_indices, time_indices = transitions["offsets"].nonzero(as_tuple=True)
        offset_df = pd.DataFrame({
            "batch_idx": batch_indices.cpu(),
            "pedal_idx": pedal_indices.cpu(),
            "t_idx": time_indices.cpu(),
            "event_type": "offset"
        })

        events_df = pd.concat([onset_df, offset_df], ignore_index=True)
        events_df = events_df.sort_values(["batch_idx", "t_idx"]).reset_index(drop=True)

        return events_df, probs, transitions["states"]