#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""Utilities for sustain-pedal frame and transition targets."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PedalTargets:
    """Frame-state and explicit transition targets for one or more pedals.

    All tensors use shape ``(..., pedals, frames)`` or any compatible shape with
    time on the final axis.  Values are floating-point ``0``/``1`` tensors ready
    for BCE-style losses.
    """

    state: torch.Tensor
    onset: torch.Tensor
    offset: torch.Tensor


def widen_binary_targets(targets: torch.Tensor, width: int = 0) -> torch.Tensor:
    """Dilate sparse binary targets by ``width`` frames on each side.

    ``width=0`` returns the original tensor.  Time is assumed to be the final
    dimension; all leading dimensions are flattened temporarily so the helper can
    be used for batched one-pedal or multi-pedal tensors.
    """

    width = int(width)
    if width <= 0 or targets.shape[-1] == 0:
        return targets

    original_shape = targets.shape
    flat = targets.reshape(-1, 1, original_shape[-1])
    padded = F.pad(flat, (width, width), mode="constant", value=0.0)
    widened = F.max_pool1d(padded, kernel_size=(2 * width) + 1, stride=1)
    return widened.reshape(original_shape).clamp_(0, 1)


def sustain_pedal_targets_from_values(
    pedal_values: torch.Tensor,
    threshold: float,
    transition_width: int = 0,
    align_to_model_diff: bool = True,
) -> PedalTargets:
    """Create sustain-pedal state/down/up targets from quantized pedal values.

    :param pedal_values: Tensor with time on the final axis, usually the sustain
      pedal roll read from HDF5 with shape ``(batch, 1, frames)``.
    :param threshold: MIDI values strictly greater than this are active, matching
      :class:`ov_piano.data.midi.MidiToPianoRoll` pedal semantics.
    :param transition_width: Optional number of frames to dilate onset/offset
      targets on each side for less brittle transition supervision.
    :param align_to_model_diff: The Onsets-and-Velocities model predicts ``T-1``
      frames from first-order spectrogram differences.  When true, returned
      targets are aligned to model outputs by comparing every frame ``t`` against
      ``t-1`` and returning frames ``1..T-1``.  This preserves transitions that
      happen at the first predicted frame of a chunk.
    """

    if pedal_values.dim() == 0:
        raise ValueError("pedal_values must have a time dimension")
    target_dtype = pedal_values.dtype if pedal_values.is_floating_point() else torch.float32
    active = (pedal_values > threshold).to(dtype=target_dtype)

    if align_to_model_diff:
        current = active[..., 1:]
        previous = active[..., :-1]
    else:
        current = active
        previous = torch.cat(
            [torch.zeros_like(active[..., :1]), active[..., :-1]],
            dim=-1,
        )

    transitions = current - previous
    onsets = (transitions > 0).to(dtype=target_dtype)
    offsets = (transitions < 0).to(dtype=target_dtype)
    onsets = widen_binary_targets(onsets, transition_width)
    offsets = widen_binary_targets(offsets, transition_width)
    return PedalTargets(state=current, onset=onsets, offset=offsets)