#!/usr/bin/env python
# -*- coding:utf-8 -*-


"""
This module hosts the main DNNs, making use of PyTorch built-ins and parts from
``building_blocks``.
"""


import torch
from .building_blocks import get_relu, SubSpectralNorm, Permuter
from .building_blocks import ContextAwareModule, DepthwiseConv2d, conv1x1net
from ..utils import init_weights


class ResidualTemporalConvBlock(torch.nn.Module):
    """Residual dilated temporal convolution block for pedal prediction.

    The main onset/velocity backbone produces note-aligned features with shape
    ``(batch, channels, keys, time)``.  After the pedal branch collapses the key
    axis to one bin, this block applies two dilated convolutions along the time
    axis only.  Stacking blocks with increasing dilation gives the pedal heads a
    broad temporal receptive field while preserving the input/output shape.
    """

    def __init__(
            self, channels, kernel_width=5, dilation=1, bn_momentum=0.1,
            leaky_relu_slope=0.1, dropout_p=0.1):
        """
        :param int channels: Number of input/output feature channels.
        :param int kernel_width: Odd temporal convolution kernel width.
        :param int dilation: Temporal dilation factor for both convolutions.
        """
        super().__init__()
        assert (kernel_width % 2) == 1, "Only odd temporal kernels supported!"
        padding = (0, (kernel_width // 2) * dilation)
        self.kernel_width = kernel_width
        self.dilation = dilation
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(
                channels, channels, (1, kernel_width),
                padding=padding, dilation=(1, dilation), bias=False),
            torch.nn.BatchNorm2d(channels, momentum=bn_momentum),
            get_relu(leaky_relu_slope),
            torch.nn.Dropout2d(dropout_p),
            torch.nn.Conv2d(
                channels, channels, (1, kernel_width),
                padding=padding, dilation=(1, dilation), bias=False),
            torch.nn.BatchNorm2d(channels, momentum=bn_momentum),
        )
        self.activation = get_relu(leaky_relu_slope)

    def forward(self, x):
        """
        :param x: Tensor of shape ``(batch, channels, 1, time)``.
        :returns: Tensor with the same shape as ``x``.
        """
        return self.activation(x + self.net(x))


# ##############################################################################
# # MAIN MODEL
# ##############################################################################
class OnsetsAndVelocities(torch.nn.Module):
    """
    Model from 'Onsets and Velocities: Affordable Real-Time Piano Transcription
    Using Convolutional Neural Networks' (Fernandez, 2023).
    """

    STEM_NUM_CAMS = 3
    STEM_CAM_HDC_CHANS = 4
    STEM_CAM_SE_BOTTLENECK = 8
    STEM_CAM_KSIZES = ((3, 5), (3, 5), (3, 5), (3, 5))
    STEM_CAM_DILATIONS = ((1, 1), (1, 2), (1, 3), (1, 4))
    STEM_CAM_PADDINGS = ((1, 2), (1, 4), (1, 6), (1, 8))
    #
    NUM_ONSET_STAGES = 3
    #
    OSTAGE_NUM_CAMS = 3
    OSTAGE_CAM_HDC_CHANS = 4
    OSTAGE_CAM_SE_BOTTLENECK = 8
    OSTAGE_CAM_KSIZES = ((1, 11), (1, 11), (1, 11))
    OSTAGE_CAM_DILATIONS = ((1, 1), (1, 2), (1, 3))
    OSTAGE_CAM_PADDINGS = ((0, 5), (0, 10), (0, 15))
    #
    VSTAGE_NUM_CAMS = 1
    VSTAGE_CAM_HDC_CHANS = 4
    VSTAGE_CAM_SE_BOTTLENECK = 8
    VSTAGE_CAM_KSIZES = ((1, 11), (1, 11), (1, 11))
    VSTAGE_CAM_DILATIONS = ((1, 1), (1, 2), (1, 3))
    VSTAGE_CAM_PADDINGS = ((0, 5), (0, 10), (0, 15))
    # Pedal output channels: sustain state, pedal-down event, pedal-up event.
    # The explicit transition channels are supervised during pedal fine-tuning
    # and decoded directly for better event timing than state threshold crossings.
    PEDAL_STATE_IDX = 0
    PEDAL_ONSET_IDX = 1
    PEDAL_OFFSET_IDX = 2
    PEDAL_NUM_OUTPUTS = 3
    PEDAL_HEAD_NAMES = (
        "pedal_state_head",
        "pedal_onset_head",
        "pedal_offset_head",
    )
    PEDAL_TEMPORAL_KERNEL = 5
    PEDAL_TEMPORAL_DILATIONS = (1, 2, 4, 8, 16)

    @staticmethod
    def get_cam_stage(in_chans, out_bins, conv1x1head=(200, 200),
                      num_cam_bottlenecks=3, cam_hdc_chans=4,
                      cam_se_bottleneck=8,
                      cam_ksizes=((1, 10), (1, 10), (1, 10)),
                      cam_dilations=((1, 1), (1, 2), (1, 3)),
                      cam_paddings=((0, 4), (0, 8), (0, 12)),
                      bn_momentum=0.1, leaky_relu_slope=0.1, dropout_p=0.1,
                      summary_width=3, conv1x1_kw=1):
        """
        Retrieve a CAM stage, which is a ``torch.nn.Sequential``. Given a
        tensor of shape ``(b, c, h, t)``, returns ``(b, 1, out_bins, t)``
        performing the following operations:

        1. Conv2D (with BN and lReLU) to expand channels
        2. A sequence of ``num_cam_bottlenecks`` CAMs (with BN and lReLU)
        3. Conv2D (with BN and lReLU) to collapse height and channels into
          ``out_bins`` channels
        4. conv1x1net (with BN, dropout and lReLU among layers), to perform
          MLP-alike operations for each entry in dimension ``t``.
        5. Swap channels with height, and return result
        """
        cam_out_chans = cam_hdc_chans * len(cam_ksizes)
        cam = torch.nn.Sequential(
            # from (b, in, h, t) to (b, cam_out, h, t)
            torch.nn.Conv2d(in_chans, cam_out_chans, (1, 1),
                            padding=(0, 0), bias=False),
            torch.nn.BatchNorm2d(cam_out_chans, momentum=bn_momentum),
            get_relu(leaky_relu_slope),
            *[torch.nn.Sequential(
                # shape-preserving
                ContextAwareModule(
                    cam_out_chans, cam_hdc_chans, cam_se_bottleneck,
                    cam_ksizes, cam_dilations, cam_paddings, bn_momentum),
                torch.nn.BatchNorm2d(cam_out_chans, momentum=bn_momentum),
                get_relu(leaky_relu_slope))
              for _ in range(num_cam_bottlenecks)],
            # from (b, cam_out, h, t) to (b, first_hid, 1, t)
            torch.nn.Conv2d(
                cam_out_chans, conv1x1head[0], (out_bins, summary_width),
                padding=(0, 1), bias=False),
            torch.nn.BatchNorm2d(conv1x1head[0], momentum=bn_momentum),
            get_relu(leaky_relu_slope),
            # from (b, first_hid, 1, t) to (b, out_bins, 1, t)
            conv1x1net((*conv1x1head, out_bins), bn_momentum,
                       last_layer_bn_relu=False,
                       dropout_drop_p=dropout_p,
                       leaky_relu_slope=leaky_relu_slope,
                       kernel_width=conv1x1_kw),
            # reshape to (b, 1, out_bins, t)
            Permuter(0, 2, 1, 3))
        #
        return cam  # (b, 1, out_bins, t)

    def __init__(self, in_chans, in_height, out_height, conv1x1head=(200, 200),
                 bn_momentum=0.1, leaky_relu_slope=0.1, dropout_drop_p=0.1,
                 init_fn=torch.nn.init.kaiming_normal_, se_init_bias=1.0):
        """
        """
        super().__init__()
        #
        stem_chans = self.STEM_CAM_HDC_CHANS * len(self.STEM_CAM_KSIZES)
        vel_in_chans = stem_chans + 1
        #
        self.specnorm = SubSpectralNorm(
            in_chans, in_height, in_height, bn_momentum)
        #
        self.stem = torch.nn.Sequential(
            # lift in chans into stem chans
            torch.nn.Conv2d(in_chans, stem_chans, (3, 3),
                            padding=(1, 1), bias=False),
            torch.nn.BatchNorm2d(stem_chans, momentum=bn_momentum),
            get_relu(leaky_relu_slope),
            # series of stem CAMs. Output: (b, stem_chans, mels, t)
            *[torch.nn.Sequential(
                ContextAwareModule(
                    stem_chans, self.STEM_CAM_HDC_CHANS,
                    self.STEM_CAM_SE_BOTTLENECK, self.STEM_CAM_KSIZES,
                    self.STEM_CAM_DILATIONS, self.STEM_CAM_PADDINGS,
                    bn_momentum),
                torch.nn.BatchNorm2d(stem_chans, momentum=bn_momentum),
                get_relu(leaky_relu_slope))
              for _ in range(self.STEM_NUM_CAMS)],
            # reshape to ``(b, stem_chans, keys, t)``
            DepthwiseConv2d(
                stem_chans, stem_chans, in_height, out_height,
                kernel_width=1, bias=False),
            torch.nn.BatchNorm2d(stem_chans, momentum=bn_momentum),
            get_relu(leaky_relu_slope))
        #
        self.onset_stages = torch.nn.ModuleList(
            [torch.nn.Sequential(
                self.get_cam_stage(
                    stem_chans, out_height, conv1x1head,
                    self.OSTAGE_NUM_CAMS, self.OSTAGE_CAM_HDC_CHANS,
                    self.OSTAGE_CAM_SE_BOTTLENECK, self.OSTAGE_CAM_KSIZES,
                    self.OSTAGE_CAM_DILATIONS, self.OSTAGE_CAM_PADDINGS,
                    bn_momentum, leaky_relu_slope, dropout_drop_p),
                SubSpectralNorm(1, out_height, out_height, bn_momentum))
             for _ in range(self.NUM_ONSET_STAGES)])
        #
        self.velocity_stage = torch.nn.Sequential(
            self.get_cam_stage(
                    vel_in_chans, out_height, conv1x1head,
                    self.VSTAGE_NUM_CAMS, self.VSTAGE_CAM_HDC_CHANS,
                    self.VSTAGE_CAM_SE_BOTTLENECK, self.VSTAGE_CAM_KSIZES,
                    self.VSTAGE_CAM_DILATIONS, self.VSTAGE_CAM_PADDINGS,
                    bn_momentum, leaky_relu_slope, dropout_drop_p),
            SubSpectralNorm(1, out_height, out_height, bn_momentum))
        # Sustain-pedal feature extractor. Pedal state is global, so collapse
        # the key axis once and then use a compact residual TCN over time.
        # With kernel 5 and dilations 1/2/4/8/16, the pedal heads can look over
        # a much wider phrase-level window than a per-frame 1x1 head while still
        # preserving the model's ``(batch, channels, time)`` output contract.
        # Separate heads below predict active state, pedal-down transition, and
        # pedal-up transition. Direct transition supervision improves event
        # timing under strict onset/offset tolerances, while the state channel
        # remains useful for frame-level monitoring and checkpoint diagnostics.
        pedal_hidden = min(96, max(32, conv1x1head[0] // 2))
        pedal_layers = [
            torch.nn.Conv2d(vel_in_chans, pedal_hidden, (out_height, 1),
                            bias=False),
            torch.nn.BatchNorm2d(pedal_hidden, momentum=bn_momentum),
            get_relu(leaky_relu_slope),
        ]
        for dilation in self.PEDAL_TEMPORAL_DILATIONS:
            pedal_layers.append(
                ResidualTemporalConvBlock(
                    pedal_hidden,
                    kernel_width=self.PEDAL_TEMPORAL_KERNEL,
                    dilation=dilation,
                    bn_momentum=bn_momentum,
                    leaky_relu_slope=leaky_relu_slope,
                    dropout_p=dropout_drop_p,
                )
            )
        self.pedal_stage = torch.nn.Sequential(*pedal_layers)
        self.pedal_state_head = torch.nn.Conv2d(pedal_hidden, 1, (1, 1))
        self.pedal_onset_head = torch.nn.Conv2d(pedal_hidden, 1, (1, 1))
        self.pedal_offset_head = torch.nn.Conv2d(pedal_hidden, 1, (1, 1))

        # initialize parameters
        if init_fn is not None:
            self.apply(lambda module: init_weights(
                module, init_fn, bias_val=0.0))
        self.apply(lambda module: self.set_se_biases(module, se_init_bias))

    @staticmethod
    def set_se_biases(module, bias_val):
        """
        Wrapper to recursively call ``set_biases`` for the CAM submodules, and
        ignore otherwise. Used in constructor.
        """
        try:
            module.se.set_biases(bias_val)
        except AttributeError:
            pass  # ignore: not a CAM module

    def pedal_modules(self):
        """Return modules that belong to the sustain-pedal prediction branch."""
        return (
            self.pedal_stage,
            self.pedal_state_head,
            self.pedal_onset_head,
            self.pedal_offset_head,
        )

    def pedal_parameters(self):
        """Yield all parameters belonging to the sustain-pedal branch."""
        for module in self.pedal_modules():
            yield from module.parameters()

    def migrate_checkpoint_state_dict(self, state_dict):
        """Migrate legacy monolithic pedal output weights into separate heads.

        Earlier pedal-aware checkpoints used one final ``pedal_stage`` 1x1
        convolution with three output channels. The index of that final
        convolution depends on the exact historical pedal extractor, so scan for
        a compatible ``pedal_stage.<idx>.weight`` tensor instead of assuming the
        current ``pedal_stage`` length. The current model keeps a shared
        temporal pedal extractor but has explicit state/onset/offset head
        modules. During non-strict loading, copying each legacy output channel
        into the matching head preserves as much learned pedal behavior as
        possible while still allowing note-only checkpoints to initialize the
        new branch from scratch.
        """
        legacy_weight_key = None
        legacy_bias_key = None
        for key, value in state_dict.items():
            key_parts = key.split(".")
            if (
                len(key_parts) == 3
                and key_parts[0] == "pedal_stage"
                and key_parts[1].isdigit()
                and key_parts[2] == "weight"
                and isinstance(value, torch.Tensor)
                and value.dim() == 4
                and value.shape[0] == self.PEDAL_NUM_OUTPUTS
                and tuple(value.shape[2:]) == (1, 1)
            ):
                legacy_weight_key = key
                legacy_bias_key = f"pedal_stage.{key_parts[1]}.bias"
                break
        if legacy_weight_key is None:
            return state_dict

        legacy_weight = state_dict[legacy_weight_key]
        migrated = state_dict.copy()
        if hasattr(state_dict, "_metadata"):
            migrated._metadata = state_dict._metadata
        legacy_bias = state_dict.get(legacy_bias_key)

        for channel_idx, head_name in enumerate(self.PEDAL_HEAD_NAMES):
            head_weight_key = f"{head_name}.weight"
            if head_weight_key not in migrated:
                migrated[head_weight_key] = legacy_weight[channel_idx:channel_idx + 1].clone()

            head_bias_key = f"{head_name}.bias"
            if (
                legacy_bias is not None
                and isinstance(legacy_bias, torch.Tensor)
                and legacy_bias.shape[0] == self.PEDAL_NUM_OUTPUTS
                and head_bias_key not in migrated
            ):
                migrated[head_bias_key] = legacy_bias[channel_idx:channel_idx + 1].clone()

        # Remove the old final convolution so it is not reported as an
        # unexpected key after successful migration.
        migrated.pop(legacy_weight_key, None)
        migrated.pop(legacy_bias_key, None)
        return migrated

    def forward_onsets(self, x):
        """
        Given a log-mel spectrogram of shape ``(b, melbins, t)``, performs
        forward pass through the NN stem, and then multi-residual-stage
        onset probability detection. Used in ``forward``.

        :returns: ``(x_stages, stem_out)``, where ``stem_out`` is a tensor of
          shape ``(b, stem_chans, keys, t-1)`` and ``x_stages`` is a list with
          one onset prediction per stage, each of shape ``(b, keys, t-1)``.
        """
        xdiff = x.diff(dim=-1)  # (b, melbins, t-1)
        # x+xdiff has shape (b, 2, melbins, t-1)
        x = torch.stack([x[:, :, 1:], xdiff]).permute(1, 0, 2, 3)
        x = self.specnorm(x)
        #
        stem_out = self.stem(x)  # (b, stem_ch, keys, t-1)
        x = self.onset_stages[0](stem_out)  # (b, 1, keys, t-1)
        #
        x_stages = [x]
        for stg in self.onset_stages[1:]:
            x = stg(stem_out) + x_stages[-1]  # residual stages
            x_stages.append(x)
        for st in x_stages:
            st.squeeze_(1)
        #
        return x_stages, stem_out

    def forward_pedals(self, features):
        """
        Predict global sustain-pedal state and transitions from note-aligned features.

        :param features: Tensor of shape ``(b, stem_chans + 1, keys, t)``.
        :returns: Tensor of shape ``(b, 3, t)`` with channels
          ``[state, onset/down, offset/up]``.
        """
        pedal_features = self.pedal_stage(features)
        return torch.cat(
            [
                self.pedal_state_head(pedal_features),
                self.pedal_onset_head(pedal_features),
                self.pedal_offset_head(pedal_features),
            ],
            dim=1,
        ).squeeze(2)

    def forward(self, x, trainable_onsets=True):
        """
        :param x: Logmel batch of shape ``(b, melbins, t)``
        :returns: ``(x_stages, velocities, pedals)``. See ``forward_onsets`` for
          a description of ``x_stages``. The ``velocities`` tensor has shape
          ``(b, keys, t-1)``, and ``pedals`` tensor has shape ``(b, 3, t-1)``
          for sustain state/down/up predictions.
        """
        if trainable_onsets:
            x_stages, stem_out = self.forward_onsets(x)
            stem_out = torch.cat([stem_out, x_stages[-1].unsqueeze(1)], dim=1)
        else:
            with torch.no_grad():
                x_stages, stem_out = self.forward_onsets(x)
                stem_out = torch.cat([stem_out, x_stages[-1].unsqueeze(1)],
                                     dim=1)
        #
        velocities = self.velocity_stage(stem_out).squeeze(1)  # (b, out_height, t)
        pedals = self.forward_pedals(stem_out)
        #
        return x_stages, velocities, pedals
