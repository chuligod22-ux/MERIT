"""Common non-invasive adapter for the three active G11 official backbones.

The selected upstream models all accept three channels. NAFNet and Restormer
return one three-channel tensor; MPRNet returns its three stage outputs and its
official deblurring test path uses sequence item 0 as the final restoration.
MERIT keeps each upstream architecture unchanged and uses output channel 0 as
the restored grayscale image.
"""

from __future__ import annotations

import torch
from torch import nn


class FirstChannelRestorationAdapter(nn.Module):
    """Expose channel 0 of a frozen, explicitly selected upstream output."""

    def __init__(self, backbone: nn.Module, output_selector: str = "tensor") -> None:
        super().__init__()
        if output_selector not in {"tensor", "sequence_index_0"}:
            raise ValueError(f"Unsupported G11 output selector: {output_selector}")
        self.backbone = backbone
        self.output_selector = output_selector

    def _select_output(self, outputs: object) -> torch.Tensor:
        if self.output_selector == "tensor":
            if not isinstance(outputs, torch.Tensor):
                raise TypeError("G11 tensor selector requires a torch.Tensor output")
            return outputs
        if not isinstance(outputs, (list, tuple)) or not outputs:
            raise TypeError(
                "G11 sequence_index_0 selector requires a non-empty list or tuple"
            )
        selected = outputs[0]
        if not isinstance(selected, torch.Tensor):
            raise TypeError("G11 selected sequence output must be a torch.Tensor")
        return selected

    def forward_preclip(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 3:
            raise ValueError(
                "G11 input must have shape [batch, 3, height, width]; "
                f"received {tuple(inputs.shape)}"
            )

        outputs = self._select_output(self.backbone(inputs))
        if outputs.ndim != 4 or outputs.shape[0] != inputs.shape[0]:
            raise ValueError(
                "G11 backbone output must have shape [batch, channels, height, width]; "
                f"received {tuple(outputs.shape)}"
            )
        if outputs.shape[1] < 1 or outputs.shape[2:] != inputs.shape[2:]:
            raise ValueError(
                "G11 backbone output must preserve spatial size and expose at least one channel; "
                f"input={tuple(inputs.shape)}, output={tuple(outputs.shape)}"
            )
        return outputs[:, :1, :, :]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the MERIT image-domain output constrained to ``[0, 1]``."""

        return torch.clamp(self.forward_preclip(inputs), min=0.0, max=1.0)
