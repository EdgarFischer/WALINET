"""Per-spectrum metrics for nuisance-removal evaluation."""

from __future__ import annotations

import torch


def channels_to_complex(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 3 or value.shape[1] < 2:
        raise ValueError(f"Expected (B, >=2, T), got {tuple(value.shape)}")
    return torch.complex(value[:, 0, :], value[:, 1, :])


def relative_l2(error: torch.Tensor, reference: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    numerator = torch.sum(torch.abs(error).square(), dim=-1)
    denominator = torch.sum(torch.abs(reference).square(), dim=-1).clamp_min(eps)
    return torch.sqrt(numerator / denominator)


def calculate_errors(
    *,
    network_input: torch.Tensor,
    true_nuisance: torch.Tensor,
    predicted_nuisance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_complex = channels_to_complex(network_input)
    truth_complex = channels_to_complex(true_nuisance)
    prediction_complex = channels_to_complex(predicted_nuisance)
    nuisance_error = relative_l2(prediction_complex - truth_complex, truth_complex)

    true_metabolite_residual = input_complex - truth_complex
    predicted_metabolite_residual = input_complex - prediction_complex
    metabolite_error = relative_l2(
        predicted_metabolite_residual - true_metabolite_residual,
        true_metabolite_residual,
    )
    return nuisance_error, metabolite_error
