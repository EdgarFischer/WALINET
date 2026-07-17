from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import time

import torch


SUPPORTED_ARCHITECTURES = {
    "unet",
    "ynet",
}


@dataclass(frozen=True)
class FixedNetworkBatch:
    """
    One fixed validation batch.

    The validation set is generated exactly once before training.
    Only the final network tensors are retained, normally on the
    CPU.

    Shapes
    ------
    network_input:
        (B, 2, T)

    network_target:
        (B, 2, T)

    network_l2:
        (B, 2, T), or None
    """

    network_input: torch.Tensor
    network_target: torch.Tensor
    network_l2: torch.Tensor | None

    @property
    def batch_size(self) -> int:
        return int(
            self.network_input.shape[0]
        )

    @property
    def n_timepoints(self) -> int:
        return int(
            self.network_input.shape[-1]
        )


def normalize_architecture(
    architecture: str,
) -> str:
    """
    Validate and normalize an architecture name.
    """
    architecture = str(
        architecture
    ).lower()

    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture {architecture!r}. "
            "Use 'ynet' or 'unet'."
        )

    return architecture


def validate_network_tensor(
    *,
    tensor: torch.Tensor,
    name: str,
) -> None:
    """
    Validate one real/imaginary network tensor.
    """
    if tensor.ndim != 3:
        raise ValueError(
            f"{name} must have shape (B, C, T), "
            f"but found {tuple(tensor.shape)}."
        )

    if tensor.shape[1] != 2:
        raise ValueError(
            f"{name} must contain two channels "
            "(real and imaginary), but found "
            f"{tensor.shape[1]} channels."
        )

    if torch.is_complex(
        tensor
    ):
        raise TypeError(
            f"{name} must be real-valued."
        )

    if not torch.is_floating_point(
        tensor
    ):
        raise TypeError(
            f"{name} must use a floating-point dtype."
        )


def validate_fixed_batch(
    *,
    batch: FixedNetworkBatch,
    architecture: str,
) -> None:
    """
    Validate one stored validation batch.
    """
    architecture = normalize_architecture(
        architecture
    )

    validate_network_tensor(
        tensor=batch.network_input,
        name="network_input",
    )

    validate_network_tensor(
        tensor=batch.network_target,
        name="network_target",
    )

    if (
        batch.network_input.shape
        != batch.network_target.shape
    ):
        raise ValueError(
            "Validation input and target must have identical "
            "shapes:\n"
            f"  input:  "
            f"{tuple(batch.network_input.shape)}\n"
            f"  target: "
            f"{tuple(batch.network_target.shape)}"
        )

    if batch.network_l2 is not None:
        validate_network_tensor(
            tensor=batch.network_l2,
            name="network_l2",
        )

        if (
            batch.network_l2.shape
            != batch.network_input.shape
        ):
            raise ValueError(
                "Validation L2 input and primary input must "
                "have identical shapes:\n"
                f"  primary input: "
                f"{tuple(batch.network_input.shape)}\n"
                f"  L2 input:      "
                f"{tuple(batch.network_l2.shape)}"
            )

    if (
        architecture == "ynet"
        and batch.network_l2 is None
    ):
        raise RuntimeError(
            "YNet requires the L2-projected second input, "
            "but network_l2 is None. Enable lipid projection "
            "in the simulation configuration."
        )


def validate_prediction(
    *,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> None:
    """
    Validate model output before loss calculation.
    """
    if prediction.shape != target.shape:
        raise RuntimeError(
            "Model prediction and target have different shapes:\n"
            f"  prediction: "
            f"{tuple(prediction.shape)}\n"
            f"  target:     "
            f"{tuple(target.shape)}"
        )

    if prediction.device != target.device:
        raise RuntimeError(
            "Model prediction and target are on different "
            "devices:\n"
            f"  prediction: {prediction.device}\n"
            f"  target:     {target.device}"
        )

    if not torch.is_floating_point(
        prediction
    ):
        raise TypeError(
            "Model prediction must be floating-point."
        )


def forward_model(
    *,
    model: torch.nn.Module,
    network_input: torch.Tensor,
    network_l2: torch.Tensor | None,
    architecture: str,
) -> torch.Tensor:
    """
    Run the architecture-specific forward pass.

    UNet:
        unprojected complete input

    YNet:
        unprojected complete input
        plus L2-projected second input
    """
    architecture = normalize_architecture(
        architecture
    )

    if architecture == "unet":
        return model(
            network_input
        )

    if network_l2 is None:
        raise RuntimeError(
            "YNet requires network_l2, but no L2-projected "
            "spectrum was supplied."
        )

    return model(
        network_input,
        network_l2,
    )


@torch.no_grad()
def create_fixed_validation_batches(
    *,
    simulator,
    generator: torch.Generator,
    n_spectra: int,
    batch_size: int,
    architecture: str,
    verbose: bool = True,
) -> tuple[FixedNetworkBatch, ...]:
    """
    Generate the complete validation set exactly once.

    The simulator may run on the GPU, but only the trainer-relevant
    network tensors are retained. These tensors are copied to the
    CPU so the fixed validation set does not permanently occupy GPU
    memory.

    A separate random generator must be used for validation so that
    validation-set generation does not change the training random
    sequence.
    """
    architecture = normalize_architecture(
        architecture
    )

    n_spectra = int(
        n_spectra
    )

    batch_size = int(
        batch_size
    )

    if n_spectra <= 0:
        raise ValueError(
            "n_spectra must be > 0."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be > 0."
        )

    validation_batches: list[
        FixedNetworkBatch
    ] = []

    generated_spectra = 0
    batch_index = 0

    if verbose:
        print(
            "Generating fixed validation set:"
        )
        print(
            f"  spectra:    {n_spectra}"
        )
        print(
            f"  batch size: {batch_size}"
        )
        print(
            f"  device:     {simulator.device}"
        )

    while generated_spectra < n_spectra:
        current_batch_size = min(
            batch_size,
            n_spectra
            - generated_spectra,
        )

        simulated = simulator.simulate(
            batch_size=current_batch_size,
            generator=generator,
        )

        if simulated.network_l2 is None:
            network_l2_cpu = None
        else:
            network_l2_cpu = (
                simulated
                .network_l2
                .detach()
                .cpu()
                .contiguous()
            )

        fixed_batch = FixedNetworkBatch(
            network_input=(
                simulated
                .network_input
                .detach()
                .cpu()
                .contiguous()
            ),
            network_target=(
                simulated
                .network_target
                .detach()
                .cpu()
                .contiguous()
            ),
            network_l2=(
                network_l2_cpu
            ),
        )

        validate_fixed_batch(
            batch=fixed_batch,
            architecture=architecture,
        )

        validation_batches.append(
            fixed_batch
        )

        generated_spectra += (
            current_batch_size
        )

        batch_index += 1

        if verbose:
            print(
                f"  Validation batch "
                f"{batch_index:03d}: "
                f"{generated_spectra}/"
                f"{n_spectra}"
            )

        del simulated

    if generated_spectra != n_spectra:
        raise RuntimeError(
            "Generated validation-set size does not match "
            "the requested number of spectra."
        )

    if not validation_batches:
        raise RuntimeError(
            "No validation batches were generated."
        )

    if verbose:
        print(
            "Fixed validation set ready:"
        )
        print(
            f"  batches: "
            f"{len(validation_batches)}"
        )
        print(
            f"  spectra: "
            f"{generated_spectra}"
        )

    return tuple(
        validation_batches
    )


def train_one_epoch(
    *,
    model: torch.nn.Module,
    simulator,
    generator: torch.Generator,
    optimizer: torch.optim.Optimizer,
    loss_func,
    architecture: str,
    batch_size: int,
    n_batches: int,
    verbose: bool,
    device: torch.device,
    epoch: int,
) -> tuple[
    torch.nn.Module,
    float,
]:
    """
    Train for one epoch using freshly simulated spectra.

    Every iteration generates a completely new batch. There is no
    Dataset, DataLoader, stored training set, or additional
    augmentation stage.
    """
    architecture = normalize_architecture(
        architecture
    )

    batch_size = int(
        batch_size
    )

    n_batches = int(
        n_batches
    )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be > 0."
        )

    if n_batches <= 0:
        raise ValueError(
            "n_batches must be > 0 for on-the-fly training."
        )

    if torch.device(
        simulator.device
    ) != torch.device(
        device
    ):
        raise RuntimeError(
            "Simulator and model must use the same device:\n"
            f"  simulator: {simulator.device}\n"
            f"  model:     {device}"
        )

    model.train()

    total_weighted_loss = 0.0
    total_spectra = 0

    epoch_start = (
        time.perf_counter()
    )

    for batch_index in range(
        n_batches
    ):
        batch_start = (
            time.perf_counter()
        )

        simulated = simulator.simulate(
            batch_size=batch_size,
            generator=generator,
        )

        if torch.device(
            simulated.device
        ) != torch.device(
            device
        ):
            raise RuntimeError(
                "Simulated batch is on the wrong device:\n"
                f"  batch: {simulated.device}\n"
                f"  model: {device}"
            )

        network_input = (
            simulated.network_input
        )

        network_target = (
            simulated.network_target
        )

        network_l2 = (
            simulated.network_l2
        )

        if (
            architecture == "ynet"
            and network_l2 is None
        ):
            raise RuntimeError(
                "YNet training requires an L2-projected "
                "second input, but network_l2 is None."
            )

        optimizer.zero_grad(
            set_to_none=True
        )

        prediction = forward_model(
            model=model,
            network_input=network_input,
            network_l2=network_l2,
            architecture=architecture,
        )

        validate_prediction(
            prediction=prediction,
            target=network_target,
        )

        loss = loss_func(
            prediction,
            network_target,
        )

        if loss.ndim != 0:
            raise RuntimeError(
                "The training loss function must return one "
                "scalar value."
            )

        if not bool(
            torch.isfinite(
                loss
            )
        ):
            raise FloatingPointError(
                "Training loss is non-finite at "
                f"epoch {epoch + 1}, "
                f"batch {batch_index + 1}."
            )

        loss.backward()

        optimizer.step()

        loss_value = float(
            loss.detach().item()
        )

        current_batch_size = int(
            simulated.batch_size
        )

        total_weighted_loss += (
            loss_value
            * current_batch_size
        )

        total_spectra += (
            current_batch_size
        )

        batch_time = (
            time.perf_counter()
            - batch_start
        )

        if verbose:
            log_batch = (
                " ~ Epoch: {:03d}, "
                "Batch: ({:03d}/{:03d}), "
                "Loss: {:.10f}, "
                "Time: {:.4f}, "
                "Retries: {}"
            )

            print(
                log_batch.format(
                    epoch + 1,
                    batch_index + 1,
                    n_batches,
                    loss_value,
                    batch_time,
                    simulated.retries_used,
                )
            )

        del (
            simulated,
            network_input,
            network_target,
            network_l2,
            prediction,
            loss,
        )

    if total_spectra <= 0:
        raise RuntimeError(
            "Training epoch did not process any spectra."
        )

    epoch_loss = (
        total_weighted_loss
        / total_spectra
    )

    epoch_time = (
        time.perf_counter()
        - epoch_start
    )

    print(
        (
            "Epoch: {:03d}, "
            "Loss: {:.10f}, "
            "Spectra: {}, "
            "Time: {:.4f}"
        ).format(
            epoch + 1,
            epoch_loss,
            total_spectra,
            epoch_time,
        )
    )

    return (
        model,
        epoch_loss,
    )


def validate_one_epoch(
    *,
    model: torch.nn.Module,
    validation_batches: Sequence[
        FixedNetworkBatch
    ],
    loss_func,
    architecture: str,
    verbose: bool,
    device: torch.device,
    epoch: int,
) -> float:
    """
    Validate on the fixed validation set.

    No new validation data are simulated here. Every epoch is
    therefore evaluated using exactly the same spectra.
    """
    architecture = normalize_architecture(
        architecture
    )

    if len(
        validation_batches
    ) == 0:
        raise ValueError(
            "validation_batches must not be empty."
        )

    model.eval()

    total_weighted_loss = 0.0
    total_spectra = 0

    epoch_start = (
        time.perf_counter()
    )

    with torch.inference_mode():
        for batch_index, fixed_batch in enumerate(
            validation_batches
        ):
            batch_start = (
                time.perf_counter()
            )

            validate_fixed_batch(
                batch=fixed_batch,
                architecture=architecture,
            )

            network_input = (
                fixed_batch
                .network_input
                .to(
                    device=device,
                )
            )

            network_target = (
                fixed_batch
                .network_target
                .to(
                    device=device,
                )
            )

            if fixed_batch.network_l2 is None:
                network_l2 = None
            else:
                network_l2 = (
                    fixed_batch
                    .network_l2
                    .to(
                        device=device,
                    )
                )

            prediction = forward_model(
                model=model,
                network_input=network_input,
                network_l2=network_l2,
                architecture=architecture,
            )

            validate_prediction(
                prediction=prediction,
                target=network_target,
            )

            loss = loss_func(
                prediction,
                network_target,
            )

            if loss.ndim != 0:
                raise RuntimeError(
                    "The validation loss function must return "
                    "one scalar value."
                )

            if not bool(
                torch.isfinite(
                    loss
                )
            ):
                raise FloatingPointError(
                    "Validation loss is non-finite at "
                    f"epoch {epoch + 1}, "
                    f"batch {batch_index + 1}."
                )

            loss_value = float(
                loss.item()
            )

            current_batch_size = (
                fixed_batch.batch_size
            )

            total_weighted_loss += (
                loss_value
                * current_batch_size
            )

            total_spectra += (
                current_batch_size
            )

            batch_time = (
                time.perf_counter()
                - batch_start
            )

            if verbose:
                print(
                    (
                        " ~ ValEp: {:03d}, "
                        "Batch: ({:03d}/{:03d}), "
                        "Loss: {:.10f}, "
                        "Time: {:.4f}"
                    ).format(
                        epoch + 1,
                        batch_index + 1,
                        len(
                            validation_batches
                        ),
                        loss_value,
                        batch_time,
                    )
                )

            del (
                network_input,
                network_target,
                network_l2,
                prediction,
                loss,
            )

    if total_spectra <= 0:
        raise RuntimeError(
            "Validation did not process any spectra."
        )

    epoch_loss = (
        total_weighted_loss
        / total_spectra
    )

    epoch_time = (
        time.perf_counter()
        - epoch_start
    )

    print(
        (
            "ValEp: {:03d}, "
            "Loss: {:.10f}, "
            "Spectra: {}, "
            "Time: {:.4f}"
        ).format(
            epoch + 1,
            epoch_loss,
            total_spectra,
            epoch_time,
        )
    )

    return epoch_loss