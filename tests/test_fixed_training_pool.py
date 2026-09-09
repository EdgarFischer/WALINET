from types import SimpleNamespace

import torch

from walinet.training.training import FixedNetworkBatch, train_one_epoch


class _UnusedSimulator:
    device = torch.device("cpu")
    config = SimpleNamespace(
        acquisition=SimpleNamespace(
            zero_filling=True,
            max_acquired_n_timepoints=4,
        )
    )

    def simulate(self, **kwargs):
        raise AssertionError("Fixed training must not simulate new spectra.")


def test_fixed_training_reuses_pool_for_requested_number_of_batches() -> None:
    pool_values = torch.arange(3, dtype=torch.float32).view(3, 1, 1)
    pool_values = pool_values.expand(-1, 2, 4).contiguous()
    fixed_pool = FixedNetworkBatch(
        network_input=pool_values,
        network_target=2 * pool_values,
        network_l2=None,
    )

    seen_batch_sizes: list[int] = []
    model = torch.nn.Conv1d(2, 2, kernel_size=1)
    model.register_forward_pre_hook(
        lambda _model, args: seen_batch_sizes.append(args[0].shape[0])
    )

    train_one_epoch(
        model=model,
        simulator=_UnusedSimulator(),
        generator=torch.Generator().manual_seed(1),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        loss_func=torch.nn.MSELoss(),
        architecture="unet",
        batch_size=2,
        n_batches=4,
        verbose=False,
        device=torch.device("cpu"),
        epoch=0,
        fixed_training_pool=fixed_pool,
        fixed_generator=torch.Generator().manual_seed(2),
    )

    assert seen_batch_sizes == [2, 2, 2, 2]
