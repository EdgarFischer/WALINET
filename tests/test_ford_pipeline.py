import json
import sys
import types

import numpy as np

from walinet.inference import ford_pipeline


def test_pipeline_passes_walinet_output_to_ford_in_memory(tmp_path, monkeypatch):
    data = np.ones((2, 8, 3), dtype=np.complex64)
    mask = np.ones((2, 3), dtype=bool)
    data_path = tmp_path / "data.npy"
    mask_path = tmp_path / "mask.npy"
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "fit"
    config_path = tmp_path / "template.json"
    basis_path = tmp_path / "basis.mat"
    model_dir.mkdir()
    basis_path.touch()
    np.save(data_path, data)
    np.save(mask_path, mask)
    config_path.write_text(
        json.dumps(
            {
                "io_config": {"basis_path": str(basis_path)},
                "preprocessor_config": {
                    "normalize_signals": "frequency",
                    "normalization_scaling_mode": "z-score",
                },
                "pytorch_config": {
                    "device": "cpu",
                    "default_type": None,
                    "float32_matmul_precision": None,
                    "num_threads": None,
                },
            }
        )
    )

    cleaned = data * 2
    monkeypatch.setattr(ford_pipeline, "infer_fid", lambda *args, **kwargs: cleaned)
    monkeypatch.setattr(ford_pipeline.torch, "set_default_device", lambda *args: None)
    monkeypatch.setattr(ford_pipeline.torch.cuda, "empty_cache", lambda: None)

    captured = {}

    class FakeConfiguration:
        @classmethod
        def from_dict(cls, values):
            captured["config_dict"] = values
            return types.SimpleNamespace(
                pytorch_config=types.SimpleNamespace(
                    default_type=None,
                    float32_matmul_precision=None,
                    num_threads=None,
                )
            )

    class FakeProblem:
        def __init__(self, config, *, subject_data, subject_mask):
            captured["data"] = subject_data
            captured["mask"] = subject_mask

        def _optimize(self):
            captured["optimized"] = True

    problem_module = types.ModuleType(
        "forD.classical_fitting.Problem_regularized_standalone"
    )
    problem_module.Problem = FakeProblem
    config_module = types.ModuleType("forD.classical_fitting.config")
    config_module.Configuration = FakeConfiguration
    monkeypatch.setitem(
        sys.modules,
        "forD.classical_fitting.Problem_regularized_standalone",
        problem_module,
    )
    monkeypatch.setitem(sys.modules, "forD.classical_fitting.config", config_module)

    ford_pipeline.run_walinet_ford_pipeline(
        data_path=data_path,
        mask_path=mask_path,
        walinet_model_dir=model_dir,
        ford_config_template=config_path,
        output_path=output_dir,
        gpu_number=2,
        fid_axis=1,
    )

    expected = np.moveaxis(cleaned, 1, -1)
    np.testing.assert_array_equal(captured["data"], expected.astype(np.complex64))
    np.testing.assert_array_equal(captured["mask"], mask)
    assert captured["optimized"] is True
    assert captured["config_dict"]["pytorch_config"]["device"] == "cuda:2"
    assert captured["config_dict"]["io_config"]["saving_path"] == str(output_dir)
    assert (
        captured["config_dict"]["preprocessor_config"]["normalize_signals"]
        == "frequency"
    )
    assert (
        captured["config_dict"]["preprocessor_config"]
        ["normalization_scaling_mode"] == "z-score"
    )
    assert (output_dir / "fitting_config_used.json").is_file()
    assert not (output_dir / "input_normalization.json").exists()


def test_pipeline_rejects_mismatching_mask_before_inference(tmp_path):
    data_path = tmp_path / "data.npy"
    mask_path = tmp_path / "mask.npy"
    model_dir = tmp_path / "model"
    config_path = tmp_path / "template.json"
    model_dir.mkdir()
    config_path.write_text("{}")
    np.save(data_path, np.ones((2, 3, 8), dtype=np.complex64))
    np.save(mask_path, np.ones((2, 4), dtype=bool))

    try:
        ford_pipeline.run_walinet_ford_pipeline(
            data_path,
            mask_path,
            model_dir,
            config_path,
            tmp_path / "output",
            0,
            fid_axis=-1,
        )
    except ValueError as error:
        assert "does not match mask shape" in str(error)
    else:
        raise AssertionError("Expected a mask shape error.")
