"""Allocator-aware compile selection and mixed-stream inference regression tests."""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uniwam.utils import inference_compile
from compiled_inference_fixture import inputs, small_policy


class InferenceCompileTest(unittest.TestCase):
    def test_expandable_allocator_uses_inductor_without_graph_capture(self):
        function = lambda value: value
        with patch.object(torch.cuda.memory, "_snapshot",
                          return_value={"allocator_settings": {"expandable_segments": True}}), \
             patch.object(torch, "compile", return_value=function) as compile:
            for vae in (False, True):
                self.assertIs(inference_compile.compile_inference(function, device="cuda", vae=vae), function)
                compile.assert_called_with(
                    function, backend="inductor", fullgraph=True,
                    options={"triton.cudagraphs": False},
                )

    def test_standard_allocator_keeps_original_compile_backends(self):
        function = lambda value: value
        with patch.object(torch.cuda.memory, "_snapshot",
                          return_value={"allocator_settings": {"expandable_segments": False}}), \
             patch.object(torch, "compile", return_value=function) as compile:
            inference_compile.compile_inference(function, device="cuda")
            compile.assert_called_with(function, mode="reduce-overhead", fullgraph=True)
            inference_compile.compile_inference(function, device="cuda", vae=True)
            compile.assert_called_with(function, backend="cudagraphs", fullgraph=True)

    def test_active_allocator_settings_take_priority_over_environment(self):
        with patch.dict(os.environ, {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}), \
             patch.object(torch.cuda.memory, "_snapshot",
                          return_value={"allocator_settings": {"expandable_segments": False}}):
            self.assertTrue(inference_compile._cuda_graphs_compatible("cuda"))

    def test_older_torch_environment_fallback(self):
        for variable in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
            for flag, compatible in (("True", False), ("False", True)):
                with self.subTest(variable=variable, flag=flag), \
                     patch.dict(os.environ, {variable: f"max_split_size_mb:128, expandable_segments:{flag}",
                                             next(x for x in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF") if x != variable): ""}), \
                     patch.object(torch.cuda.memory, "_snapshot", side_effect=AttributeError):
                    self.assertEqual(inference_compile._cuda_graphs_compatible("cuda"), compatible)

    def test_cpu_compile_never_inspects_cuda_or_uses_cuda_graphs(self):
        function = lambda value: value
        with patch.object(torch.cuda.memory, "_snapshot") as snapshot, \
             patch.object(torch, "compile", return_value=function) as compile:
            inference_compile.compile_inference(function, device="cpu", vae=True)
        snapshot.assert_not_called()
        compile.assert_called_once_with(
            function, backend="inductor", fullgraph=True,
            options={"triton.cudagraphs": False},
        )

    def test_all_mixed_stream_compile_entrypoints_use_allocator_compatible_selection(self):
        compiled = small_policy("cpu", compiled=True)
        eager = small_policy("cpu", compiled=False)
        kwargs_seen = []
        def record(function, **kwargs):
            kwargs_seen.append(kwargs)
            return function
        with patch.object(torch, "compile", side_effect=record), torch.inference_mode():
            for mode in ("paired", "manip_only", "nav_only", "paired"):
                output = compiled.infer_action(**inputs("cpu", mode))
                reference = eager.infer_action(**inputs("cpu", mode))
                for branch in ("manip_action", "nav_action"):
                    self.assertEqual(branch in output, branch in reference)
                    if branch in reference:
                        torch.testing.assert_close(output[branch], reference[branch])
        self.assertEqual(len(kwargs_seen), 3)
        for kwargs in kwargs_seen:
            self.assertEqual(kwargs, {
                "backend": "inductor", "fullgraph": True, "options": {"triton.cudagraphs": False},
            })


def cuda_probe():
    if not torch.cuda.is_available():
        raise RuntimeError("The CUDA probe requires an available GPU.")
    device = "cuda:0"
    compiled = small_policy(device, compiled=True)
    reference = small_policy(device, compiled=False)
    capture_enabled = inference_compile._cuda_graphs_compatible(device)
    checks = []
    with torch.inference_mode():
        for mode, prefix in (
            ("paired", 0), ("paired", 0), ("paired", 0),
            ("manip_only", 0), ("manip_only", 0), ("manip_only", 0),
            ("nav_only", 0), ("nav_only", 0), ("nav_only", 0),
            ("paired", 0), ("paired", 6), ("manip_only", 6),
            ("nav_only", 6), ("paired", 0),
        ):
            output = compiled.infer_action(**inputs(device, mode, prefix))
            expected = reference.infer_action(**inputs(device, mode, prefix))
            for branch in ("manip_action", "nav_action"):
                if branch in expected:
                    torch.testing.assert_close(output[branch], expected[branch], rtol=2e-4, atol=2e-4)
                    if prefix:
                        torch.testing.assert_close(output[branch][:prefix], torch.zeros_like(output[branch][:prefix]))
            checks.append({"mode": mode, "prefix": prefix})
            print(f"CUDA_PROBE_STEP_OK mode={mode} prefix={prefix}", flush=True)
    torch.cuda.synchronize()
    print(json.dumps({
        "torch": torch.__version__,
        "allocator": torch.cuda.memory._snapshot()["allocator_settings"],
        "cuda_graphs_enabled": capture_enabled,
        "checks": checks,
        "compiled_matches_eager": True,
    }))


if __name__ == "__main__":
    if "--cuda-probe" in sys.argv:
        cuda_probe()
    else:
        unittest.main()
