"""CPU-only contracts for the public overlay and evaluation runner.

Run with: python -m unittest discover -s tests -v
"""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_eval.sh"
BUILDER = ROOT / "llava" / "model" / "multimodal_encoder" / "builder.py"


class VisionTowerBuilderTests(unittest.TestCase):
    def load_builder(self):
        package_name = "release_test_encoder"
        package = types.ModuleType(package_name)
        package.__path__ = []
        encoder = types.ModuleType(package_name + ".clip_encoder")

        class StubCLIPVisionTower:
            def __init__(self, name, args, **kwargs):
                self.name = name
                self.args = args
                self.kwargs = kwargs

        # Deliberately provide only the class shipped by this overlay.
        encoder.CLIPVisionTower = StubCLIPVisionTower
        spec = importlib.util.spec_from_file_location(package_name + ".builder", BUILDER)
        builder = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {
            package_name: package,
            encoder.__name__: encoder,
        }):
            spec.loader.exec_module(builder)
        return builder

    def test_import_and_standard_clip_construction(self):
        builder = self.load_builder()
        config = types.SimpleNamespace(mm_vision_tower="openai/clip-vit-large-patch14-336")
        tower = builder.build_vision_tower(config, delay_load=True)
        self.assertEqual(tower.name, config.mm_vision_tower)
        self.assertIs(tower.args, config)
        self.assertEqual(tower.kwargs, {"delay_load": True})

    def test_s2_is_explicitly_rejected(self):
        builder = self.load_builder()
        config = types.SimpleNamespace(mm_vision_tower="openai/clip-vit-large-patch14-336", s2=True)
        with self.assertRaisesRegex(ValueError, "CLIP|[Ss]tandard|[Ss]2"):
            builder.build_vision_tower(config)


MOCK_PYTHON = textwrap.dedent("""\
    import argparse
    import json
    import os
    from pathlib import Path
    import sys

    if sys.argv[1:3] != ['-u', '-m']:
        os.execv(os.environ['RELEASE_TEST_PYTHON'],
                 [os.environ['RELEASE_TEST_PYTHON'], *sys.argv[1:]])

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path')
    parser.add_argument('--model-base')
    parser.add_argument('--question-file')
    parser.add_argument('--image-folder')
    parser.add_argument('--answers-file')
    parser.add_argument('--conv-mode')
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--pruning_method')
    parser.add_argument('--visual_token_num', type=int)
    parser.add_argument('--num-chunks', type=int, default=1)
    parser.add_argument('--chunk-idx', type=int, default=0)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    args = parser.parse_args(sys.argv[4:])
    names = [
        'STAGE1_SCORER', 'STAGE1_MULT', 'TEXT_AGG_MODE',
        'STAGE2_SELECTOR', 'PRUNING_SCHEDULE_MODE', 'CUSTOM_PRUNING_SCHEDULE',
        'STAGE2_ANCHOR_M', 'STAR_CAUSAL_FIX', 'STAR_KEEP_POSIDS',
    ]
    Path(os.environ['RELEASE_TEST_CAPTURE']).write_text(json.dumps({
        'argv': sys.argv[1:],
        'parsed': vars(args),
        'env': {name: os.environ.get(name) for name in names},
    }), encoding='utf-8')
    output = Path(args.answers_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'question_id': 'fixture', 'text': 'test'}) + '\\n',
                      encoding='utf-8')
""")


class EvaluationRunnerTests(unittest.TestCase):
    def run_runner(self, args=(), env_overrides=None, existing_output=None,
                   question_text='{"question_id": "fixture", "text": "What is shown?", "image": "fixture.png"}\n',
                   question_name="questions.jsonl"):
        with tempfile.TemporaryDirectory(prefix="release-entrypoints-") as temp:
            scratch = Path(temp)
            bin_dir = scratch / "bin"
            bin_dir.mkdir()
            python = bin_dir / "python"
            python.write_text("#!/usr/bin/env python3\n" + MOCK_PYTHON, encoding="utf-8")
            python.chmod(0o755)
            output = scratch / "answers.jsonl"
            capture = scratch / "capture.json"
            questions = scratch / question_name
            questions.write_text(question_text, encoding="utf-8")
            images = scratch / "images"
            images.mkdir()
            if existing_output is not None:
                output.write_text(existing_output, encoding="utf-8")
            env = os.environ.copy()
            env.update({
                "PATH": str(bin_dir) + os.pathsep + os.environ.get("PATH", ""),
                "LLAVA_ROOT": str(scratch / "llava"),
                "MODEL_PATH": str(scratch / "llava-model"),
                "QUESTION_FILE": str(questions),
                "IMAGE_FOLDER": str(images),
                "OUTPUT_FILE": str(output),
                "ENTRYPOINT": "model_vqa_loader",
                "METHOD": "star_pro",
                "T": "64",
                "CONV_MODE": "llava_v1",
                "RELEASE_TEST_CAPTURE": str(capture),
                "RELEASE_TEST_PYTHON": sys.executable,
            })
            if env_overrides:
                for name, value in env_overrides.items():
                    if value is None:
                        env.pop(name, None)
                    else:
                        env[name] = value
            # Running inside the temporary directory contains even a regressed
            # output-path override within disposable test fixtures.
            completed = subprocess.run(
                ["bash", str(RUNNER), *args],
                cwd=scratch, env=env, text=True, capture_output=True, timeout=15,
            )
            return types.SimpleNamespace(
                process=completed,
                capture=json.loads(capture.read_text()) if capture.exists() else None,
                output=output.read_text() if output.exists() else None,
                expected={
                    "model_path": env["MODEL_PATH"], "model_base": None,
                    "question_file": env["QUESTION_FILE"], "image_folder": env.get("IMAGE_FOLDER"),
                    "answers_file": str(output), "conv_mode": "llava_v1",
                    "temperature": 0.0, "pruning_method": "star_pro", "visual_token_num": 64,
                },
            )

    def assert_rejected_before_python(self, result):
        self.assertNotEqual(result.process.returncode, 0, result.process.stdout)
        self.assertIsNone(result.capture, "Rejected input reached the evaluation process")
        self.assertIsNone(result.output)

    def test_missing_or_nonregular_questions_fail_before_launch(self):
        for entrypoint in ("model_vqa", "model_vqa_loader", "model_vqa_science", "model_vqa_mmbench"):
            for question_file in ("missing.jsonl", "images"):
                with self.subTest(entrypoint=entrypoint, question_file=question_file):
                    result = self.run_runner(env_overrides={
                        "ENTRYPOINT": entrypoint, "QUESTION_FILE": question_file,
                    })
                    self.assert_rejected_before_python(result)
                    self.assertIn("QUESTION_FILE", result.process.stderr)

    def test_empty_questions_fail_before_launch(self):
        for entrypoint in ("model_vqa_loader", "model_vqa_mmbench"):
            with self.subTest(entrypoint=entrypoint):
                result = self.run_runner(question_text="", env_overrides={"ENTRYPOINT": entrypoint})
                self.assert_rejected_before_python(result)
                self.assertIn("QUESTION_FILE", result.process.stderr)

    def test_image_file_entrypoints_require_an_existing_image_directory(self):
        for entrypoint in ("model_vqa", "model_vqa_loader", "model_vqa_science"):
            for image_folder in (None, "missing-images", "questions.jsonl"):
                with self.subTest(entrypoint=entrypoint, image_folder=image_folder):
                    result = self.run_runner(env_overrides={
                        "ENTRYPOINT": entrypoint, "IMAGE_FOLDER": image_folder,
                    })
                    self.assert_rejected_before_python(result)
                    self.assertIn("IMAGE_FOLDER", result.process.stderr)

    def test_mmbench_uses_tsv_without_an_image_directory(self):
        for image_folder, expected in ((None, "."), ("", "."), ("missing-images", "missing-images")):
            with self.subTest(image_folder=image_folder):
                result = self.run_runner(
                    env_overrides={"ENTRYPOINT": "model_vqa_mmbench", "IMAGE_FOLDER": image_folder},
                    question_name="questions.tsv",
                    question_text="index\timage\tquestion\n1\tfixture-base64\tWhat is shown?\n",
                )
                self.assertEqual(result.process.returncode, 0, result.process.stderr)
                self.assertEqual(result.capture["argv"][:3], ["-u", "-m", "llava.eval.model_vqa_mmbench"])
                self.assertEqual(result.capture["parsed"]["question_file"], result.expected["question_file"])
                self.assertEqual(result.capture["parsed"]["image_folder"], expected)
                self.assertIn("artifact_ok rows=1", result.process.stdout)

    def test_unsupported_entrypoint_and_budget_are_rejected(self):
        for overrides in ({"ENTRYPOINT": "model_vqa_loader_accelerate"}, {"T": "192"}):
            with self.subTest(overrides=overrides):
                self.assert_rejected_before_python(self.run_runner(env_overrides=overrides))

    def test_reserved_parameters_cannot_be_overridden(self):
        options = [
            "--model-path", "--model-base", "--question-file", "--image-folder",
            "--answers-file", "--conv-mode", "--temperature", "--pruning_method",
            "--visual_token_num",
        ]
        for option in options:
            for args in ((option, "1"), (option + "=1",)):
                with self.subTest(args=args):
                    self.assert_rejected_before_python(self.run_runner(args))

    def test_abbreviated_parameters_cannot_change_fixed_configuration(self):
        # These are accepted as long-option abbreviations by the real argparse
        # entrypoints unless the shell runner rejects or safely overrides them.
        options = [
            "--model-p", "--model-b", "--question-f", "--image-f", "--answers-f",
            "--conv-m", "--temper", "--pruning_m", "--visual_token_n",
        ]
        for option in options:
            for args in ((option, "1"), (option + "=1",)):
                with self.subTest(args=args):
                    result = self.run_runner(args)
                    if result.process.returncode != 0:
                        self.assertIsNone(result.capture)
                        self.assertIsNone(result.output)
                    else:
                        self.assertIsNotNone(result.capture)
                        for key, expected in result.expected.items():
                            self.assertEqual(result.capture["parsed"][key], expected, key)

    def test_paper_settings_and_allowed_options_reach_evaluation(self):
        result = self.run_runner(
            ("--num-chunks", "2", "--chunk-idx", "1", "--max_new_tokens", "8"),
            env_overrides={
                "STAGE1_SCORER": "random", "STAGE1_MULT": "4",
                "TEXT_AGG_MODE": "last", "STAGE2_SELECTOR": "wqr",
                "PRUNING_SCHEDULE_MODE": "uniform", "CUSTOM_PRUNING_SCHEDULE": "[]",
                "STAGE2_ANCHOR_M": "8", "STAR_CAUSAL_FIX": "1", "STAR_KEEP_POSIDS": "1",
            },
        )
        self.assertEqual(result.process.returncode, 0, result.process.stderr)
        self.assertIsNotNone(result.capture)
        self.assertEqual(result.capture["argv"][:3], ["-u", "-m", "llava.eval.model_vqa_loader"])
        for key, expected in result.expected.items():
            self.assertEqual(result.capture["parsed"][key], expected, key)
        self.assertEqual(result.capture["parsed"]["num_chunks"], 2)
        self.assertEqual(result.capture["parsed"]["chunk_idx"], 1)
        self.assertEqual(result.capture["parsed"]["max_new_tokens"], 8)
        self.assertEqual(result.capture["env"], {
            "STAGE1_SCORER": "qr", "STAGE1_MULT": "2", "TEXT_AGG_MODE": "average_all",
            "STAGE2_SELECTOR": "topk", "PRUNING_SCHEDULE_MODE": "progressive",
            "CUSTOM_PRUNING_SCHEDULE": None, "STAGE2_ANCHOR_M": "0",
            "STAR_CAUSAL_FIX": "0", "STAR_KEEP_POSIDS": "0",
        })
        self.assertEqual(json.loads(result.output), {"question_id": "fixture", "text": "test"})
        self.assertIn("artifact_ok rows=1", result.process.stdout)

    def test_each_public_method_reaches_the_correct_evaluation_path(self):
        methods = {
            "star_pro": "star_pro", "starpro": "star_pro", "vanilla": "vanilla",
            "divprune": "divprune", "DivPrune": "divprune",
            "cdpruner": "cdp3", "CDPruner": "cdp3", "cdp3": "cdp3",
            "fastv": "fastv", "FastV": "fastv",
            "sparsevlm": "sparsevlm", "SparseVLM": "sparsevlm",
        }
        for public, internal in methods.items():
            with self.subTest(method=public):
                result = self.run_runner(env_overrides={"METHOD": public})
                self.assertEqual(result.process.returncode, 0, result.process.stderr)
                self.assertEqual(result.capture["parsed"]["pruning_method"], internal)
                self.assertEqual(result.capture["parsed"]["visual_token_num"], 64)
                self.assertIn("method=" + internal, result.process.stdout)

    def test_next_budget_reaches_model_without_double_conversion(self):
        # The model knows the checkpoint's actual crop configuration. The shell
        # must keep total T intact, including for pre-decoder baseline methods.
        for method in ("star_pro", "divprune", "cdpruner", "fastv", "sparsevlm"):
            with self.subTest(method=method):
                result = self.run_runner(env_overrides={"METHOD": method, "T": "320"})
                self.assertEqual(result.process.returncode, 0, result.process.stderr)
                self.assertEqual(result.capture["parsed"]["visual_token_num"], 320)

    def test_unsupported_methods_and_unavailable_schedules_fail_before_launch(self):
        cases = [{"METHOD": "not_a_method"}, {"METHOD": "vscan"}]
        cases += [{"METHOD": method, "T": budget}
                  for method in ("fastv", "sparsevlm") for budget in ("32", "160")]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                self.assert_rejected_before_python(self.run_runner(env_overrides=overrides))

    def test_existing_output_is_preserved(self):
        result = self.run_runner(existing_output="preserve this fixture\n")
        self.assertNotEqual(result.process.returncode, 0)
        self.assertIsNone(result.capture)
        self.assertEqual(result.output, "preserve this fixture\n")


if __name__ == "__main__":
    unittest.main()
