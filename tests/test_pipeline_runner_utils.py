from __future__ import annotations

import json
from pathlib import Path

import scripts.pipeline_runner_utils as runner_utils


def test_shell_display_preserves_simple_values_and_quotes_spaces() -> None:
    assert runner_utils.shell_display("--output=report/run-1") == "--output=report/run-1"
    assert runner_utils.shell_display("two words") == repr("two words")
    assert runner_utils.shell_display("") == repr("")


def test_run_command_dry_run_only_prints_command(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    def fail_if_called(*args, **kwargs) -> None:
        raise AssertionError("dry-run must not start a subprocess")

    monkeypatch.setattr(runner_utils.subprocess, "run", fail_if_called)

    runner_utils.run_command(
        ["python", "script with spaces.py"],
        cwd=tmp_path,
        dry_run=True,
        stage="test stage",
    )

    assert capsys.readouterr().out == "\n== test stage ==\npython 'script with spaces.py'\n"


def test_training_complete_requires_model_summary_and_final_evaluation(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "train_summary.json").write_text(
        json.dumps({"step": 9}), encoding="utf-8"
    )
    (run_dir / "final_gaussians.pt").touch()
    (run_dir / "eval_history.csv").write_text(
        "step,psnr\n0,1.0\n10.0,2.0\n", encoding="utf-8"
    )

    assert runner_utils.training_complete(run_dir, max_steps=10)

    (run_dir / "eval_history.csv").write_text(
        "step,psnr\n0,1.0\n9,2.0\n", encoding="utf-8"
    )
    assert not runner_utils.training_complete(run_dir, max_steps=10)


def test_training_complete_rejects_invalid_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "train_summary.json").write_text("not-json", encoding="utf-8")
    (run_dir / "final_gaussians.pt").touch()
    (run_dir / "eval_history.csv").write_text("step\n10\n", encoding="utf-8")

    assert not runner_utils.training_complete(run_dir, 10)


def test_last_csv_step_and_comparison_steps(tmp_path: Path) -> None:
    history = tmp_path / "history.csv"
    history.write_text("step,value\n0,1\n3.0,2\n", encoding="utf-8")

    assert runner_utils.last_csv_step(history) == 3
    assert runner_utils.comparison_steps(2_000) == [0, 1_000, 2_000]
    assert runner_utils.comparison_steps(10_000) == [0, 1_000, 3_000, 8_000, 10_000]


def test_runner_modules_reexport_shared_helpers() -> None:
    from scripts import run_colmap_pipeline, run_dense_gif_comparison

    for module in (run_colmap_pipeline, run_dense_gif_comparison):
        assert module.run_command is runner_utils.run_command
        assert module.shell_display is runner_utils.shell_display
        assert module.training_complete is runner_utils.training_complete
        assert module.last_csv_step is runner_utils.last_csv_step
        assert module.comparison_steps is runner_utils.comparison_steps
