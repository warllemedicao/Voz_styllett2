#!/usr/bin/env python3
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".m4a", ".flac"}


def count_files(path: Path, audio_only: bool = False) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        if audio_only and item.suffix.lower() not in AUDIO_EXTS:
            continue
        total += 1
    return total


def command_output(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"ERRO: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnostico do ambiente Colab do super_Voz.")
    parser.add_argument("--config", default="styletts2_colab_config.yml")
    parser.add_argument("--output", default="logs/diagnostico_colab.json")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[1]
    config_path = project_dir / args.config
    output_path = project_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "project_dir": str(project_dir),
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "platform": platform.platform(),
        "config_path": str(config_path),
        "config_exists": config_path.exists(),
        "files": {
            "run_colab_styletts2.py": (project_dir / "scripts" / "run_colab_styletts2.py").exists(),
            "prepare_styletts2_dataset.py": (project_dir / "scripts" / "prepare_styletts2_dataset.py").exists(),
            "limpeza_ia.py": (project_dir / "limpeza_ia.py").exists(),
        },
        "commands": {
            "git_version": command_output(["git", "--version"]),
            "ffmpeg_version": command_output(["ffmpeg", "-version"]),
        },
        "drive": {},
    }

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        report["config"] = cfg

        raw_items = []
        for item in cfg.get("raw_audio_candidates", []):
            path = Path(item)
            raw_items.append({
                "path": str(path),
                "exists": path.exists(),
                "audio_files": count_files(path, audio_only=True),
                "all_files": count_files(path),
            })

        processed_items = []
        for item in cfg.get("processed_audio_candidates", []):
            path = Path(item)
            processed_items.append({
                "path": str(path),
                "exists": path.exists(),
                "train_txt": (path / "train.txt").exists(),
                "metadata_csv": (path / "metadata.csv").exists(),
                "wav_files": len(list(path.rglob("*.wav"))) if path.exists() else 0,
                "all_files": count_files(path),
            })

        report["drive"]["raw_audio_candidates"] = raw_items
        report["drive"]["processed_audio_candidates"] = processed_items
        drive_project = Path(cfg.get("drive_project_dir", ""))
        report["drive"]["project_dir"] = {
            "path": str(drive_project),
            "exists": drive_project.exists(),
        }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nDiagnostico salvo em: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
