#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".m4a", ".flac"}


def run(cmd, cwd=None, check=True):
    print("\n$ " + " ".join(map(str, cmd)))
    if cwd:
        print("cwd:", cwd)
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check)


def clone_or_pull(url: str, dest: Path) -> None:
    if dest.exists():
        run(["git", "-C", str(dest), "pull", "--ff-only"], check=False)
    else:
        run(["git", "clone", url, str(dest)])


def find_path_case_insensitive(path_str: str) -> Path | None:
    """Resolve um caminho de forma insensível a maiúsculas/minúsculas."""
    path = Path(path_str)
    if path.exists():
        return path
    
    parts = path.parts
    if not parts:
        return None
        
    # No Linux (Colab), caminhos absolutos começam com /
    if path_str.startswith('/'):
        current = Path('/')
        parts = parts[1:]
    else:
        current = Path('.')
        
    for part in parts:
        next_path = current / part
        if next_path.exists():
            current = next_path
        else:
            # Busca insensível a maiúsculas no diretório atual
            found = False
            if current.exists() and current.is_dir():
                try:
                    for item in current.iterdir():
                        if item.name.lower() == part.lower():
                            current = item
                            found = True
                            break
                except (PermissionError, OSError):
                    return None
            if not found:
                return None
    return current if current.exists() else None


def first_existing(paths: list[str]) -> Path | None:
    for item in paths:
        path = find_path_case_insensitive(item)
        if path:
            return path
    return None


def copy_tree_files(src_dir: Path, dst_dir: Path, allowed=None) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in sorted(src_dir.rglob("*")):
        if not src.is_file():
            continue
        if allowed and not allowed(src):
            continue
        dst = dst_dir / src.relative_to(src_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return copied


def count_files(path: Path, allowed=None) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file() and (allowed is None or allowed(item)):
            total += 1
    return total


def install_dependencies(style_dir: Path) -> None:
    # Verificar pacotes de sistema
    missing_sys = []
    for pkg in ["ffmpeg", "sox", "espeak-ng"]:
        if shutil.which(pkg) is None:
            missing_sys.append(pkg)
    
    if missing_sys:
        print(f"[INFO] Instalando pacotes de sistema: {missing_sys}")
        run(["apt-get", "update"])
        run(["apt-get", "install", "-y", "ffmpeg", "sox", "libsndfile1", "espeak-ng"])

    # Pip install é rápido se já estiver instalado, mas vamos manter o -q
    print("[INFO] Verificando/Instalando dependências Python...")
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "torch",
        "torchaudio",
        "torchvision",
        "accelerate",
        "huggingface_hub",
        "pyyaml",
        "librosa",
        "soundfile",
        "phonemizer",
        "openai-whisper",
        "demucs",
    ])
    requirements = style_dir / "requirements.txt"
    if requirements.exists():
        run([sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)], check=False)


def download_pretrained(style_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    filename = "Models/LibriTTS/epochs_2nd_00020.pth"
    path = hf_hub_download(
        repo_id="yl4579/StyleTTS2-LibriTTS",
        filename=filename,
        local_dir=str(style_dir),
        local_dir_use_symlinks=False,
    )
    return Path(path)


def patch_styletts2_config(style_dir: Path, dataset_dir: Path, cfg: dict) -> Path:
    import yaml

    config_path = style_dir / "Configs" / "config_ft.yml"
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["log_dir"] = "Models/super_Voz"
    config["epochs"] = int(cfg.get("epochs", 50))
    config["batch_size"] = int(cfg.get("batch_size", 8))
    config["max_len"] = int(cfg.get("max_len", 400))
    config["save_freq"] = int(cfg.get("save_freq", 5))
    config["log_interval"] = int(cfg.get("log_interval", 10))
    config["device"] = "cuda"
    config["pretrained_model"] = "Models/LibriTTS/epochs_2nd_00020.pth"
    config["load_only_params"] = True

    data_params = config.setdefault("data_params", {})
    data_params["train_data"] = "Data/train_list.txt"
    data_params["val_data"] = "Data/val_list.txt"
    data_params["root_path"] = str(dataset_dir / "wavs")
    data_params["OOD_data"] = "Data/OOD_texts.txt"

    loss_params = config.setdefault("loss_params", {})
    loss_params["diff_epoch"] = int(cfg.get("diff_epoch", 10))
    loss_params["joint_epoch"] = int(cfg.get("joint_epoch", 999))

    out_path = style_dir / "Configs" / "config_super_voz.yml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    return out_path


def prepare_local_audio(project_dir: Path, cfg: dict) -> tuple[Path, bool]:
    raw_candidates = cfg.get("raw_audio_candidates", [])
    processed_candidates = cfg.get("processed_audio_candidates", [])
    
    # Busca inteligente (case-insensitive)
    raw_drive = first_existing(raw_candidates)
    processed_drive = first_existing(processed_candidates)
    
    local_raw = project_dir / "Audios_brutos"
    local_processed = project_dir / "Audios_processados"
    local_raw.mkdir(parents=True, exist_ok=True)
    local_processed.mkdir(parents=True, exist_ok=True)

    print("\n[DRIVE] Candidatos para audios brutos:")
    for item in raw_candidates:
        path = find_path_case_insensitive(item)
        if path:
            total = count_files(path, allowed=lambda p: p.suffix.lower() in AUDIO_EXTS)
            print(f"- {item} -> {path} | existe=True | audios={total}")
        else:
            print(f"- {item} | existe=False")

    print("\n[DRIVE] Candidatos para audios processados:")
    for item in processed_candidates:
        path = find_path_case_insensitive(item)
        if path:
            total = count_files(path, allowed=lambda p: p.suffix.lower() == ".wav" or p.name in {"train.txt", "metadata.csv"})
            print(f"- {item} -> {path} | existe=True | arquivos_validos={total} | train.txt={(path / 'train.txt').exists()}")
        else:
            print(f"- {item} | existe=False")

    imported_processed = False
    if processed_drive:
        copied = copy_tree_files(
            processed_drive,
            local_processed,
            allowed=lambda p: p.suffix.lower() == ".wav" or p.name in {"train.txt", "metadata.csv"},
        )
        imported_processed = copied > 0 and (local_processed / "train.txt").exists()
        print(f"Processados importados do Drive: {copied}")

    if raw_drive:
        copied = copy_tree_files(raw_drive, local_raw, allowed=lambda p: p.suffix.lower() in AUDIO_EXTS)
        print(f"Audios brutos importados do Drive: {copied}")

    if not imported_processed:
        if not any(local_raw.rglob("*")):
            tried = "\n".join(f"- {item}" for item in [*processed_candidates, *raw_candidates])
            drive_mounted = Path("/content/drive/MyDrive").exists()
            error_msg = (
                "\n" + "!"*60 + "\n"
                " [ERRO] NENHUM ÁUDIO ENCONTRADO NO DRIVE\n"
                "!"*60 + "\n"
                f"Status do Drive: {'MONTADO' if drive_mounted else 'NÃO MONTADO'}\n\n"
                "Caminhos verificados:\n"
                f"{tried}\n\n"
                "DICA: Verifique se você montou o Drive e se a pasta 'super_Voz' existe na raiz do seu Google Drive.\n"
                "DICA: O script procura por Audios_brutos ou Audios_processados.\n"
                "!"*60
            )
            raise FileNotFoundError(error_msg)
        run([
            sys.executable,
            "limpeza_ia.py",
            "--input_dir",
            str(local_raw),
            "--output_dir",
            str(local_processed),
            "--ambiente",
            "colab",
        ], cwd=project_dir)

    train_txt = local_processed / "train.txt"
    if not train_txt.exists() or not train_txt.read_text(encoding="utf-8").strip():
        raise RuntimeError(f"train.txt nao encontrado ou vazio: {train_txt}")
    return local_processed, imported_processed


def sync_outputs(style_dir: Path, dataset_dir: Path, cfg: dict) -> None:
    drive_path_str = cfg.get("drive_project_dir", "/content/drive/MyDrive/super_Voz")
    drive_project = find_path_case_insensitive(drive_path_str)
    
    if not drive_project:
        # Se não existir, tenta criar na raiz do Drive
        drive_project = Path(drive_path_str)
        print(f"[AVISO] Pasta do projeto não encontrada no Drive. Tentando criar em: {drive_project}")

    checkpoint_dst = drive_project / "checkpoints"
    dataset_dst = drive_project / "styletts2_data"
    checkpoint_src = style_dir / "Models" / "super_Voz"

    if checkpoint_src.exists():
        copy_tree_files(checkpoint_src, checkpoint_dst)
        print(f"Checkpoints copiados para: {checkpoint_dst}")

    if dataset_dir.exists():
        copy_tree_files(dataset_dir / "Data", dataset_dst / "Data")
        report = dataset_dir / "prepare_report.txt"
        if report.exists():
            dataset_dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(report, dataset_dst / report.name)
        print(f"Listas do dataset copiadas para: {dataset_dst}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline Colab super_Voz com StyleTTS2.")
    parser.add_argument("--config", default="styletts2_colab_config.yml")
    parser.add_argument("--skip_train", action="store_true")
    args = parser.parse_args()

    import yaml

    project_dir = Path(__file__).resolve().parents[1]
    with (project_dir / args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    style_dir = Path(cfg.get("styletts2_dir", "/content/StyleTTS2"))
    clone_or_pull(cfg.get("styletts2_repo", "https://github.com/yl4579/StyleTTS2.git"), style_dir)
    install_dependencies(style_dir)

    processed_dir, imported_processed = prepare_local_audio(project_dir, cfg)
    print(f"Dataset processado: {processed_dir}")
    print(f"Limpeza/transcricao pulada: {imported_processed}")

    dataset_dir = Path("/content/super_Voz_styletts2_data")
    prepare_cmd = [
        sys.executable,
        str(project_dir / "scripts" / "prepare_styletts2_dataset.py"),
        "--input_dir",
        str(processed_dir),
        "--output_dir",
        str(dataset_dir),
        "--speaker",
        str(cfg.get("speaker", "0")),
        "--sample_rate",
        str(cfg.get("sample_rate", 24000)),
        "--val_ratio",
        str(cfg.get("val_ratio", 0.05)),
    ]
    if cfg.get("phonemize", True):
        prepare_cmd.extend(["--phonemize", "--phonemizer_language", str(cfg.get("phonemizer_language", "pt-br"))])
    run(prepare_cmd, cwd=project_dir)

    (style_dir / "Data").mkdir(parents=True, exist_ok=True)
    for name in ["train_list.txt", "val_list.txt", "OOD_texts.txt"]:
        shutil.copy2(dataset_dir / "Data" / name, style_dir / "Data" / name)

    pretrained = download_pretrained(style_dir)
    print(f"Checkpoint base: {pretrained}")
    config_path = patch_styletts2_config(style_dir, dataset_dir, cfg)
    print(f"Config StyleTTS2: {config_path}")

    if not args.skip_train:
        train_script = style_dir / "train_finetune_accelerate.py"
        if not train_script.exists():
            raise FileNotFoundError(f"Script de fine-tuning nao encontrado: {train_script}")
        run([
            "accelerate",
            "launch",
            "--mixed_precision=fp16",
            "--num_processes=1",
            str(train_script.name),
            "--config_path",
            str(config_path),
        ], cwd=style_dir)

    sync_outputs(style_dir, dataset_dir, cfg)
    print("Pipeline super_Voz finalizado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
