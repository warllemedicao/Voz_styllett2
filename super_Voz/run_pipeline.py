#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ============================================================
# CONFIGURAÇÃO E DETECÇÃO DE AMBIENTE
# ============================================================

AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".m4a", ".flac"}

def run(cmd, cwd=None, check=True):
    print("\n$ " + " ".join(map(str, cmd)))
    if cwd:
        print("cwd:", cwd)
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check)

def detect_environment():
    if Path("/kaggle/working").exists():
        return "kaggle"
    try:
        import google.colab
        return "colab"
    except ImportError:
        return "local"

def find_path_case_insensitive(path_str: str) -> Path | None:
    path = Path(path_str)
    if path.exists():
        return path
    parts = path.parts
    if not parts: return None
    current = Path("/") if path_str.startswith("/") else Path(".")
    if path_str.startswith("/") : parts = parts[1:]
    
    for part in parts:
        found = False
        if current.exists() and current.is_dir():
            try:
                for item in current.iterdir():
                    if item.name.lower() == part.lower():
                        current = item
                        found = True
                        break
            except: return None
        if not found: return None
    return current if current.exists() else None

def first_existing(paths: list[str]) -> Path | None:
    for item in paths:
        path = find_path_case_insensitive(item)
        if path: return path
    return None

def copy_tree_files(src_dir: Path, dst_dir: Path, allowed=None) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in sorted(src_dir.rglob("*")):
        if not src.is_file(): continue
        if allowed and not allowed(src): continue
        dst = dst_dir / src.relative_to(src_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return copied

def count_files(path: Path, allowed=None) -> int:
    if not path.exists(): return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file() and (allowed is None or allowed(item)):
            total += 1
    return total

# ============================================================
# INSTALAÇÃO E PATCHES
# ============================================================

def install_dependencies(style_dir: Path, env: str):
    print("\n--- Instalando Dependências ---")
    if env in ["colab", "kaggle"]:
        run(["apt-get", "update"], check=False)
        run(["apt-get", "install", "-y", "ffmpeg", "sox", "libsndfile1", "espeak-ng"], check=False)

    pkgs = ["torch", "torchaudio", "torchvision", "accelerate", "huggingface_hub", "pyyaml", "librosa", "soundfile", "phonemizer", "openai-whisper", "demucs", "boto3"]
    run([sys.executable, "-m", "pip", "install", "-q"] + pkgs)

    reqs = style_dir / "requirements.txt"
    if reqs.exists():
        run([sys.executable, "-m", "pip", "install", "-q", "-r", str(reqs)], check=False)

def patch_styletts2(style_dir: Path):
    """Aplica patches de compatibilidade e memória."""
    models_py = style_dir / "models.py"
    if models_py.exists():
        content = models_py.read_text(encoding="utf-8")
        new_content = content.replace("torch.load(model_path, map_location='cpu')", "torch.load(model_path, map_location='cpu', weights_only=False)")
        new_content = new_content.replace('torch.load(model_path, map_location="cpu")', 'torch.load(model_path, map_location="cpu", weights_only=False)')
        if content != new_content:
            models_py.write_text(new_content, encoding="utf-8")
            print("✅ Patch PyTorch 2.6 aplicado.")

    train_py = style_dir / "train_finetune_accelerate.py"
    if train_py.exists():
        content = train_py.read_text(encoding="utf-8")
        new_content = content.replace("mel_len_st = int(mel_input_length.min().item() / 2 - 1)", "mel_len_st = min(int(mel_input_length.min().item() / 2 - 1), max_len // 2)")
        if content != new_content:
            train_py.write_text(new_content, encoding="utf-8")
            print("✅ Patch Anti-OOM aplicado.")

# ============================================================
# DOWNLOAD E PREPARAÇÃO
# ============================================================
# DOWNLOAD E PREPARAÇÃO (R2)
# ============================================================

def get_r2_client(cfg: dict):
    import boto3
    from botocore.config import Config
    r2_cfg = cfg.get("cloudflare_r2", {})
    if not r2_cfg or "INSERIR" in r2_cfg.get("access_key_id", ""): return None, None
    s3 = boto3.client("s3", endpoint_url=r2_cfg.get("endpoint_url"),
                      aws_access_key_id=r2_cfg.get("access_key_id"),
                      aws_secret_access_key=r2_cfg.get("secret_access_key"),
                      config=Config(signature_version="s3v4"), region_name="auto")
    return s3, r2_cfg.get("bucket_name")

def download_from_r2(s3, bucket, prefix, local_dir: Path):
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[R2] Baixando prefixo '{prefix}' para {local_dir}...")
    paginator = s3.get_paginator("list_objects_v2")
    downloaded = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" in page:
            for obj in page["Contents"]:
                key = obj["Key"]
                if key.endswith("/"): continue
                rel_path = os.path.relpath(key, prefix)
                dst_path = local_dir / rel_path
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, key, str(dst_path))
                downloaded += 1
    return downloaded

def upload_to_r2(s3, bucket, prefix, local_dir: Path):
    print(f"[R2] Enviando {local_dir} para prefixo '{prefix}'...")
    uploaded = 0
    for root, _, files in os.walk(local_dir):
        for file in files:
            src = Path(root) / file
            key = os.path.join(prefix, os.path.relpath(src, local_dir)).replace("\\", "/")
            s3.upload_file(str(src), bucket, key)
            uploaded += 1
    return uploaded

def prepare_data(project_dir: Path, cfg: dict, env: str):
    local_raw = project_dir / "Audios_brutos"
    local_processed = project_dir / "Audios_processados"
    local_raw.mkdir(parents=True, exist_ok=True)
    local_processed.mkdir(parents=True, exist_ok=True)

    s3, bucket = get_r2_client(cfg)
    imported = False

    if s3:
        r2_cfg = cfg.get("cloudflare_r2", {})
        proc_prefix = r2_cfg.get("processed_audio_prefix")
        if proc_prefix:
            downloaded = download_from_r2(s3, bucket, proc_prefix, local_processed)
            imported = downloaded > 0 and (local_processed / "train.txt").exists()
            if imported: print(f"✅ Processados importados do R2: {downloaded}")
        
        if not imported:
            raw_prefix = r2_cfg.get("raw_audio_prefix")
            if raw_prefix:
                downloaded = download_from_r2(s3, bucket, raw_prefix, local_raw)
                print(f"✅ Brutos importados do R2: {downloaded}")
    else:
        print("[AVISO] R2 não configurado. Verificando candidatos locais...")
        raw_drive = first_existing(cfg.get("raw_audio_candidates", []))
        processed_drive = first_existing(cfg.get("processed_audio_candidates", []))
        if processed_drive:
            copied = copy_tree_files(processed_drive, local_processed, allowed=lambda p: p.suffix.lower() == ".wav" or p.name in {"train.txt", "metadata.csv"})
            imported = copied > 0 and (local_processed / "train.txt").exists()
            if imported: print(f"✅ Processados encontrados: {copied}")
        if raw_drive and not imported:
            copied = copy_tree_files(raw_drive, local_raw, allowed=lambda p: p.suffix.lower() in AUDIO_EXTS)
            if copied > 0: print(f"✅ Brutos encontrados: {copied}")

    if not imported and not any(local_raw.rglob("*")):
        print("❌ NENHUM ÁUDIO ENCONTRADO!")
        return None, False

    if not imported:
        run([sys.executable, "limpeza_ia.py", "--input_dir", str(local_raw), "--output_dir", str(local_processed), "--ambiente", env], cwd=project_dir)
    
    return local_processed, imported

def sync_outputs(style_dir: Path, dataset_dir: Path, cfg: dict):
    s3, bucket = get_r2_client(cfg)
    if not s3: return
    r2_cfg = cfg.get("cloudflare_r2", {})
    output_prefix = r2_cfg.get("output_prefix", "super_Voz_outputs/")
    
    # Upload checkpoints
    checkpoint_src = style_dir / "Models" / "super_Voz"
    if checkpoint_src.exists():
        upload_to_r2(s3, bucket, os.path.join(output_prefix, "checkpoints"), checkpoint_src)
    
    # Upload dataset Data
    if dataset_dir.exists():
        upload_to_r2(s3, bucket, os.path.join(output_prefix, "styletts2_data/Data"), dataset_dir / "Data")
        report = dataset_dir / "prepare_report.txt"
        if report.exists():
            s3.upload_file(str(report), bucket, os.path.join(output_prefix, "styletts2_data/prepare_report.txt"))
    print(f"✅ Resultados sincronizados com R2!")

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    env = detect_environment()
    print(f"--- Rodando em modo: {env.upper()} ---")

    import yaml
    project_dir = Path(__file__).resolve().parent
    with open(project_dir / args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    style_dir = Path(cfg.get("styletts2_dir", "/kaggle/working/StyleTTS2" if env == "kaggle" else "/content/StyleTTS2"))
    repo = cfg.get("styletts2_repo", "https://github.com/yl4579/StyleTTS2.git")
    if not style_dir.exists():
        run(["git", "clone", repo, str(style_dir)])
    else:
        run(["git", "-C", str(style_dir), "pull"], check=False)

    patch_styletts2(style_dir)
    install_dependencies(style_dir, env)
    
    proc_dir, imported = prepare_data(project_dir, cfg, env)
    if not proc_dir: return 1

    dataset_dir = project_dir / "styletts2_prepared_data"
    prep_cmd = [sys.executable, "scripts/prepare_styletts2_dataset.py", "--input_dir", str(proc_dir), "--output_dir", str(dataset_dir)]
    prep_cmd += ["--speaker", str(cfg.get("speaker", "0")), "--sample_rate", str(cfg.get("sample_rate", 24000))]
    if cfg.get("phonemize", True):
        prep_cmd += ["--phonemize", "--phonemizer_language", str(cfg.get("phonemizer_language", "pt-br"))]
    
    run(prep_cmd, cwd=project_dir)

    (style_dir / "Data").mkdir(parents=True, exist_ok=True)
    for f in ["train_list.txt", "val_list.txt", "OOD_texts.txt"]:
        shutil.copy2(dataset_dir / "Data" / f, style_dir / "Data" / f)

    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id="yl4579/StyleTTS2-LibriTTS", filename="Models/LibriTTS/epochs_2nd_00020.pth", local_dir=str(style_dir))

    import yaml as y2
    cft_p = style_dir / "Configs" / "config_ft.yml"
    with cft_p.open("r") as f: cft = y2.safe_load(f)
    cft.update({"log_dir": "Models/super_Voz", "epochs": cfg.get("epochs", 50), "batch_size": cfg.get("batch_size", 2), "device": "cuda", "pretrained_model": "Models/LibriTTS/epochs_2nd_00020.pth", "load_only_params": True})
    cft["data_params"].update({"train_data": "Data/train_list.txt", "val_data": "Data/val_list.txt", "root_path": str(dataset_dir / "wavs"), "OOD_data": "Data/OOD_texts.txt"})
    with (style_dir / "Configs" / "config_super_voz.yml").open("w") as f: y2.safe_dump(cft, f)

    run(["accelerate", "launch", "--mixed_precision=fp16", "--num_processes=1", "train_finetune_accelerate.py", "--config_path", "Configs/config_super_voz.yml"], cwd=style_dir)
    
    sync_outputs(style_dir, dataset_dir, cfg)
    return 0

if __name__ == "__main__":
    sys.exit(main())
