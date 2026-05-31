#!/usr/bin/env python3
# ============================================================
# limpeza_ia.py — LIMPEZA DE ÁUDIO COM ANÁLISE INTELIGENTE (V6)
# Solução Ultra-Robusta: GPU Warm-up e Device Synchronization
# ============================================================

import os
import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime
import sys
import numpy as np
import librosa
import torch
import torchaudio
import warnings
import soundfile as sf

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURAÇÃO
# ============================================================

CACHE_ANÁLISE = "analise_audio_cache.json"
PROCESSADOS_LOG = "processados.json"
DNSMOS_MODEL_URL = "https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx"

# ============================================================
# DNSMOS: NOTA DE QUALIDADE (MÉTRICA DA MICROSOFT)
# ============================================================

class DNSMOS:
    def __init__(self, model_path=None):
        self.model_path = model_path or "dnsmos_model.onnx"
        self.session = None
        self._check_model()

    def _check_model(self):
        if not Path(self.model_path).exists():
            print(f"[INFO] Baixando modelo DNSMOS...")
            try:
                import urllib.request
                urllib.request.urlretrieve(DNSMOS_MODEL_URL, self.model_path)
            except Exception as e:
                print(f"[AVISO] Falha ao baixar DNSMOS: {e}")

        try:
            import onnxruntime as ort
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' not in available_providers:
                providers = ['CPUExecutionProvider']
            self.session = ort.InferenceSession(self.model_path, providers=providers)
        except Exception as e:
            print(f"[ERRO CRÍTICO] Motor DNSMOS falhou: {e}")

    def score(self, audio: np.ndarray, sr: int) -> dict:
        if self.session is None: return {"ovrl": 0.4, "sig": 0.4, "bak": 0.4} 
        try:
            if sr != 16000: audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            target_len = 144160
            if len(audio) < target_len: audio = np.pad(audio, (0, target_len - len(audio)))
            else: audio = audio[:target_len]
            audio_input = audio.astype(np.float32)[np.newaxis, :]
            inputs = {self.session.get_inputs()[0].name: audio_input}
            outputs = self.session.run(None, inputs)
            return {
                "sig": (outputs[0][0][0] - 1) / 4,
                "bak": (outputs[0][0][1] - 1) / 4,
                "ovrl": (outputs[0][0][2] - 1) / 4
            }
        except Exception as e:
            return {"ovrl": 0.5, "sig": 0.5, "bak": 0.5}

# ============================================================
# AUDIO ENHANCER: RESEMBLE ENHANCE (SOLUÇÃO V6)
# ============================================================

class AudioEnhancer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.has_resemble = False
        self._warmup_done = False
        try:
            import resemble_enhance
            from resemble_enhance.enhancer.inference import enhance
            self.has_resemble = True
            print(f"[INFO] Motor de restauração detectado no device: {self.device}")
        except Exception as e:
            print(f"[AVISO] resemble-enhance indisponível: {e}")

    def _warmup(self):
        """Força o carregamento dos modelos na GPU com um tensor dummy."""
        if not self.has_resemble or self._warmup_done or str(self.device) == "cpu":
            return
        
        print("[INFO] Aquecendo motores da IA (GPU Warm-up)...")
        try:
            from resemble_enhance.enhancer.inference import enhance
            # Criar 1 segundo de silêncio a 44.1kHz
            dummy_wav = torch.zeros(44100).to(self.device).to(torch.float32)
            # Rodar uma vez para estabilizar o device interno da biblioteca
            _, _ = enhance(dummy_wav, 44100, device=self.device, nfe=1)
            self._warmup_done = True
            print("[INFO] GPU pronta e sincronizada.")
        except Exception as e:
            print(f"[AVISO] Falha no Warm-up: {e}")

    def process(self, input_path: Path, output_path: Path):
        if not self.has_resemble: return False
        
        # Executar warmup apenas uma vez
        if not self._warmup_done:
            self._warmup()

        try:
            from resemble_enhance.enhancer.inference import enhance
            
            # 1. Carregar áudio
            dwav, sr = torchaudio.load(str(input_path))
            
            # 2. Converter para mono e float32
            if dwav.shape[0] > 1: dwav = dwav.mean(dim=0, keepdim=True)
            dwav = dwav.to(self.device).to(torch.float32)
            
            # 3. Resampling Manual para 44.1kHz (Obrigatório para Resemble)
            if sr != 44100:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=44100).to(self.device)
                dwav = resampler(dwav)
                sr = 44100
            
            # 4. Garantir 1D para a API
            dwav_1d = dwav.squeeze()
            
            # 5. Inferência com Sincronização
            torch.cuda.empty_cache() # Limpar fragmentos de memória
            hwav, new_sr = enhance(dwav_1d, sr, device=self.device, nfe=32, solver="midpoint", lambd=0.5)
            
            # 6. Salvar resultado
            audio_out = hwav.cpu().numpy()
            sf.write(str(output_path), audio_out, new_sr)
            return True
        except Exception as e:
            print(f"  [ERRO ENHANCER] {e}")
            return False

# ============================================================
# CLASSE DE ANÁLISE DE ÁUDIO
# ============================================================

class AudioAnalyzer:
    def __init__(self, sr: int = 24000):
        self.sr = sr
        self.mos_tool = DNSMOS()

    def analyze(self, audio_path: str, verbose: bool = False) -> dict:
        try:
            audio, sr = librosa.load(audio_path, sr=self.sr, mono=True)
        except Exception as e:
            return {"status": "erro", "erro": str(e), "processamento_necessario": True, "score_geral": 0}

        audio = audio.astype(np.float32)
        mos_scores = self.mos_tool.score(audio, sr)
        
        # Heurísticas rápidas
        D = np.abs(librosa.stft(audio, n_fft=1024, hop_length=256))
        freq_bins = np.fft.rfftfreq(1024, 1/sr)
        hissing_idx = freq_bins > 8000
        razao_hissing = (np.mean(np.abs(D[hissing_idx, :])) / np.mean(np.abs(D))) if np.any(hissing_idx) else 0
        hissing_heu = min(razao_hissing * 5, 1.0)
        
        problemas = []
        if mos_scores['ovrl'] < 0.6: problemas.append(f"Voz degradada (Nota IA: {mos_scores['ovrl']*5:.1f}/5.0)")
        if hissing_heu > 0.5: problemas.append("Chiado agudo detectado")

        score_geral = mos_scores['ovrl'] * 0.8 + (1.0 - hissing_heu) * 0.2
        processamento_necessario = len(problemas) > 0 or score_geral < 0.75

        resultado = {
            "status": "sucesso", "audio_path": str(audio_path),
            "processamento_necessario": processamento_necessario,
            "problemas": problemas, "score_geral": round(score_geral, 3),
            "scores_detalhados": {"dnsmos_ovrl": mos_scores['ovrl'], "hissing": hissing_heu}
        }
        if verbose: self._imprimir_resultado(resultado)
        return convert_numpy_types(resultado)

    def _imprimir_resultado(self, r: dict):
        print(f"\n--- QUALIDADE: {Path(r['audio_path']).name} ---")
        print(f"Score Geral: {r['score_geral']:.1%}")
        for p in r['problemas']: print(f"  ❌ {p}")
        if r['processamento_necessario']: print("  ⚡ AÇÃO: Restaurando...")
        else: print("  ✅ AÇÃO: Preservando original.")

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(output_dir)

    analyzer = AudioAnalyzer()
    enhancer = AudioEnhancer()
    
    print("[INFO] Carregando Whisper...")
    import whisper
    model = whisper.load_model("medium")

    audio_files = sorted(list(input_dir.glob("*.wav")) + list(input_dir.glob("*.mp3")))
    print(f"\n🚀 Processando {len(audio_files)} arquivos...")
    
    metadata = []
    for idx, audio_path in enumerate(audio_files):
        print(f"[{idx+1}/{len(audio_files)}] {audio_path.name}")
        info = analyzer.analyze(str(audio_path), verbose=True)
        
        file_id = f"voz_{idx:04d}_{audio_path.stem}"
        final_wav = output_dir / f"{file_id}.wav"
        
        if info["processamento_necessario"]:
            if not enhancer.process(audio_path, final_wav):
                import shutil
                shutil.copy2(audio_path, final_wav)
        else:
            import shutil
            shutil.copy2(audio_path, final_wav)

        # Padronização final para StyleTTS2
        try:
            y, sr = librosa.load(str(final_wav), sr=24000, mono=True)
            y = librosa.util.normalize(librosa.effects.trim(y, top_db=25)[0]) * 0.95
            sf.write(str(final_wav), y, 24000, subtype='PCM_16')
            
            res = model.transcribe(str(final_wav), language="pt")
            text = res["text"].strip()
            if text: metadata.append(f"{file_id}|{text}|{text}")
        except Exception as e:
            print(f"  [ERRO FINAL] {e}")

    with open("train.txt", "w", encoding="utf-8") as f: f.write("\n".join(metadata))
    print(f"✅ Dataset pronto em: {output_dir}")

if __name__ == "__main__":
    main()
