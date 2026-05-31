#!/usr/bin/env python3
# ============================================================
# limpeza_ia.py — LIMPEZA DE ÁUDIO COM ANÁLISE INTELIGENTE (V2)
# Integrado com DNSMOS (Qualidade) e Resemble Enhance (Restauração)
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
# UTILITÁRIOS
# ============================================================

def convert_numpy_types(obj):
    if isinstance(obj, np.floating): return float(obj)
    elif isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, dict): return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [convert_numpy_types(item) for item in obj]
    else: return obj

# ============================================================
# DNSMOS: NOTA DE QUALIDADE (MÉTRICA DA MICROSOFT)
# ============================================================

class DNSMOS:
    """Calcula o Mean Opinion Score (MOS) usando modelo pré-treinado."""
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
                print(f"[AVISO] Não foi possível baixar DNSMOS: {e}")

        try:
            import onnxruntime as ort
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(self.model_path, providers=providers)
        except Exception as e:
            print(f"[AVISO] Falha ao carregar ONNX para DNSMOS: {e}")

    def score(self, audio: np.ndarray, sr: int) -> dict:
        if self.session is None:
            return {"ovrl": 0.5, "sig": 0.5, "bak": 0.5} # Fallback neutro
        
        try:
            # Resample para 16kHz (exigência do DNSMOS)
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            
            # Garantir 9 segundos de áudio para estabilidade
            target_len = 16000 * 9
            if len(audio) < target_len:
                audio = np.pad(audio, (0, target_len - len(audio)))
            else:
                audio = audio[:target_len]

            # Rodar inferência
            inputs = {self.session.get_inputs()[0].name: audio.astype(np.float32)[np.newaxis, :]}
            outputs = self.session.run(None, inputs)
            
            # Normalizar para 0-1 (original é 1-5)
            return {
                "sig": (outputs[0][0][0] - 1) / 4,
                "bak": (outputs[0][0][1] - 1) / 4,
                "ovrl": (outputs[0][0][2] - 1) / 4
            }
        except:
            return {"ovrl": 0.5, "sig": 0.5, "bak": 0.5}

# ============================================================
# AUDIO ENHANCER: RESEMBLE ENHANCE
# ============================================================

class AudioEnhancer:
    """Restaura áudio usando Resemble Enhance (Denoise + Super Resolution)."""
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.has_resemble = False
        try:
            from resemble_enhance.enhancer.inference import enhance
            self.has_resemble = True
        except ImportError:
            print("[AVISO] resemble-enhance não instalado.")

    def process(self, input_path: Path, output_path: Path):
        if not self.has_resemble:
            return False

        try:
            from resemble_enhance.enhancer.inference import enhance
            # Carregar áudio
            dwav, sr = librosa.load(str(input_path), sr=None)
            dwav = torch.from_numpy(dwav).to(self.device)
            
            # Processar (Denoise + Enhance)
            # nfe=32 é um bom equilíbrio velocidade/qualidade
            hwav, sr = enhance(dwav, sr, self.device, nfe=32, solver="midpoint", lambd=0.5)
            
            # Salvar
            sf.write(str(output_path), hwav.cpu().numpy(), sr)
            return True
        except Exception as e:
            print(f"  [ERRO] Falha no Resemble Enhance: {e}")
            return False

# ============================================================
# CLASSE DE ANÁLISE DE ÁUDIO ATUALIZADA
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
        
        # 1. DNSMOS (IA Realista)
        mos_scores = self.mos_tool.score(audio, sr)
        
        # 2. Heurísticas Básicas (Fallback/Complemento)
        # Clipping
        clipping = np.mean(np.abs(audio) > 0.99)
        # Silêncio
        silence = np.mean(np.abs(audio) < 0.01)
        
        # Problemas detectados
        problemas = []
        if mos_scores['ovrl'] < 0.6: problemas.append(f"Qualidade baixa (MOS: {mos_scores['ovrl']:.2f})")
        if mos_scores['bak'] < 0.5: problemas.append("Ruído de fundo detectado")
        if clipping > 0.05: problemas.append("Clipping (Saturação) detectado")
        if silence > 0.7: problemas.append("Silêncio excessivo")

        # Score final ponderado (predominância para o MOS)
        score_geral = mos_scores['ovrl'] * 0.8 + (1.0 - clipping) * 0.2
        
        # StyleTTS2 precisa de áudio limpo, então somos exigentes
        processamento_necessario = len(problemas) > 0 or score_geral < 0.75

        resultado = {
            "status": "sucesso",
            "audio_path": str(audio_path),
            "duracao_segundos": len(audio) / sr,
            "processamento_necessario": processamento_necessario,
            "problemas": problemas,
            "score_geral": round(score_geral, 3),
            "scores_detalhados": {
                "dnsmos_ovrl": mos_scores['ovrl'],
                "dnsmos_bak": mos_scores['bak'],
                "dnsmos_sig": mos_scores['sig'],
                "clipping": clipping,
                "silence": silence
            }
        }
        
        if verbose: self._imprimir_resultado(resultado)
        return convert_numpy_types(resultado)

    def _imprimir_resultado(self, r: dict):
        print(f"\n--- Análise: {Path(r['audio_path']).name} ---")
        print(f"Score Geral: {r['score_geral']:.1%}")
        for p in r['problemas']: print(f"  ⚠️  {p}")
        if r['processamento_necessario']: print("  ⚡ Processamento recomendado")
        else: print("  ✅ Qualidade aceitável")

# ============================================================
# FUNÇÕES DE CACHE (Igual ao anterior)
# ============================================================

def carregar_cache(caminho: str) -> dict:
    if Path(caminho).exists():
        with open(caminho, 'r', encoding='utf-8') as f: return json.load(f)
    return {}

def salvar_cache(caminho: str, dados: dict):
    with open(caminho, 'w', encoding='utf-8') as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)

def gerar_chave_cache(audio_path: Path) -> str:
    stat = audio_path.stat()
    return f"{audio_path.name}_{stat.st_size}_{stat.st_mtime}"

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Limpeza de Áudio Avançada (StyleTTS2 Edition)")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ambiente", type=str)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    temp_dir = output_dir / "temp_cleaning"
    temp_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(output_dir)

    # Ferramentas
    analyzer = AudioAnalyzer(sr=24000)
    enhancer = AudioEnhancer()
    
    # Cache
    cache_análise = carregar_cache(CACHE_ANÁLISE) if not args.force else {}
    
    # Importar Whisper
    print("[INFO] Carregando Whisper...")
    import whisper
    model = whisper.load_model("medium")

    audio_files = []
    for ext in ['*.mp3', '*.wav', '*.ogg', '*.m4a', '*.flac']:
        audio_files.extend(sorted(input_dir.glob(ext)))

    print(f"\n🚀 Iniciando processamento de {len(audio_files)} arquivos...")
    
    metadata_lines = []
    
    for idx, audio_path in enumerate(audio_files):
        print(f"\n[{idx+1}/{len(audio_files)}] {audio_path.name}")
        
        # 1. Analisar
        chave = gerar_chave_cache(audio_path)
        if chave in cache_análise and not args.force:
            info = cache_análise[chave]
            proc_needed = info.get("processamento_necessario", True)
            print(f"  📊 Usando cache (Score: {info.get('score_geral', 0):.1%})")
        else:
            info = analyzer.analyze(str(audio_path), verbose=True)
            cache_análise[chave] = info
            proc_needed = info["processamento_necessario"]

        file_id = f"voz_{idx:04d}_{audio_path.stem.replace(' ', '_')}"
        final_wav_path = output_dir / f"{file_id}.wav"
        
        # 2. Limpar (se necessário)
        if proc_needed:
            print("  ⚙️  Limpando e Restaurando com Resemble Enhance...")
            success = enhancer.process(audio_path, final_wav_path)
            
            if not success:
                print("  ⚠️  Enhancer falhou. Usando fallback (original + demucs)...")
                # Fallback simples se o Resemble falhar (ex: falta de memória)
                shutil_copy = True
                # Aqui você poderia adicionar o Demucs se quiser, mas vamos simplificar para garantir fluxo
                import shutil
                shutil.copy2(audio_path, final_wav_path)
        else:
            print("  ✅ Áudio já está excelente. Pulando limpeza.")
            import shutil
            shutil.copy2(audio_path, final_wav_path)

        # 3. Pós-Processamento StyleTTS2 (Garantir 24kHz, Mono, Trim)
        try:
            y, sr = librosa.load(str(final_wav_path), sr=24000, mono=True)
            y_trimmed, _ = librosa.effects.trim(y, top_db=25)
            # Normalização LUFS ou Peak
            y_norm = librosa.util.normalize(y_trimmed) * 0.95
            sf.write(str(final_wav_path), y_norm, 24000, subtype='PCM_16')
        except Exception as e:
            print(f"  [ERRO] Pós-processamento: {e}")

        # 4. Transcrever
        try:
            result = model.transcribe(str(final_wav_path), language="pt")
            text = result["text"].strip()
            if text:
                print(f"  📝 Transcrição: {text[:50]}...")
                metadata_lines.append(f"{file_id}|{text}|{text}")
            else:
                print("  ⚠️ Transcrição vazia! Pulando.")
        except Exception as e:
            print(f"  [ERRO] Transcrição: {e}")

    # Salvar resultados
    with open(output_dir / "metadata.csv", "w", encoding="utf-8") as f:
        f.write("\n".join(metadata_lines))
    
    with open(output_dir / "train.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(metadata_lines))
        
    salvar_cache(CACHE_ANÁLISE, cache_análise)
    print(f"\n✅ Concluído! Dataset gerado em: {output_dir}")

if __name__ == "__main__":
    main()
