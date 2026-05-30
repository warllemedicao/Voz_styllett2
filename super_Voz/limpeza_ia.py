#!/usr/bin/env python3
# ============================================================
# limpeza_ia.py — LIMPEZA DE ÁUDIO COM ANÁLISE INTELIGENTE
# Detecta problemas e pula processamento se não for necessário
# INTEGRADO: audio_analyzer.py funcionalidade
# ============================================================

import os
import subprocess
import argparse
import glob
import json
from pathlib import Path
from datetime import datetime
import sys
import numpy as np
import librosa
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURAÇÃO
# ============================================================

CACHE_ANÁLISE = "analise_audio_cache.json"
PROCESSADOS_LOG = "processados.json"

# ============================================================
# CLASSE DE ANÁLISE DE ÁUDIO INTEGRADA
# ============================================================

def convert_numpy_types(obj):
    """
    Converte tipos numpy para tipos Python nativos para JSON serialization.
    """
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

class AudioAnalyzer:
    """
    Analisa áudio para determinar se precisa de processamento.
    Detecta:
    - Ruído (frequências aleatórias)
    - Hissing (frequências altas acima de 8kHz)
    - Sons musicais (harmônicos bem definidos)
    - Silêncios significativos
    """

    def __init__(self, sr: int = 22050, hop_length: int = 256):
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = 1024

    def analyze(self, audio_path: str, verbose: bool = False) -> dict:
        """
        Análise completa de um arquivo de áudio.
        Retorna dicionário com diagnóstico.
        """
        try:
            audio, sr = librosa.load(audio_path, sr=self.sr, mono=True)
        except Exception as e:
            if verbose:
                print(f"[ERRO] Não foi possível carregar {audio_path}: {e}")
            return {
                "status": "erro",
                "erro": str(e),
                "processamento_necessario": True,
                "problemas": ["Erro ao carregar áudio"],
                "score": 0
            }

        audio = audio.astype(np.float32)

        # Computar spectrograma
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=80,
            power=2.0
        )
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Computar STFT
        D = np.abs(librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length))

        # Extrair features
        problemas = []
        scores = {}

        # 1. Detecção de Silêncio
        silencio_score = self._detectar_silencio(audio)
        scores['silencio'] = silencio_score
        if silencio_score > 0.5:
            problemas.append("Silêncio significativo detectado")

        # 2. Detecção de Ruído
        ruido_score = self._detectar_ruido(mel_db)
        scores['ruido'] = ruido_score
        if ruido_score > 0.6:
            problemas.append("Ruído detectado")

        # 3. Detecção de Hissing (assobio)
        hissing_score = self._detectar_hissing(D, sr)
        scores['hissing'] = hissing_score
        if hissing_score > 0.5:
            problemas.append("Hissing (assobio) detectado")

        # 4. Detecção de Sons Musicais
        musical_score = self._detectar_musical(mel_db, D, sr)
        scores['musical'] = musical_score
        if musical_score > 0.6:
            problemas.append("Sons musicais detectados")

        # 5. Razão de Clipping
        clipping_score = self._detectar_clipping(audio)
        scores['clipping'] = clipping_score
        if clipping_score > 0.1:
            problemas.append("Clipping detectado")

        # Score geral de qualidade (0-1, onde 1 é perfeito)
        score_geral = 1.0 - np.mean([
            silencio_score * 0.1,
            ruido_score * 0.3,
            hissing_score * 0.2,
            musical_score * 0.2,
            clipping_score * 0.2
        ])

        # Determinar se processamento é necessário
        processamento_necessario = len(problemas) > 0 or score_geral < 0.7

        resultado = {
            "status": "sucesso",
            "audio_path": str(audio_path),
            "duracao_segundos": len(audio) / sr,
            "sample_rate": sr,
            "processamento_necessario": processamento_necessario,
            "problemas": problemas,
            "score_geral": round(score_geral, 3),
            "scores_detalhados": {k: round(v, 3) for k, v in scores.items()}
        }

        # Converter tipos numpy para JSON serialization
        resultado = convert_numpy_types(resultado)

        if verbose:
            self._imprimir_resultado(resultado)

        return resultado

    def _detectar_silencio(self, audio: np.ndarray) -> float:
        """Detecta porcentagem de silêncio no áudio."""
        # Define silêncio como amplitude < 0.01
        threshold = 0.01
        silencio = np.abs(audio) < threshold
        porcentagem_silencio = np.mean(silencio)

        # Score de 0 a 1, onde 1 significa 100% de silêncio
        return min(porcentagem_silencio, 1.0)

    def _detectar_ruido(self, mel_db: np.ndarray) -> float:
        """
        Detecta ruído usando análise espectral.
        Ruído tem distribuição plana no espectro.
        """
        # Calcular a variância espectral ao longo do tempo
        # Se é constante, provavelmente é ruído
        media_espectral = np.mean(mel_db, axis=1)
        desvio_espectral = np.std(media_espectral)

        # Ruído branco tem desvio baixo (espectro plano)
        # Fala tem picos em bandas específicas
        # Normalizar para 0-1
        ruido_score = max(0, 1 - (desvio_espectral / 10.0))

        return min(ruido_score, 1.0)

    def _detectar_hissing(self, D: np.ndarray, sr: int) -> float:
        """
        Detecta hissing (assobio) em frequências altas (>8kHz).
        Hissing é muito concentrado em frequências altas.
        """
        # Converter para escala de frequência
        freq_bins = np.fft.rfftfreq(self.n_fft, 1/sr)

        # Frequências acima de 8kHz
        hissing_threshold = 8000
        hissing_idx = freq_bins > hissing_threshold

        if not np.any(hissing_idx):
            return 0.0

        # Energia em frequências altas vs total
        energia_hissing = np.mean(np.abs(D[hissing_idx, :]))
        energia_total = np.mean(np.abs(D))

        if energia_total == 0:
            return 0.0

        razao_hissing = energia_hissing / energia_total

        # Score: quanto mais energia em altas frequências, mais hissing
        # Esperamos menos de 5% de energia em altas frequências
        return min(razao_hissing * 5, 1.0)

    def _detectar_musical(self, mel_db: np.ndarray, D: np.ndarray, sr: int) -> float:
        """
        Detecta sons musicais (harmônicos bem definidos).
        Música tem picos espectrais claros.
        """
        # Calcular crista do espectro (razão entre pico e média)
        media_tempo = np.mean(mel_db, axis=1)
        picos = np.max(mel_db, axis=1)

        # Razão de pico
        razao_pico = np.mean(picos) - np.mean(media_tempo)

        # Música tem razão de pico mais alta
        # Normalizar (esperamos ~10-20dB de razão)
        musical_score = max(0, (razao_pico / 20.0))

        # Também verificar periodicidade do espectro (harmônicos)
        diferenca_freq = np.abs(np.diff(np.mean(D, axis=1)))
        periodicidade = np.std(diferenca_freq)

        # Harmônicos bem definidos têm periodicidade alta
        harmonica_score = max(0, (periodicidade / 100.0))

        return min((musical_score + harmonica_score) / 2, 1.0)

    def _detectar_clipping(self, audio: np.ndarray) -> float:
        """Detecta clipping (distorção por saturação)."""
        # Valores muito próximos de 1.0 ou -1.0 indicam clipping
        clipping_threshold = 0.99
        clipped = np.abs(audio) > clipping_threshold
        razao_clipping = np.mean(clipped)

        return min(razao_clipping * 10, 1.0)

    def _imprimir_resultado(self, resultado: dict):
        """Imprime resultado da análise de forma legível."""
        print("\n" + "="*60)
        print("ANÁLISE DE ÁUDIO")
        print("="*60)

        print(f"Arquivo: {Path(resultado.get('audio_path', 'desconhecido')).name}")
        print(f"Duração: {resultado.get('duracao_segundos', 0):.2f}s")
        print(f"Taxa de amostragem: {resultado.get('sample_rate', 0)} Hz")
        print(f"Score geral: {resultado.get('score_geral', 0):.1%}")

        print("\nProblemas detectados:")
        if resultado.get('problemas'):
            for problema in resultado['problemas']:
                print(f"  ⚠️  {problema}")
        else:
            print("  ✅ Nenhum problema detectado")

        print("\nScores detalhados:")
        for chave, valor in resultado.get('scores_detalhados', {}).items():
            barra = "█" * int(valor * 20) + "░" * (20 - int(valor * 20))
            print(f"  {chave:12s}: [{barra}] {valor:.1%}")

        if resultado.get('processamento_necessario'):
            print("\n⚡ Processamento NECESSÁRIO")
        else:
            print("\n✅ Áudio de qualidade aceitável (pode pular processamento)")

        print("="*60 + "\n")

    def analisar_batch(self, audio_files: list, cache_path: str = None) -> dict:
        """
        Analisa múltiplos arquivos com cache opcional.
        """
        resultados = {}
        cache = {}

        if cache_path and Path(cache_path).exists():
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache = json.load(f)

        for audio_path in audio_files:
            chave = str(Path(audio_path).stat().st_mtime)  # Use mtime como cache key

            if chave in cache:
                resultados[str(audio_path)] = cache[chave]
            else:
                resultado = self.analyze(audio_path, verbose=True)
                resultados[str(audio_path)] = resultado
                cache[chave] = resultado

        # Salvar cache
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            # Converter tipos numpy antes de salvar
            cache_convertido = {k: convert_numpy_types(v) for k, v in cache.items()}
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_convertido, f, ensure_ascii=False, indent=2)

        return resultados

# ============================================================
# FUNÇÕES DE GERENCIAMENTO DE CACHE
# ============================================================

def carregar_cache(caminho: str) -> dict:
    """Carrega cache de análises anteriores."""
    if Path(caminho).exists():
        with open(caminho, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def salvar_cache(caminho: str, dados: dict):
    """Salva cache de análises."""
    with open(caminho, 'w', encoding='utf-8') as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)

def gerar_chave_cache(audio_path: Path) -> str:
    """
    Gera chave única para um arquivo de áudio.
    Usa tamanho do arquivo + modificação.
    """
    try:
        stat = audio_path.stat()
        return f"{audio_path.name}_{stat.st_size}_{stat.st_mtime}"
    except OSError as e:
        print(f"[AVISO] Falha ao gerar chave de cache para {audio_path}: {e}")
        return audio_path.name

def analisar_audio_necessario(audio_path: Path, cache_análise: dict) -> tuple:
    """
    Retorna (processamento_necessário, info_análise)
    """
    # Usar a classe AudioAnalyzer integrada
    analyzer = AudioAnalyzer()
    chave = gerar_chave_cache(audio_path)

    # Verificar se já foi analisado
    if chave in cache_análise:
        análise_anterior = cache_análise[chave]
        print(f"\n  📊 Usando análise anterior para {audio_path.name}")
        print(f"     Score geral: {análise_anterior.get('score_geral', '?'):.1%}")
        print(f"     Processamento necessário: {análise_anterior.get('processamento_necessario', True)}")

        return análise_anterior.get("processamento_necessario", True), análise_anterior

    # Executar nova análise
    print(f"\n  🔍 Analisando áudio: {audio_path.name}")
    resultado = analyzer.analyze(str(audio_path), verbose=False)

    # Armazenar no cache
    cache_análise[chave] = resultado

    return resultado.get("processamento_necessario", True), resultado

# ============================================================
# FUNÇÕES DE DETECÇÃO DE AMBIENTE
# ============================================================

def detectar_ambiente():
    """Detecta se está rodando no Colab, Kaggle ou local."""
    ambiente = "local"
    
    # Verificar Colab
    try:
        import google.colab
        ambiente = "colab"
    except ImportError:
        # Verificar Kaggle
        try:
            import kagglehub
            ambiente = "kaggle"
        except ImportError:
            ambiente = "local"
    
    print(f"[INFO] Ambiente detectado: {ambiente}")
    return ambiente

def configurar_caminhos(ambiente: str):
    """Configura caminhos baseados no ambiente detectado."""
    if ambiente == "colab":
        # Caminhos típicos do Colab com Google Drive
        base_path = "/content/drive/MyDrive"
        audios_brutos = f"{base_path}/Audios_brutos"
        audios_processado = f"{base_path}/Audios_processado"
    elif ambiente == "kaggle":
        # Caminhos típicos do Kaggle
        base_path = "/tmp"
        audios_brutos = f"{base_path}/Audios_brutos"
        audios_processado = f"{base_path}/Audios_processado"
    else:
        # Ambiente local - usar caminhos relativos
        audios_brutos = "Audios_brutos"
        audios_processado = "Audios_processado"
    
    return audios_brutos, audios_processado

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Limpeza de Áudio com Análise Inteligente")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Diretório com os áudios originais (sujos)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Diretório onde salvar os áudios limpos e o dataset")
    parser.add_argument("--force", action="store_true",
                        help="Força reprocessamento mesmo se análise diz para pular")
    parser.add_argument("--skip-analysis", action="store_true",
                        help="Pula análise de áudio e processa tudo")
    parser.add_argument("--ambiente", type=str, choices=["local", "colab", "kaggle"],
                        help="Força ambiente (detecta automaticamente se não especificado)")

    args = parser.parse_args()

    # Detectar ambiente
    if args.ambiente:
        ambiente = args.ambiente
    else:
        ambiente = detectar_ambiente()

    # Configurar caminhos
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    demucs_out = output_dir / "demucs_temp"
    
    # MUDANÇA: Agora salvamos os wavs diretamente na raiz do output_dir
    dataset_wavs = output_dir

    # Criar pastas temporárias se necessário
    demucs_out.mkdir(parents=True, exist_ok=True)
    # dataset_wavs já existe como output_dir, garantimos apenas que output_dir existe
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mudar para output_dir para salvar caches lá
    os.chdir(output_dir)

    # Carregar caches
    cache_análise = carregar_cache(CACHE_ANÁLISE) if not args.force else {}
    processados_log = carregar_cache(PROCESSADOS_LOG) if not args.force else {}

    # Procurar arquivos de áudio
    audio_files = []
    for ext in ['*.mp3', '*.wav', '*.ogg', '*.m4a']:
        audio_files.extend(sorted(input_dir.glob(ext)))

    if not audio_files:
        print(f"[AVISO] Nenhum arquivo de áudio encontrado em {input_dir}")
        return

    print("="*70)
    print(" 🎤 LIMPEZA DE ÁUDIO COM ANÁLISE INTELIGENTE")
    print("="*70)
    print(f"Entrada: {input_dir}")
    print(f"Saída: {output_dir}")
    print(f"Arquivos encontrados: {len(audio_files)}")
    print(f"Modo: {'Forçado (reprocessar tudo)' if args.force else 'Normal (usar cache)'}")
    print(f"Ambiente: {ambiente}")

    # Importar Whisper
    print("\n[INFO] Carregando modelo Whisper para transcrição...")
    import whisper
    model = whisper.load_model("medium")

    metadata_lines = []
    processados_agora = []
    pulados = []

    for idx, audio_path in enumerate(audio_files):
        print(f"\n[{idx+1}/{len(audio_files)}] Processando: {audio_path.name}")

        # =========================================================
        # ANÁLISE DE ÁUDIO
        # =========================================================
        if not args.skip_analysis:
            processamento_necessario, info_análise = analisar_audio_necessario(
                audio_path, cache_análise
            )

            if not processamento_necessario and not args.force:
                print(f"  ✅ Áudio já está bom. Pulando Demucs.")
                # Usar o áudio original como "limpo"
                vocal_path = audio_path
                pulados.append(audio_path.name)
            else:
                processamento_necessario = True
        else:
            processamento_necessario = True

        if not processamento_necessario and not args.force:
            # Usar áudio original
            vocal_path = audio_path
            print(f"  ⏭️  Pulando processamento para {audio_path.name}")
        else:
            # Processar com Demucs
            print("  ⚙️  Rodando Demucs (separação de voz)...")
            cmd_demucs = [
                "demucs",
                "--two-stems=vocals",
                "-o", str(demucs_out),
                str(audio_path)
            ]

            result = subprocess.run(cmd_demucs, capture_output=True, text=True)
            if result.returncode != 0 and "No such file or directory" not in result.stderr:
                print(f"  ⚠️  Demucs retornou código {result.returncode}")

            # O Demucs salva em: demucs_temp/htdemucs/{nome_do_arquivo}/vocals.wav
            vocal_path = demucs_out / "htdemucs" / audio_path.stem / "vocals.wav"

            if not vocal_path.exists():
                print(f"  [ERRO] Falha ao extrair voz. Usando original...")
                vocal_path = audio_path

        # =========================================================
        # SALVAR ÁUDIO LIMPO
        # =========================================================
        file_id = f"voz_{idx:04d}_{audio_path.stem.replace(' ', '_')}"
        final_wav_path = dataset_wavs / f"{file_id}.wav"

        import shutil
        try:
            shutil.copy2(vocal_path, final_wav_path)
            print(f"  ✅ Áudio salvo: {final_wav_path.name}")
        except Exception as e:
            print(f"  [ERRO] Falha ao salvar áudio: {e}")
            continue

        # =========================================================
        # TRANSCREVER COM WHISPER
        # =========================================================
        print("  🎙️  Rodando Whisper (transcrição)...")
        try:
            result = model.transcribe(str(final_wav_path), language="pt")
            text = result["text"].strip()

            if not text:
                print(f"  [AVISO] Whisper retornou texto vazio!")
                text = "VAZIO"

            print(f"  📝 Transcrição: {text[:50]}...")
            metadata_lines.append(f"{file_id}|{text}|{text}")
            processados_agora.append(file_id)

        except Exception as e:
            print(f"  [ERRO] Falha na transcrição: {e}")

    # =========================================================
    # SALVAR METADADOS E TRAIN.TXT
    # =========================================================
    metadata_path = output_dir / "metadata.csv"
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("\n".join(metadata_lines))

    train_txt_path = output_dir / "train.txt"
    with open(train_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(metadata_lines))

    print(f"  ✅ Gerado metadata.csv e train.txt em {output_dir}")

    # =========================================================
    # SALVAR CACHES
    # =========================================================
    salvar_cache(CACHE_ANÁLISE, cache_análise)
    processados_log["timestamp"] = datetime.now().isoformat()
    processados_log["processados"] = processados_agora
    processados_log["pulados"] = pulados
    salvar_cache(PROCESSADOS_LOG, processados_log)

    # =========================================================
    # RELATÓRIO FINAL
    # =========================================================
    print("\n" + "="*70)
    print(" ✅ PROCESSAMENTO CONCLUÍDO!")
    print("="*70)
    print(f"Áudios limpos: {dataset_wavs}")
    print(f"Metadados: {metadata_path}")
    print(f"Processados: {len(processados_agora)}")
    if pulados:
        print(f"Pulados (já estavam bons): {len(pulados)}")
        for nome in pulados[:5]:
            print(f"  - {nome}")
        if len(pulados) > 5:
            print(f"  ... e mais {len(pulados)-5}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
