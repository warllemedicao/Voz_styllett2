# super_Voz

Projeto de treinamento TTS baseado em StyleTTS2.

## Fluxo de Execução (Colab/Kaggle)

O projeto foi desenhado para rodar em ambientes de nuvem (Google Colab ou Kaggle) com suporte a GPU.

1. **Montagem/Acesso a Dados:**
   - **Colab:** Monta o Google Drive.
   - **Kaggle:** Faz o download dos dados do Google Drive (usando gdown) ou usa Datasets do Kaggle.
2. **Ambiente:** Clona/atualiza este repositório do GitHub.
3. **Dados:** Lê `Audios_brutos` e/ou `Audios_processados`.
4. **Limpeza:** Se necessário, usa `limpeza_ia.py` (Demucs + Whisper) para limpar e transcrever.
5. **Conversão:** Converte o dataset para o formato do StyleTTS2 (`wav|texto|speaker`).
6. **StyleTTS2:** Clona o StyleTTS2 oficial e aplica patches de compatibilidade (PyTorch 2.6+ e Anti-OOM).
7. **Treino:** Executa fine-tuning com `accelerate launch`.
8. **Sincronização:** Copia checkpoints e resultados de volta para o armazenamento persistente (Drive ou Kaggle Output).

## Estrutura de Pastas Recomendada

```text
super_Voz/
  Audios_brutos/       # Áudios originais (mp3, wav, etc)
  Audios_processados/  # Áudios limpos + train.txt (gerado automaticamente)
  checkpoints/         # Checkpoints salvos durante o treino
  outputs/             # Logs e outros artefatos
```

Se `Audios_processados/train.txt` já existir, a etapa de limpeza/transcrição é pulada.

## Como rodar no Kaggle

1. Crie um novo Notebook no Kaggle.
2. Ative a **GPU** (T4 x2 ou P100).
3. Importe o notebook `run_kaggle_styletts2.ipynb` ou copie as células dele.
4. Ajuste o arquivo `styletts2_kaggle_config.yml` se necessário (especialmente IDs do Drive se for baixar dados de lá).

## Erros Comuns e Correções

- **SIGSEGV (Signal 11):** Geralmente ocorre se o ambiente não detecta a GPU corretamente ou se há incompatibilidade de bibliotecas. O script agora inclui verificações de GPU e patches de compatibilidade para PyTorch 2.6+.
- **CUDA Out of Memory (OOM):** StyleTTS2 é pesado. O projeto aplica patches automáticos para reduzir o consumo de memória na fase de validação e referência.

## Observação importante sobre português

O StyleTTS2 oficial foi publicado principalmente com suporte e checkpoints voltados para inglês. Para português, o pipeline abaixo consegue preparar os dados e iniciar fine-tuning, mas a qualidade final depende de fonemização, dataset e compatibilidade do PL-BERT usado.

