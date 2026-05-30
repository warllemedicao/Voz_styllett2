# super_Voz

Projeto de treinamento TTS baseado em StyleTTS2.

## Fluxo de Execução (Colab/Kaggle)

O projeto usa o **Cloudflare R2** (S3-compatible) para armazenamento persistente de áudios e checkpoints.

1. **Acesso a Dados:**
   - O pipeline baixa automaticamente os áudios do bucket R2 configurado em `styletts2_colab_config.yml` ou `styletts2_kaggle_config.yml`.
2. **Ambiente:** Clona/atualiza este repositório do GitHub.
3. **Dados:** Sincroniza `Audios_brutos` e/ou `Audios_processados` do bucket.
4. **Limpeza:** Se necessário, usa `limpeza_ia.py` (Demucs + Whisper) para limpar e transcrever.
5. **Conversão:** Converte o dataset para o formato do StyleTTS2 (`wav|texto|speaker`).
6. StyleTTS2: Clona o StyleTTS2 oficial e aplica patches de compatibilidade.
7. Treino: Executa fine-tuning com `accelerate launch`.
8. Finalização: Os checkpoints e resultados ficam disponíveis localmente no Colab (`/content`) ou Kaggle (`/kaggle/working`) para download manual.

## Configuração do Bucket (Cloudflare R2) - APENAS ENTRADA

Para usar este projeto, você deve configurar o Cloudflare R2 apenas para baixar os áudios. A sincronização de saída foi desativada para economizar egress e permitir controle manual dos arquivos.

```yaml
cloudflare_r2:
  endpoint_url: "https://<ACCOUNT_ID>.r2.cloudflarestorage.com"
  access_key_id: "SUA_ACCESS_KEY"
  secret_access_key: "SUA_SECRET_KEY"
  bucket_name: "NOME_DO_BUCKET"
```

### Estrutura do Bucket:
- `Audios_brutos/`: Coloque aqui seus áudios originais para processamento.
- `Audios_processados/`: (Opcional) Dataset já processado (WAVs + `train.txt`).

**Nota:** A pasta `super_Voz_outputs/` não será mais alimentada automaticamente pelo pipeline.

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
4. Ajuste o arquivo `styletts2_kaggle_config.yml` se necessário.
5. Após o treino, baixe os arquivos da pasta `/kaggle/working`.

## Erros Comuns e Correções

- **SIGSEGV (Signal 11):** Geralmente ocorre se o ambiente não detecta a GPU corretamente ou se há incompatibilidade de bibliotecas. O script agora inclui verificações de GPU e patches de compatibilidade para PyTorch 2.6+.
- **CUDA Out of Memory (OOM):** StyleTTS2 é extremamente pesado para a GPU T4 (15GB). O projeto possui mitigação robusta de duas formas:
  1. Aplica patches automáticos no `train_finetune_accelerate.py` para limitar o cálculo do tamanho máximo de validação e referência.
  2. Implementa filtros rigorosos na construção do dataset:
     - `max_len: 128` (Tamanho menor no lote).
     - `max_audio_seconds: 10` (Recusa áudios com mais de 10s no dataset de treinamento para não estourar a memória de alinhamento de sequência). Se o script de áudio falhar ao extrair o tempo, a amostra também é recusada (9999s artificial) para proteger o pipeline de surpresas.
     - `batch_size: 2` (O mínimo que o modelo requer devido às camadas de Batch Normalization no discriminador).

## Observação importante sobre português

O StyleTTS2 oficial foi publicado principalmente com suporte e checkpoints voltados para inglês. Para português, o pipeline abaixo consegue preparar os dados e iniciar fine-tuning, mas a qualidade final depende de fonemização, dataset e compatibilidade do PL-BERT usado.

