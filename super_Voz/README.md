# super_Voz

Projeto de treinamento TTS baseado em StyleTTS2.

Fluxo no Colab:

1. Monta o Google Drive.
2. Clona/atualiza este repositório do GitHub.
3. Lê `Audios_brutos` e/ou `Audios_processados` do Drive.
4. Se necessário, usa `limpeza_ia.py` para limpar e transcrever os áudios.
5. Converte o dataset para o formato do StyleTTS2:
   `arquivo.wav|transcricao|speaker`
6. Clona o StyleTTS2 oficial em `/content/StyleTTS2`.
7. Instala dependências, prepara `Data/train_list.txt`, `Data/val_list.txt` e `Configs/config_ft.yml`.
8. Executa fine-tuning com `train_finetune_accelerate.py`.
9. Copia checkpoints e listas para o Drive.

## Estrutura esperada no Drive

O notebook procura automaticamente alguns caminhos. O recomendado é:

```text
MyDrive/super_Voz/
  Audios_brutos/
  Audios_processados/
  checkpoints/
  outputs/
```

Se `Audios_processados/train.txt` já existir, a etapa de limpeza/transcrição é pulada.
Se não existir, o notebook usa `Audios_brutos` como entrada e gera `Audios_processados`.

## Observação importante sobre português

O StyleTTS2 oficial foi publicado principalmente com suporte e checkpoints voltados para inglês. Para português, o pipeline abaixo consegue preparar os dados e iniciar fine-tuning, mas a qualidade final depende de fonemização, dataset e compatibilidade do PL-BERT usado. Para qualidade séria em português, o ideal é usar um PL-BERT compatível/multilíngue ou treinar/adaptar esse componente.
