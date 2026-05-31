# Histórico do Projeto super_Voz - Combate ao ZeroDivisionError

## Problema Recorrente
O treinamento do StyleTTS2 falha com `ZeroDivisionError: division by zero` no script `train_finetune_accelerate.py`.

## Diagnóstico
Embora tenhamos aplicado um patch matemático para evitar a divisão por zero (`iters_test = max(1, iters_test)`), o fato de o erro persistir ou de a validação resultar em `0` iterações indica que o **Dataset de Validação está sendo totalmente rejeitado** pelo StyleTTS2.

### Possíveis Causas nos Áudios Processados:
1. **Silêncios Longos:** Áudios com muito silêncio no início/fim podem ser filtrados ou causar falhas no alinhamento.
2. **Formato Incompatível:** O StyleTTS2 é extremamente rígido. Ele espera:
   - Sample Rate específico (geralmente 24kHz).
   - Áudio Mono.
   - Bit depth de 16-bit PCM.
   - Sem silêncios excessivos (o modelo tenta alinhar texto -> áudio; se houver áudio sem fala correspondente, ele falha).
3. **Duração:** Áudios muito curtos (< 1s) ou muito longos (> 12s) costumam ser descartados pelo dataloader interno.

## Plano de Ação (30/05/2026)
1. **Documentar Histórico:** Criação deste arquivo `super_voz.md`.
2. **Forçar Reprocessamento:** Remover a busca por `Audios_processados` no config para garantir que o `limpeza_ia.py` rode do zero.
3. **Otimizar `limpeza_ia.py`:** Revisar o script para garantir que ele aplique:
   - Trim de silêncio agressivo.
   - Normalização de volume.
   - Conversão exata para o formato StyleTTS2.

## Melhoria na Qualidade de Áudio (31/05/2026)
Implementação de ferramentas de estado-da-arte para análise e limpeza, focando na qualidade exigida pelo StyleTTS2.

### Novas Tecnologias Integradas:
1. **DNSMOS (Microsoft):** Substituímos a análise manual por uma rede neural que dá notas de 1 a 5 para a qualidade da voz (MOS). Isso evita processar áudios que já estão perfeitos e garante que áudios ruins sejam detectados com precisão.
2. **Resemble Enhance:** Substituímos o Demucs pelo Resemble Enhance como ferramenta principal. Ele não apenas limpa o ruído, mas faz **Super-Resolution**, reconstruindo frequências perdidas em áudios de baixa qualidade (ex: gravações de WhatsApp ou microfones baratos).
3. **Sistema Híbrido de Análise:** Restauramos as **Heurísticas de Ruído e Assobio (Hissing)** para trabalhar em conjunto com a IA. Agora, o programa reporta exatamente quais defeitos foram encontrados (ex: "Ruído constante", "Chiado agudo"), dando mais transparência ao usuário.

### Impacto no Processo:
- **Segurança:** O programa agora é mais inteligente. Se o `DNSMOS` der uma nota alta, o áudio original é preservado para evitar artefatos de IA.
- **Fidelidade StyleTTS2:** O áudio final é garantido em 24kHz, Mono, 16-bit PCM e normalizado em -1dB, eliminando a principal causa do `ZeroDivisionError`.
- **Robustez de Instalação:** Corrigidos conflitos de dependências no Colab/Kaggle, garantindo que o `resemble-enhance` e o `onnxruntime-gpu` carreguem com sucesso.

## Solução Definitiva de Qualidade e Robustez (31/05/2026)
Após análise dos logs de execução e pesquisa técnica, implementamos a solução final para as ferramentas de IA.

### Correções Críticas Realizadas:
1. **DNSMOS (Dimensões ONNX):** Identificamos que o modelo ONNX da Microsoft exige exatamente **144.160 samples** (9.01 segundos a 16kHz). O erro `INVALID_ARGUMENT` ocorria porque estávamos enviando 144.000. Ajustamos o preenchimento (padding) para a dimensão exata exigida pelo motor de IA.
2. **Resemble Enhance (DeepSpeed):** Descobrimos que a IA de restauração exige a biblioteca `deepspeed` em tempo de execução, mesmo com a instalação `--no-deps`. Adicionamos a instalação do `deepspeed` com a flag `DS_BUILD_OPS=0`, o que permite uma instalação rápida e compatível com o Colab/Kaggle sem necessidade de compilação demorada.
3. **Sincronização de Repositório:** Ajustamos a lógica do `git pull`. Agora o pipeline força o `checkout main` antes de tentar atualizar, eliminando o erro "You are not currently on a branch" que ocorria em sessões reiniciadas no Google Drive.

### Melhoria na Transparência:
- Criamos o **📊 RELATÓRIO DE QUALIDADE** no terminal. Agora você vê exatamente por que um áudio foi rejeitado ou limpo, com notas de 1 a 5 para a IA e porcentagens reais para chiados, ruídos e saturação.

## Modificações Realizadas
- [x] Criação de `super_voz.md`.
- [x] Atualização de `styletts2_colab_config.yml` (removendo candidatos de áudios processados e ativando Google Drive).
- [x] Upgrade do `limpeza_ia.py` para a Versão 2 (DNSMOS + Resemble Enhance).
- [x] Correção técnica do DNSMOS (144160 samples) e restauração das heurísticas detalhadas.
- [x] Atualização do `run_pipeline.py`, `run_colab_styletts2.py` e `run_kaggle_styletts2.py` para incluir `deepspeed` e suporte robusto a Git.
- [x] Adição do script de auto-montagem do Google Drive no notebook do Colab.

## ⚠️ AVISO IMPORTANTE SOBRE COLAB/KAGGLE
O ambiente do Colab e Kaggle **clona este repositório do GitHub**. 
Se as modificações feitas aqui não forem enviadas para o seu GitHub (**git commit** e **git push**), o Colab continuará rodando a versão antiga e o erro persistirá.

**Para que a correção funcione no Colab:**
1. Salve todas as alterações.
2. Faça o `commit` e `push` para o seu repositório.
3. Reinicie a execução no Colab.

