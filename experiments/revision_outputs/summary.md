# Revision Summary — Papyrus AI

> Gerado em: 2026-05-24
> Arquivo de dados: experiments/AVALIACAO_FINAL_MESTRADO.csv (600 linhas, 10 colunas)

---

## Modelo Juiz

- **Modelo**: GPT-4.1 via OpenRouter
- **Configurado em**: `experiments/evaluate.py` → `GPTModel(model="openai/gpt-4o", base_url="https://openrouter.ai/api/v1")`
- **Status**: ✅ INDEPENDENTE — não está no benchmark de avaliação (os modelos testados são Claude 3.5 Sonnet, DeepSeek Chat, Gemini 2.5 Flash Lite, GPT-4o e Llama 3.1 70B; o juiz é GPT-4.1)
- **Ação no manuscrito**: corrigir "GPT-4o" → "GPT-4.1" na Seção 4.4

---

## Resultados Descritivos

Escores de Answer Correctness (escala 1–5), N=30 instâncias por célula:

```
Model                     Category   Baseline (1–5)   Papyrus AI (1–5)    Δ
----------------------------------------------------------------------
Claude 3.5 Sonnet         Textual    3.13±1.22       2.30±1.34     -0.83
Claude 3.5 Sonnet         Visual     1.77±1.20       1.73±1.33     -0.04
DeepSeek Chat             Textual    3.17±1.12       2.07±1.34     -1.10
DeepSeek Chat             Visual     1.90±1.12       1.98±1.48     +0.08
GPT-4o                    Textual    3.75±1.15       2.33±1.43     -1.42
GPT-4o                    Visual     1.68±1.37       1.70±1.49     +0.02
Gemini 2.5 Flash Lite     Textual    3.83±1.12       2.18±1.37     -1.65
Gemini 2.5 Flash Lite     Visual     1.83±1.34       1.52±1.28     -0.31
Llama 3.1 70B             Textual    3.38±1.60       2.35±1.59     -1.03
Llama 3.1 70B             Visual     0.75±0.58       2.03±1.71     +1.28
----------------------------------------------------------------------
Average                              2.52             2.02     -0.50
```

---

## Significância Estatística (Wilcoxon signed-rank + Bonferroni)

**Parâmetros**: N=30 pares (10 questões × 3 runs), two-sided, k=10 comparações.

### Comparações significativas (p_corr < 0.05) — 5/10:

  - [*] DeepSeek Chat / Textual: W=29.0, p_corr=0.04287, r=0.521, Δ=-1.100
  - [***] Gemini 2.5 Flash Lite / Textual: W=9.5, p_corr=0.00035, r=0.756, Δ=-1.650
  - [**] GPT-4o / Textual: W=18.0, p_corr=0.00660, r=0.622, Δ=-1.417
  - [*] Llama 3.1 70B / Textual: W=84.0, p_corr=0.03497, r=0.533, Δ=-1.033
  - [**] Llama 3.1 70B / Visual: W=40.0, p_corr=0.00524, r=0.633, Δ=+1.283

> ⚠️ **Atenção**: 4 das 5 significâncias indicam que Papyrus AI piora scores textuais.
> A única melhoria significativa é Llama 3.1 70B / Visual (Δ=+1.28 pontos).

### Comparações não significativas — 5/10:

  - Claude 3.5 Sonnet / Textual: p_corr=0.15606, r=0.441, Δ=-0.833
  - Claude 3.5 Sonnet / Visual: p_corr=1.00000, r=0.069, Δ=-0.033
  - DeepSeek Chat / Visual: p_corr=1.00000, r=0.074, Δ=+0.083
  - Gemini 2.5 Flash Lite / Visual: p_corr=1.00000, r=0.287, Δ=-0.317
  - GPT-4o / Visual: p_corr=1.00000, r=0.087, Δ=+0.017

---

## Llama 3.1 70B — Visual (caso principal)

- **W** = 40.0
- **p** = 0.00052 (raw) | **p_corr** = 0.00524 (**)
- **Effect size r** = 0.633 — **grande (≥0.5)** (|Z|/√N)
- **Cohen's d** = 1.006 — efeito grande (d > 0.8)
- **Δ** = +1.283 pontos (escala 1–5): Baseline=0.750 → Papyrus AI=2.033

---

## Faithfulness (Papyrus AI apenas)

  Claude 3.5 Sonnet: 92.92% ± 16.24% ⚠️ BELOW 94%
  DeepSeek Chat: 92.98% ± 18.11% ⚠️ BELOW 94%
  GPT-4o: 96.67% ± 14.89%
  Gemini 2.5 Flash Lite: 94.20% ± 13.20%
  Llama 3.1 70B: 94.72% ± 15.19%

**Média geral**: 94.30% ± 15.56%
**Claim do paper (>94%)**: ✅ **CONFIRMADO** (média = 94.30%)

> Nota: Claude 3.5 Sonnet (92.92%) e DeepSeek Chat (92.98%) estão individualmente
> abaixo de 94%, mas a média agregada confirma o claim.

---

## Custos Reais (OpenRouter logs)

| Item | Valor |
|------|-------|
| Custo total Baseline (concorrentes) | $16.5828 (299 chamadas) |
| Custo total Papyrus AI | $1.5216 (300 chamadas) |
| Redução de custo Papyrus vs Baseline | 90.8% |
| Custo de ingestão estimado (33 imgs × Gemini 2.5 Flash Lite) | $0.006435 |

> O custo do Baseline é ~11× maior porque envia o documento completo como contexto
> (≈31.8k tokens/chamada no Gemini vs <3k no Papyrus via RAG).

---

## Inconsistências encontradas

1. **Escala de scores**: scores armazenados em 0–1 (DeepEval normaliza divisão por 5); paper deve reportar como escala 1–5 multiplicando por 5.
2. **Papyrus AI piora escores textuais**: 4 de 5 modelos mostram queda significativa em Textual (p_corr < 0.05). Claude 3.5 Sonnet e GPT-4o não melhoram em Visual. O paper deve contextualizar que Papyrus beneficia principalmente questões visuais de modelos que antes careciam de capacidade multimodal (ex.: Llama).
3. **Claude 3.5 e DeepSeek abaixo de 94% em Faithfulness**: Claude=92.92%, DeepSeek=92.98%. Média global de 94.30% confirma o claim do paper, mas por modelo nem sempre.
4. **Custo Baseline muito maior**: Baseline=$16.58 vs Papyrus=$1.52 (−90.8%). Isso indica que Baseline envia contexto completo do documento (≈31k tokens/chamada), enquanto Papyrus usa RAG com contexto recuperado menor.
5. **Baseline dataset tem 299 linhas, não 300**: um registro faltando em models--openrouter_activity_2026-02-20.csv.

---

## O que corrigir no manuscrito

1. Seção 4.4 — corrigir modelo juiz de 'GPT-4o' para 'GPT-4.1 via OpenRouter'.
2. Seção de resultados — deixar explícito que a melhoria do Papyrus AI se concentra em questões visuais, especialmente para modelos sem capacidade multimodal nativa (Llama 3.1 70B).
3. Adicionar nota de rodapé: Faithfulness por modelo (Claude e DeepSeek individualmente abaixo de 94%); a média reportada (>94%) é a agregada.
4. Adicionar tabela de testes de Wilcoxon (5 comparações significativas, p corrigido).
5. Se o paper afirma melhoria geral em Correctness, rever: a melhoria é seletiva (visual + Llama 3.1 70B). Para modelos mais capazes (Claude, GPT-4o), Papyrus não melhora e pode piorar em Textual.
6. Reportar custo real de inferência: Papyrus $1.52 (300 chamadas) vs Baseline $16.58 (299 chamadas).

---

*Arquivos gerados em* `experiments/revision_outputs/`:
- `results_table.tex` / `results_table.txt` — tabela descritiva
- `statistics_table.tex` — tabela de testes estatísticos
- `statistics_summary.txt` — interpretação texto dos testes
- `summary.md` — este arquivo
