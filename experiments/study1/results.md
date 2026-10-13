# Study 1 results

Recomputed by `analyze.py` from `scores.csv` and the billing exports.

## Correctness (G-Eval, 0 to 1)

Delta = Papyrus minus baseline. p and r: Wilcoxon on question means (N=10; N_eff after dropping zero differences). p_corr: Bonferroni k=10. p30: supporting test on the 30 answers, uncorrected.

| Model | Category | Baseline | Papyrus | Delta | N_eff | p | p_corr | r | p30 |
|---|---|---|---|---|---|---|---|---|---|
| GPT-4o | text | 0.75 | 0.47 | -0.28 | 8 | 0.039 | 0.391 | 0.65 | 0.001 |
| Claude 3.5 Sonnet | text | 0.63 | 0.46 | -0.17 | 9 | 0.242 | 1.000 | 0.37 | 0.012 |
| Gemini 2.5 Flash Lite | text | 0.77 | 0.44 | -0.33 | 9 | 0.020 | 0.195 | 0.74 | 0.000 |
| DeepSeek-V3 | text | 0.63 | 0.41 | -0.22 | 8 | 0.117 | 1.000 | 0.50 | 0.005 |
| Llama 3.1 70B | text | 0.68 | 0.47 | -0.21 | 10 | 0.152 | 1.000 | 0.45 | 0.006 |
| GPT-4o | visual | 0.34 | 0.34 | +0.00 | 9 | 0.820 | 1.000 | 0.07 | 0.512 |
| Claude 3.5 Sonnet | visual | 0.35 | 0.35 | -0.01 | 10 | 0.719 | 1.000 | 0.11 | 0.661 |
| Gemini 2.5 Flash Lite | visual | 0.37 | 0.30 | -0.06 | 9 | 0.203 | 1.000 | 0.40 | 0.130 |
| DeepSeek-V3 | visual | 0.38 | 0.40 | +0.02 | 8 | 1.000 | 1.000 | 0.00 | 0.687 |
| Llama 3.1 70B | visual | 0.15 | 0.41 | +0.26 | 9 | 0.039 | 0.391 | 0.65 | 0.001 |
| All models | text | 0.69 | 0.45 | | | | | | |
| All models | visual | 0.32 | 0.36 | | | | | | |

## Faithfulness (Papyrus answers only)

- Mean 0.943 over 300 answers; 84.3% scored 1.0
- Spearman correlation with correctness: 0.19
- GPT-4o: 0.967
- Claude 3.5 Sonnet: 0.929
- Gemini 2.5 Flash Lite: 0.942
- DeepSeek-V3: 0.930
- Llama 3.1 70B: 0.947

## Citations

- Answers that name a source: Papyrus 95.0%, baseline 0.7%
- Counting page mentions too: Papyrus 95.0%, baseline 5.3%

## Cost, tokens and latency (OpenRouter billing)

| Condition | Model | Queries | Total cost (US$) | Cost per query (US$) | Prompt tokens | Generation time (s) |
|---|---|---|---|---|---|---|
| Baseline | GPT-4o | 59 | 2.2804 | 0.03865 | 29,844 | 2.36 |
| Baseline | Claude 3.5 Sonnet | 60 | 12.8420 | 0.21403 | 34,713 | 8.99 |
| Baseline | Gemini 2.5 Flash Lite | 60 | 0.0279 | 0.00047 | 31,839 | 1.22 |
| Baseline | DeepSeek-V3 | 60 | 0.6487 | 0.01081 | 29,620 | 6.80 |
| Baseline | Llama 3.1 70B | 60 | 0.7838 | 0.01306 | 30,211 | 3.47 |
| Baseline | All | 299 | 16.58 | 0.0555 | 31,250 | 4.57 |
| Papyrus | GPT-4o | 60 | 0.2665 | 0.00444 | 1,955 | 1.57 |
| Papyrus | Claude 3.5 Sonnet | 60 | 1.1417 | 0.01903 | 2,299 | 5.72 |
| Papyrus | Gemini 2.5 Flash Lite | 60 | 0.0135 | 0.00023 | 2,066 | 1.92 |
| Papyrus | DeepSeek-V3 | 60 | 0.0477 | 0.00080 | 1,951 | 4.41 |
| Papyrus | Llama 3.1 70B | 60 | 0.0522 | 0.00087 | 1,993 | 2.30 |
| Papyrus | All | 300 | 1.52 | 0.0051 | 2,053 | 3.18 |

- Prompt tokens: 31,250 to 2,053 (93.4% less)
- Total cost: US$16.58 to US$1.52 (90.8% less)
