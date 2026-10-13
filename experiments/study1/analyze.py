"""Study 1 statistics, recomputed from the stored judge scores and OpenRouter billing exports.

Reads scores.csv, billing_baseline.csv and billing_papyrus.csv; writes results.md. No API calls.
Unit of analysis: the question. The three runs are averaged per question, giving N=10 paired
observations per model and category. Two-sided Wilcoxon signed-rank test, Bonferroni k=10,
r = |Z|/sqrt(N). A supporting test on the 30 individual answers is also reported. Means are rounded
to 6 decimals so that floating-point noise does not create spurious nonzero differences."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, norm

HERE = Path(__file__).resolve().parent
MODELS = ["GPT-4o", "Claude 3.5 Sonnet", "Gemini 2.5 Flash Lite", "DeepSeek-V3", "Llama 3.1 70B"]
KEYS = [("gpt-4o", "GPT-4o"), ("claude", "Claude 3.5 Sonnet"), ("deepseek", "DeepSeek-V3"),
        ("llama", "Llama 3.1 70B"), ("gemini", "Gemini 2.5 Flash Lite")]
CITATION = r"(?i)source:|page \d|pg\.|p\. ?\d|\.pdf"


def model_name(slug):
    slug = slug.lower()
    return next(v for k, v in KEYS if k in slug)


def test(b, p):
    """Two-sided Wilcoxon signed-rank test of p against b; returns p-value and r."""
    b, p = np.round(b, 6), np.round(p, 6)
    if np.allclose(b, p):
        return 1.0, 0.0
    pv = wilcoxon(p, b, zero_method="wilcox", alternative="two-sided").pvalue
    return pv, abs(norm.ppf(pv / 2)) / np.sqrt(len(b))


d = pd.read_csv(HERE / "scores.csv")
d["M"] = d.model.map(model_name)
d["cond"] = np.where(d.model.str.startswith("Papyrus"), "papyrus", "baseline")
out = ["# Study 1 results", "", "Recomputed by `analyze.py` from `scores.csv` and the billing exports.", ""]

out += ["## Correctness (G-Eval, 0 to 1)", "",
        "Delta = Papyrus minus baseline. p and r: Wilcoxon on question means (N=10). "
        "p_corr: Bonferroni k=10. p30: supporting test on the 30 answers, uncorrected.", "",
        "| Model | Category | Baseline | Papyrus | Delta | p | p_corr | r | p30 |",
        "|---|---|---|---|---|---|---|---|---|"]
for cat in ["text", "visual"]:
    for m in MODELS:
        g = d[(d.M == m) & (d.category == cat)]
        qb = g[g.cond == "baseline"].groupby("id").score_correctness.mean()
        qp = g[g.cond == "papyrus"].groupby("id").score_correctness.mean()
        pv, r = test(qb.values, qp.loc[qb.index].values)
        ab = g[g.cond == "baseline"].sort_values(["id", "iteration"]).score_correctness.values
        ap = g[g.cond == "papyrus"].sort_values(["id", "iteration"]).score_correctness.values
        p30, _ = test(ab, ap) if len(ab) == len(ap) else (np.nan, np.nan)
        out.append(f"| {m} | {cat} | {qb.mean():.2f} | {qp.mean():.2f} | {qp.mean() - qb.mean():+.2f} "
                   f"| {pv:.3f} | {min(1, pv * 10):.3f} | {r:.2f} | {p30:.3f} |")
for cat in ["text", "visual"]:
    g = d[d.category == cat]
    out.append(f"| All models | {cat} | {g[g.cond == 'baseline'].score_correctness.mean():.2f} "
               f"| {g[g.cond == 'papyrus'].score_correctness.mean():.2f} | | | | | |")

f = d[d.cond == "papyrus"]
out += ["", "## Faithfulness (Papyrus answers only)", "",
        f"- Mean {f.score_faithfulness.mean():.3f} over {f.score_faithfulness.notna().sum()} answers; "
        f"{(f.score_faithfulness == 1).mean() * 100:.1f}% scored 1.0",
        f"- Spearman correlation with correctness: "
        f"{f[['score_faithfulness', 'score_correctness']].corr('spearman').iloc[0, 1]:.2f}"]
for m in MODELS:
    out.append(f"- {m}: {f[f.M == m].score_faithfulness.mean():.3f}")

cit_p = f.actual_output.astype(str).str.contains(CITATION).mean()
cit_b = d[d.cond == "baseline"].actual_output.astype(str).str.contains(r"(?i)source:|\.pdf").mean()
out += ["", "## Citations", "",
        f"- Papyrus answers with a source line: {cit_p * 100:.1f}%",
        f"- Baseline answers that mention a source: {cit_b * 100:.1f}%"]

out += ["", "## Cost, tokens and latency (OpenRouter billing)", "",
        "| Condition | Model | Queries | Total cost (US$) | Cost per query (US$) | Prompt tokens | Generation time (s) |",
        "|---|---|---|---|---|---|---|"]
total = {}
for name, fn in [("Baseline", "billing_baseline.csv"), ("Papyrus", "billing_papyrus.csv")]:
    a = pd.read_csv(HERE / fn)
    a["M"] = a.model_permaslug.map(model_name)
    for m in MODELS:
        g = a[a.M == m]
        out.append(f"| {name} | {m} | {len(g)} | {g.cost_total.sum():.4f} | {g.cost_total.mean():.5f} "
                   f"| {g.tokens_prompt.mean():,.0f} | {g.generation_time_ms.mean() / 1000:.2f} |")
    out.append(f"| {name} | All | {len(a)} | {a.cost_total.sum():.2f} | {a.cost_total.mean():.4f} "
               f"| {a.tokens_prompt.mean():,.0f} | {a.generation_time_ms.mean() / 1000:.2f} |")
    total[name] = a
b, p = total["Baseline"], total["Papyrus"]
out += ["", f"- Prompt tokens: {b.tokens_prompt.mean():,.0f} to {p.tokens_prompt.mean():,.0f} "
        f"({(1 - p.tokens_prompt.mean() / b.tokens_prompt.mean()) * 100:.1f}% less)",
        f"- Total cost: US${b.cost_total.sum():.2f} to US${p.cost_total.sum():.2f} "
        f"({(1 - p.cost_total.sum() / b.cost_total.sum()) * 100:.1f}% less)"]

(HERE / "results.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
