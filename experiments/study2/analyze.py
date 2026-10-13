"""Analysis of the unified-settings ablation (Study 2).

Reads generations.csv, scores.csv, evidence_map.csv, records.json, retrieval_study.csv and
judge_check.csv; writes analysis.json, the summary tables in tables/, and the figures in figs/.
Primary unit: the question (N=10 per category), averaging the two repetitions and, for pooled
comparisons, the three models. Paired two-sided Wilcoxon signed-rank tests with Bonferroni
correction over the pre-declared family of comparisons; r = |Z|/sqrt(N); 95% bootstrap CIs of
the mean paired difference (10,000 resamples of questions)."""
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
OUT_T, OUT_F = HERE / "tables", HERE / "figs"
OUT_T.mkdir(exist_ok=True); OUT_F.mkdir(exist_ok=True)
RNG = np.random.default_rng(2026)

CONDS = ["FT", "PAGE_TEXT", "PAGE_FULL", "SUB_TEXT", "SUB_OCRCAP", "SUB_FULL"]
LABEL = {"FT": "Full text", "PAGE_TEXT": "Page, text only", "PAGE_FULL": "Page, enriched",
         "SUB_TEXT": "Sub-page, text only", "SUB_OCRCAP": "Sub-page, OCR+caption",
         "SUB_FULL": "Sub-page, enriched"}
MODELS = {"meta-llama/llama-3.1-70b-instruct": "Llama 3.1 70B", "deepseek/deepseek-chat": "DeepSeek-V3",
          "google/gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite"}
# (name, condition A, condition B, category); difference is A - B
FAMILY = [
    ("enrichment, page chunking", "PAGE_FULL", "PAGE_TEXT", "visual"),
    ("enrichment, sub-page chunking", "SUB_FULL", "SUB_TEXT", "visual"),
    ("NMM description over OCR+caption", "SUB_FULL", "SUB_OCRCAP", "visual"),
    ("sub-page vs page chunking", "SUB_FULL", "PAGE_FULL", "text"),
    ("original RAG vs full text", "PAGE_FULL", "FT", "text"),
    ("revised RAG vs full text", "SUB_FULL", "FT", "text"),
    ("revised RAG vs full text", "SUB_FULL", "FT", "visual"),
]
GOLD_IMAGE = {11: 2, 17: 2, 18: 2, 19: 2, 20: 2, 12: 11, 13: 12, 14: 12, 15: 13, 16: 13}


def wilcoxon(a, b):
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    if np.allclose(d, 0):
        return dict(n=n, mean_diff=0.0, ci_low=0.0, ci_high=0.0, W=np.nan, p=1.0, r=0.0)
    res = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", method="approx")
    z = abs(stats.norm.ppf(res.pvalue / 2))
    boots = RNG.choice(d, size=(10000, n), replace=True).mean(1)
    return dict(n=n, mean_diff=d.mean(), ci_low=np.percentile(boots, 2.5), ci_high=np.percentile(boots, 97.5),
                W=res.statistic, p=res.pvalue, r=z / np.sqrt(n))


def main():
    gen = pd.read_csv(HERE / "generations.csv")
    sc = pd.read_csv(HERE / "scores.csv").drop_duplicates(["condition", "model", "id", "iteration"], keep="last")
    n_err = int(sc.score_correctness.isna().sum())
    sc = sc[sc.score_correctness.notna()]
    df = gen.merge(sc[["condition", "model", "id", "iteration", "score_correctness"]],
                   on=["condition", "model", "id", "iteration"], how="left")
    res = {"n_generations": len(gen), "n_scored": int(df.score_correctness.notna().sum()), "n_judge_errors": n_err}

    # Correctness tables
    cm = df.groupby(["condition", "model", "category"]).score_correctness.mean().unstack("category")
    cm["overall"] = df.groupby(["condition", "model"]).score_correctness.mean()
    cm = cm.reindex(CONDS, level=0).round(3)
    cm.to_csv(OUT_T / "correctness_by_condition_model.csv")
    pooled = df.groupby(["condition", "category"]).score_correctness.mean().unstack().reindex(CONDS)
    pooled["overall"] = df.groupby("condition").score_correctness.mean()
    pooled = pooled.round(3)
    pooled.to_csv(OUT_T / "correctness_pooled.csv")
    res["pooled"] = pooled.to_dict(orient="index")
    res["by_model"] = {f"{c}|{MODELS[m]}": v for (c, m), v in cm.to_dict(orient="index").items()}

    # Per-question means (pooled over models and repetitions), and per model
    q = df.groupby(["condition", "id", "category"]).score_correctness.mean().reset_index()
    qm = df.groupby(["condition", "model", "id", "category"]).score_correctness.mean().reset_index()
    tests = []
    for name, a, b, cat in FAMILY:
        A = q[(q.condition == a) & (q.category == cat)].set_index("id").score_correctness
        B = q[(q.condition == b) & (q.category == cat)].set_index("id").score_correctness.reindex(A.index)
        t = wilcoxon(A.values, B.values)
        t.update(comparison=name, a=a, b=b, category=cat, mean_a=A.mean(), mean_b=B.mean())
        t["p_bonf"] = min(1.0, t["p"] * len(FAMILY))
        tests.append(t)
        for m in MODELS:
            Am = qm[(qm.condition == a) & (qm.model == m) & (qm.category == cat)].set_index("id").score_correctness
            Bm = qm[(qm.condition == b) & (qm.model == m) & (qm.category == cat)].set_index("id").score_correctness.reindex(Am.index)
            tm = wilcoxon(Am.values, Bm.values)
            tm.update(comparison=f"{name} [{MODELS[m]}]", a=a, b=b, category=cat, mean_a=Am.mean(), mean_b=Bm.mean(),
                      p_bonf=np.nan)
            tests.append(tm)
    tt = pd.DataFrame(tests)[["comparison", "category", "a", "b", "mean_a", "mean_b", "mean_diff", "ci_low", "ci_high",
                               "W", "p", "p_bonf", "r", "n"]]
    tt.round(4).to_csv(OUT_T / "paired_tests.csv", index=False)
    res["tests"] = tt.round(4).to_dict(orient="records")

    # Supporting: N=30 (question x model) pairs, repetitions averaged
    sup = []
    for name, a, b, cat in FAMILY:
        A = qm[(qm.condition == a) & (qm.category == cat)].set_index(["model", "id"]).score_correctness
        B = qm[(qm.condition == b) & (qm.category == cat)].set_index(["model", "id"]).score_correctness.reindex(A.index)
        t = wilcoxon(A.values, B.values)
        t.update(comparison=name, category=cat)
        sup.append(t)
    pd.DataFrame(sup).round(4).to_csv(OUT_T / "paired_tests_n30.csv", index=False)
    res["tests_n30"] = pd.DataFrame(sup).round(4).to_dict(orient="records")

    # Repetition agreement (stability of generation + judge)
    w = df.pivot_table(index=["condition", "model", "id"], columns="iteration", values="score_correctness")
    res["rep_agreement"] = dict(spearman=stats.spearmanr(w[1], w[2], nan_policy="omit").statistic,
                                mad=float((w[1] - w[2]).abs().mean()))

    # Citations: presence of a source line, and whether a cited page is the evidence page
    ev = pd.read_csv(HERE / "evidence_map.csv").set_index("id")
    recs = json.loads((HERE / "records.json").read_text())
    img_page = {im["img_doc_index"]: im["page"] for im in recs["images"]}
    gold_page = {i: int(ev.loc[i, "pdf_page"]) for i in range(1, 11)}
    gold_page.update({i: img_page[GOLD_IMAGE[i]] for i in GOLD_IMAGE})
    page_re = re.compile(r"(?:p[aá]g(?:e|ina|\.)?|page)\s*(\d+)", re.I)

    def cites(row):
        txt = str(row.actual_output)
        has = bool(re.search(r"source|fonte", txt, re.I))
        pages = {int(x) for x in page_re.findall(txt)}
        return pd.Series(dict(has_source=has, correct_page=gold_page[row.id] in pages))
    cdf = pd.concat([df, df.apply(cites, axis=1)], axis=1)
    cit = cdf.groupby(["condition", "category"])[["has_source", "correct_page"]].mean().unstack().reindex(CONDS).round(3)
    cit.to_csv(OUT_T / "citations.csv")
    res["citations"] = {c: {f"{a}|{b}": v for (a, b), v in row.items()} for c, row in cit.to_dict(orient="index").items()}

    # Cost and tokens per answer
    cost = df.groupby(["condition", "model"]).agg(prompt_tokens=("prompt_tokens", "mean"),
                                                   completion_tokens=("completion_tokens", "mean"),
                                                   cost_per_answer=("cost", "mean"), total_cost=("cost", "sum"))
    cost = cost.reindex(CONDS, level=0)
    cost.round(6).to_csv(OUT_T / "cost_tokens.csv")
    cp = df.groupby("condition").agg(prompt_tokens=("prompt_tokens", "mean"), cost_per_answer=("cost", "mean"),
                                     total_cost=("cost", "sum")).reindex(CONDS)
    cp["cost_vs_FT_pct"] = 100 * (cp.cost_per_answer / cp.loc["FT", "cost_per_answer"] - 1)
    cp["tokens_vs_FT_pct"] = 100 * (cp.prompt_tokens / cp.loc["FT", "prompt_tokens"] - 1)
    res["cost_pooled"] = cp.round(6).to_dict(orient="index")
    res["cost_by_model"] = {f"{c}|{MODELS[m]}": v for (c, m), v in cost.round(6).to_dict(orient="index").items()}
    res["generation_total_cost"] = float(gen.cost.sum())

    # Retrieval study and judge drift
    res["retrieval"] = pd.read_csv(HERE / "retrieval_study.csv").to_dict(orient="records")
    jc = pd.read_csv(HERE / "judge_check.csv")
    old, new = jc.columns[jc.columns.str.contains("score_correctness")][0], "rescore"
    res["judge_drift"] = dict(n=len(jc), spearman=stats.spearmanr(jc[old], jc[new]).statistic,
                              mean_old=jc[old].mean(), mean_new=jc[new].mean(),
                              mad=(jc[old] - jc[new]).abs().mean(), exact=(jc[old].round(2) == jc[new].round(2)).mean())

    (HERE / "analysis.json").write_text(json.dumps(res, indent=1, default=float))
    figures(df, q, cp)
    print(pooled.to_string()); print(); print(tt[tt.p_bonf.notna()].round(3).to_string()); print()
    print(cit.to_string()); print(); print(cp.round(5).to_string()); print(res["rep_agreement"], res["judge_drift"])


def figures(df, q, cp):
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    # Figure 1: mean correctness by condition and category, 95% bootstrap CI over questions
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9), sharey=True)
    for ax, cat in zip(axes, ["text", "visual"]):
        for i, c in enumerate(CONDS):
            v = q[(q.condition == c) & (q.category == cat)].score_correctness.values
            b = RNG.choice(v, size=(10000, len(v)), replace=True).mean(1)
            lo, hi = np.percentile(b, [2.5, 97.5])
            ax.errorbar(v.mean(), i, xerr=[[v.mean() - lo], [hi - v.mean()]], fmt="o", color="black",
                        mfc="white" if c == "FT" else "black", capsize=2, lw=0.8, ms=4)
        ax.set_title(f"{cat.capitalize()} questions (N=10)", fontsize=9)
        ax.set_xlim(0, 1); ax.set_xlabel("G-Eval correctness (0 to 1)")
        ax.grid(axis="x", lw=0.3, color="0.85")
    axes[0].set_yticks(range(len(CONDS))); axes[0].set_yticklabels([LABEL[c] for c in CONDS]); axes[0].invert_yaxis()
    fig.tight_layout(); fig.savefig(OUT_F / "ablation_correctness.pdf"); plt.close(fig)

    # Figure 2: per-model heatmap
    m = df.groupby(["model", "condition", "category"]).score_correctness.mean().unstack(["category", "condition"])
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 1.9), sharey=True)
    for ax, cat in zip(axes, ["text", "visual"]):
        mat = m[cat][CONDS].reindex(list(MODELS)).values
        ax.imshow(mat, cmap="Greys", vmin=0, vmax=1, aspect="auto")
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.55 else "black")
        ax.set_xticks(range(len(CONDS))); ax.set_xticklabels(["FT", "P-T", "P-E", "S-T", "S-OC", "S-E"], fontsize=7)
        ax.set_title(f"{cat.capitalize()} questions", fontsize=9)
        ax.spines[:].set_visible(False); ax.tick_params(length=0)
    axes[0].set_yticks(range(len(MODELS))); axes[0].set_yticklabels(list(MODELS.values()), fontsize=8)
    fig.tight_layout(); fig.savefig(OUT_F / "ablation_heatmap.pdf"); plt.close(fig)

    # Figure 3: cost per answer vs correctness, one panel per category
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    OFF = {"FT": (-6, 6, "right"), "PAGE_TEXT": (6, 6, "left"), "PAGE_FULL": (6, -7, "left"),
           "SUB_TEXT": (-6, -9, "right"), "SUB_OCRCAP": (-6, 0, "right"), "SUB_FULL": (-6, 7, "right")}
    SHORT = {"FT": "FT", "PAGE_TEXT": "P-T", "PAGE_FULL": "P-E", "SUB_TEXT": "S-T", "SUB_OCRCAP": "S-OC", "SUB_FULL": "S-E"}
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6), sharey=True)
    pc = df.groupby(["condition", "category"]).score_correctness.mean()
    for ax, cat in zip(axes, ["text", "visual"]):
        for c in CONDS:
            x, y = cp.loc[c, "cost_per_answer"] * 1000, pc[(c, cat)]
            ax.scatter(x, y, color="black", marker="s" if c == "FT" else "o",
                       facecolor="white" if c == "FT" else "black", s=22, zorder=3)
            dx, dy, ha = OFF[c]
            ax.annotate(SHORT[c], (x, y), textcoords="offset points", xytext=(dx, dy), fontsize=7, ha=ha, va="center")
        ax.set_xscale("log"); ax.set_xlim(0.2, 20)
        ax.xaxis.set_major_locator(FixedLocator([0.3, 1, 3, 10])); ax.xaxis.set_major_formatter(FixedFormatter(["0.3", "1", "3", "10"]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("Mean cost per answer (US\\$ $\\times 10^{-3}$, log)")
        ax.set_title(f"{cat.capitalize()} questions", fontsize=9)
        ax.set_ylim(0, 1); ax.grid(lw=0.3, color="0.85")
    axes[0].set_ylabel("Mean correctness")
    fig.tight_layout(); fig.savefig(OUT_F / "ablation_cost_accuracy.pdf"); plt.close(fig)


if __name__ == "__main__":
    main()
