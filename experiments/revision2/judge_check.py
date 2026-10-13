"""Judge drift check: re-score a stratified sample of the original (Feb 2026) answers with the
fixed-step judge and compare against the stored scores."""
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import spearmanr, wilcoxon
from judge import score, steps, HERE

old = pd.read_csv(HERE.parent / "AVALIACAO_FINAL_MESTRADO.csv")
old = old[old.score_correctness.notna()]
old["cond"] = old.model.str.startswith("Papyrus").map({True: "papyrus", False: "baseline"})
sample = old.groupby(["cond", "category"]).sample(10, random_state=7)
gold = pd.read_csv(HERE.parent / "questions.csv").set_index("id")
ev = steps()
with ThreadPoolExecutor(4) as ex:
    res = list(ex.map(lambda r: score(r, gold.loc[r.id], ev), [r for _, r in sample.iterrows()]))
sample["rescore"] = [s for s, _ in res]
sample[["model", "id", "iteration", "cond", "category", "score_correctness", "rescore"]].to_csv(HERE / "judge_check.csv", index=False)
d = sample.dropna(subset=["rescore"])
rho, p = spearmanr(d.score_correctness, d.rescore)
print(f"n={len(d)} spearman={rho:.2f} (p={p:.1e}) mean_old={d.score_correctness.mean():.3f} mean_new={d.rescore.mean():.3f} "
      f"MAD={abs(d.score_correctness - d.rescore).mean():.3f} exact={(d.score_correctness.round(2) == d.rescore.round(2)).mean():.2f}")
print(d.groupby(["cond", "category"])[["score_correctness", "rescore"]].mean().round(3))
