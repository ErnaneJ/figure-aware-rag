"""Robustness check of the judge: Study 2 rescored with the question and a single scale.

judge.py --with-question gives GPT-4.1 the question as well as the answer and the reference, and
uses evaluation steps (geval_steps_q.json) that name only the 0 to 10 scale of the DeepEval
template. It wrote scores_q.csv (original run) and scores_layout_q.csv (caption rerun).
This script recomputes the planned family of analyze.py with those scores, for the original run
and for the run with layout captions, compares the two judge configurations on the same answers,
and checks how much each one rewards long answers. Writes tables/rescore.json. No API calls."""
import json
import numpy as np
import pandas as pd
from scipy import stats
import analyze
from analyze import FAMILY, HERE, wilcoxon

KEY = ["condition", "model", "id", "iteration"]
RERUN = ["PAGE_FULL", "SUB_OCRCAP", "SUB_FULL"]


def scored(gen, sc, col):
    s = pd.read_csv(HERE / sc).drop_duplicates(KEY, keep="last")
    s = s[s.score_correctness.notna()].rename(columns={"score_correctness": col})
    return pd.read_csv(HERE / gen).merge(s[KEY + [col]], on=KEY, how="inner")


def family(df, col):
    analyze.RNG = np.random.default_rng(2026)
    q = df.groupby(["condition", "category", "id"])[col].mean()
    out = []
    for name, a, b, cat in FAMILY:
        A, B = q[a, cat], q[b, cat]
        t = wilcoxon(A.values, B.reindex(A.index).values)
        x = np.round(A.values, 6) - np.round(B.reindex(A.index).values, 6)
        lo, hi = stats.t.interval(0.95, len(x) - 1, loc=x.mean(), scale=stats.sem(x))
        out.append({"comparison": name, "category": cat, "mean_a": round(A.mean(), 3), "mean_b": round(B.mean(), 3),
                    "mean_diff": round(t["mean_diff"], 3), "ci_low": round(t["ci_low"], 3), "ci_high": round(t["ci_high"], 3),
                    "t_low": round(lo, 3), "t_high": round(hi, 3), "p": round(t["p"], 4),
                    "p_bonf": round(min(1.0, t["p"] * len(FAMILY)), 4), "r": round(t["r"], 3), "n_eff": t["n_eff"]})
    return out


def main():
    old = scored("generations.csv", "scores.csv", "s_old")
    new = scored("generations.csv", "scores_q.csv", "s_new")
    both = old.merge(new[KEY + ["s_new"]], on=KEY)
    res = {"n_rescored": len(new), "n_paired": len(both)}

    # Agreement between the two judge configurations on the same answers
    both["len"] = both.actual_output.astype(str).str.len()
    res["agreement"] = {"spearman": round(stats.spearmanr(both.s_old, both.s_new).statistic, 3),
                        "mean_old": round(both.s_old.mean(), 3), "mean_new": round(both.s_new.mean(), 3),
                        "mean_abs_diff": round((both.s_old - both.s_new).abs().mean(), 3),
                        "identical": round((both.s_old.round(3) == both.s_new.round(3)).mean(), 3)}
    res["shift_by_condition"] = (both.groupby(["condition", "category"])[["s_old", "s_new"]].mean()
                                 .round(3).reset_index().to_dict("records"))
    # Verbosity: rank correlation of score with answer length, within question, for each judge
    res["length_correlation"] = {c: round(both.groupby("id").apply(
        lambda g: stats.spearmanr(g.len, g[c]).statistic, include_groups=False).mean(), 3) for c in ("s_old", "s_new")}
    # Score scale actually used by the new judge
    res["new_score_values"] = {str(k): int(v) for k, v in new.s_new.round(2).value_counts().sort_index().items()}

    res["family_original_run"] = family(new, "s_new")
    lay = HERE / "scores_layout_q.csv"
    if lay.exists():
        newl = scored("generations_layout.csv", "scores_layout_q.csv", "s_new")
        mixed = pd.concat([new[~new.condition.isin(RERUN)], newl])
        res["n_rescored_layout"] = len(newl)
        res["family_layout_run"] = family(mixed, "s_new")
    (HERE / "tables" / "rescore.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
