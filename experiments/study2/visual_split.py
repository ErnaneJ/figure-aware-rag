"""Visual questions split by whether their answer is also stated in the prose of the paper.

evidence_map.csv finds the reference answers of questions 17, 18 and 20 in the page text with full
token overlap, and a reading of those pages confirms that the prose states them: digital recording
in the 1960s (pdf page 1), the proliferation of attributes in the 1980s (page 9) and the energy
crisis of the 1970s (page 7). The other seven visual questions need the figure. Reads Study 1 and
Study 2 scores; writes tables/visual_split.csv. No API calls."""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
PROSE = [17, 18, 20]
FIGURE = [11, 12, 13, 14, 15, 16, 19]
KEY = ["condition", "model", "id", "iteration"]

s1 = pd.read_csv(HERE.parent / "study1" / "scores.csv")
s1 = s1[s1.category == "visual"]
s1 = s1.assign(condition=np.where(s1.model.str.startswith("Papyrus"), "S1_PAPYRUS", "S1_BASELINE"))
gen = pd.read_csv(HERE / "generations.csv")
sc = pd.read_csv(HERE / "scores.csv").drop_duplicates(KEY, keep="last")
s2 = gen.merge(sc[KEY + ["score_correctness"]], on=KEY)
s2 = s2[(s2.category == "visual") & s2.score_correctness.notna()]

rows = []
for d in (s1, s2):
    for c, g in d.groupby("condition"):
        q = g.groupby("id").score_correctness.mean()
        rows.append({"condition": c, "prose_answerable": round(q[PROSE].mean(), 3),
                     "figure_only": round(q[FIGURE].mean(), 3), "all_visual": round(q.mean(), 3)})
out = pd.DataFrame(rows)
out.to_csv(HERE / "tables" / "visual_split.csv", index=False)
print(out.to_string(index=False))
