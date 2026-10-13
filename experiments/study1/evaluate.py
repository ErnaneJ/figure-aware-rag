"""Study 1 judge: G-Eval correctness for all 600 answers and faithfulness for the 300 Papyrus answers,
both with GPT-4.1 through OpenRouter (DeepEval 3.8.4). Writes scores.csv.

DeepEval reads OPENAI_API_KEY; set it to the OpenRouter key in .env."""
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from deepeval.metrics import GEval, FaithfulnessMetric
from deepeval.models import GPTModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parent.parent / ".env", override=True)

obj_model = GPTModel(
    model="openai/gpt-4.1",
    base_url="https://openrouter.ai/api/v1"
)

correctness_metric = GEval(
    name="Correctness",
    model=obj_model,
    criteria="Determine if the answer is factually correct and complete in relation to the Ground Truth. Rate it from 1 to 5.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

faithfulness_metric = FaithfulnessMetric(
    threshold=0.7,
    model=obj_model,
    include_reason=True
)

df_questions = pd.read_csv(HERE.parent / "questions.csv")
df_baseline = pd.read_csv(HERE / "answers_baseline.csv")
df_papyrus = pd.read_csv(HERE / "answers_papyrus.csv")

def evaluate_set(df_input, is_papyrus=False):
    df = df_input.copy()
    
    correctness_scores = []
    correctness_reasons = []
    faithfulness_scores = []
    faithfulness_reasons = []
    
    for idx, row in df.iterrows():
        gold = df_questions[df_questions['id'] == row['id']].iloc[0]
        
        print(f"[{'PAPYRUS' if is_papyrus else 'BASELINE'}] {row['model']} - question {row['id']}")
        
        context = [str(row['retrieval_context'])] if is_papyrus and pd.notna(row.get('retrieval_context')) else None

        test_case = LLMTestCase(
            input=str(gold['question']),
            actual_output=str(row['actual_output']),
            expected_output=str(gold['ground_truth']),
            retrieval_context=context
        )
        
        try:
            correctness_metric.measure(test_case)
            correctness_scores.append(correctness_metric.score)
            correctness_reasons.append(correctness_metric.reason)
        except Exception as e:
            print(f"Correctness error: {e}")
            correctness_scores.append(0)
            correctness_reasons.append(f"Error: {str(e)}")

        if is_papyrus:
            try:
                faithfulness_metric.measure(test_case)
                faithfulness_scores.append(faithfulness_metric.score)
                faithfulness_reasons.append(faithfulness_metric.reason)
            except Exception as e:
                print(f"Faithfulness error: {e}")
                faithfulness_scores.append(0)
                faithfulness_reasons.append(f"Error: {str(e)}")
        else:
            faithfulness_scores.append(None)
            faithfulness_reasons.append(None)

    df['score_correctness'] = correctness_scores
    df['reason_correctness'] = correctness_reasons
    df['score_faithfulness'] = faithfulness_scores
    df['reason_faithfulness'] = faithfulness_reasons
    
    return df

df_final = pd.concat([evaluate_set(df_baseline), evaluate_set(df_papyrus, is_papyrus=True)], ignore_index=True)
df_final = df_final.sort_values(by=["id", "model"])
df_final.to_csv(HERE / "scores.csv", index=False)
print(f"Wrote {HERE / 'scores.csv'} ({len(df_final)} rows)")
