import os
import pandas as pd
from dotenv import load_dotenv
from deepeval.metrics import GEval, FaithfulnessMetric
from deepeval.models import GPTModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

load_dotenv(override=True)

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

df_questions = pd.read_csv("./experiments/questions.csv")
df_concorrentes = pd.read_csv("./experiments/concorrentes_results.csv")
df_papyrus = pd.read_csv("./experiments/results_papyrus_multimodel.csv")

def evaluate_set(df_input, is_papyrus=False):
    df = df_input.copy()
    
    correctness_scores = []
    correctness_reasons = []
    faithfulness_scores = []
    faithfulness_reasons = []
    
    for idx, row in df.iterrows():
        gold = df_questions[df_questions['id'] == row['id']].iloc[0]
        
        print(f"[{'PAPYRUS' if is_papyrus else 'CONCORRENTE'}] Avaliando {row['model']} - Pergunta {row['id']}...")
        
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
            print(f"Erro ao medir Correctness: {e}")
            correctness_scores.append(0)
            correctness_reasons.append(f"Erro: {str(e)}")

        if is_papyrus:
            try:
                faithfulness_metric.measure(test_case)
                faithfulness_scores.append(faithfulness_metric.score)
                faithfulness_reasons.append(faithfulness_metric.reason)
            except Exception as e:
                print(f"Erro ao medir Faithfulness: {e}")
                faithfulness_scores.append(0)
                faithfulness_reasons.append(f"Erro: {str(e)}")
        else:
            faithfulness_scores.append(None)
            faithfulness_reasons.append(None)

    df['score_correctness'] = correctness_scores
    df['reason_correctness'] = correctness_reasons
    df['score_faithfulness'] = faithfulness_scores
    df['reason_faithfulness'] = faithfulness_reasons
    
    return df

print("--- Iniciando Avaliação dos Concorrentes ---")
df_concorrentes_scored = evaluate_set(df_concorrentes, is_papyrus=False)

print("\n--- Iniciando Avaliação do Papyrus ---")
df_papyrus_scored = evaluate_set(df_papyrus, is_papyrus=True)

df_final = pd.concat([df_concorrentes_scored, df_papyrus_scored], ignore_index=True)

df_final = df_final.sort_values(by=['id', 'model'])

df_final.to_csv("AVALIACAO_FINAL_MESTRADO.csv", index=False)

print("\n" + "="*50)
print("✅ AVALIAÇÃO CONCLUÍDA COM SUCESSO!")
print(f"Arquivo gerado: AVALIACAO_FINAL_MESTRADO.csv")
print(f"Total de registros: {len(df_final)}")
print("="*50)