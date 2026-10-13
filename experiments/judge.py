from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
import pandas as pd

# Métrica de Corretude
correctness_metric = GEval(
    name="Correctness",
    criteria="A resposta está factualmente correta comparada ao Gabarito?",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT]
)

def run_evaluation():
    df_p = pd.read_csv("results_papyrus.csv")
    df_c = pd.read_csv("results_concorrentes.csv")
    df_gold = pd.read_csv("perguntas_ouro.csv")
    
    all_data = pd.concat([df_p, df_c])
    
    for _, row in all_data.iterrows():
        gold = df_gold[df_gold['id'] == row['question_id']].iloc[0]
        
        test_case = LLMTestCase(
            input=gold['question'],
            actual_output=row['answer'],
            expected_output=gold['ground_truth']
        )
        
        correctness_metric.measure(test_case)
        row['score'] = correctness_metric.score
        
    all_data.to_csv("METRICAS_FINAIS_ARTIGO.csv")