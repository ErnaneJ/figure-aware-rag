import pandas as pd

# 1. Carrega os dados da avaliação
df_eval = pd.read_csv("./experiments/AVALIACAO_FINAL_MESTRADO.csv")

# 2. Função de limpeza com TODOS os seus modelos
def clean_model_name(name):
    name = str(name).lower()
    if 'gpt-4o' in name: return 'GPT-4o'
    if 'claude-3.5-sonnet' in name: return 'Claude 3.5 Sonnet'
    if 'deepseek-chat' in name: return 'DeepSeek V3'
    if 'llama-3.1-70b' in name: return 'Llama 3.1 70B'
    if 'gemini' in name: return 'Gemini Flash'
    return name.split('/')[-1]

# 3. Classifica a Arquitetura e o Modelo Base
df_eval['Architecture'] = df_eval['model'].apply(lambda x: 'With Framework' if 'Papyrus' in str(x) else 'Native Accuracy')
df_eval['Base Model'] = df_eval['model'].apply(clean_model_name)

# 4. Filtra apenas a categoria VISUAL
df_visual = df_eval[df_eval['category'] == 'visual']

# 5. Calcula a média e multiplica por 100
accuracy_table = df_visual.groupby(['Base Model', 'Architecture'])['score_correctness'].mean().unstack() * 100

print("\n" + "="*50)
print(" RESULTADOS REAIS DE TODOS OS MODELOS (Pronto para LaTeX)")
print("="*50 + "\n")

# Lista com TODOS os 5 modelos
todos_modelos = ['GPT-4o', 'Claude 3.5 Sonnet', 'Llama 3.1 70B', 'DeepSeek V3', 'Gemini Flash']

for model in todos_modelos:
    if model in accuracy_table.index:
        native_acc = accuracy_table.loc[model, 'Native Accuracy']
        papyrus_acc = accuracy_table.loc[model, 'With Framework']
        print(f"        {model} & {native_acc:.0f}\\% & {papyrus_acc:.0f}\\% \\\\ \\hline")
    else:
        print(f"        {model} & DADOS NÃO ENCONTRADOS \\\\ \\hline")

print("\n" + "="*50)