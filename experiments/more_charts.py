import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. SETUP E CARREGAMENTO
# ==========================================
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
color_raw = "#E63946"     # Vermelho (Antes / Bruto)
color_papyrus = "#2A9D8F" # Verde (Depois / Papyrus)

os.makedirs("figures", exist_ok=True)

df_eval = pd.read_csv("./experiments/AVALIACAO_FINAL_MESTRADO.csv")
df_raw_act = pd.read_csv("./experiments/models--openrouter_activity_2026-02-20.csv")
df_pap_act = pd.read_csv("./experiments/papyrus--openrouter_activity_2026-02-21.csv")

def clean_model_name(name):
    name = str(name).lower()
    if 'gpt-4o' in name: return 'GPT-4o'
    if 'claude-3.5-sonnet' in name: return 'Claude 3.5'
    if 'deepseek-chat' in name: return 'DeepSeek V3'
    if 'llama-3.1-70b' in name: return 'Llama 3.1 70B'
    if 'gemini' in name: return 'Gemini Flash'
    return name.split('/')[-1]

df_raw_act['Model'] = df_raw_act['model_permaslug'].apply(clean_model_name)
df_pap_act['Model'] = df_pap_act['model_permaslug'].apply(clean_model_name)
df_eval['Architecture'] = df_eval['model'].apply(lambda x: 'Papyrus AI' if 'Papyrus' in str(x) else 'Raw Context')
df_eval['Base Model'] = df_eval['model'].apply(clean_model_name)

if df_eval['score_correctness'].max() <= 1.0:
    df_eval['Correctness'] = df_eval['score_correctness'] * 4 + 1
else:
    df_eval['Correctness'] = df_eval['score_correctness']

# ==========================================
# 2. PREPARAÇÃO DOS DADOS (Foco no VISUAL)
# ==========================================
df_visual = df_eval[df_eval['category'] == 'visual']
perf_summary = df_visual.groupby(['Base Model', 'Architecture'])['Correctness'].mean().reset_index()

cost_raw = df_raw_act.groupby('Model')['cost_total'].sum()
cost_pap = df_pap_act.groupby('Model')['cost_total'].sum()

# Criar um DataFrame unificado para desenhar as setas
models = perf_summary['Base Model'].unique()
data = []

for m in models:
    acc_raw = perf_summary[(perf_summary['Base Model'] == m) & (perf_summary['Architecture'] == 'Raw Context')]['Correctness'].values[0]
    acc_pap = perf_summary[(perf_summary['Base Model'] == m) & (perf_summary['Architecture'] == 'Papyrus AI')]['Correctness'].values[0]
    
    c_raw = cost_raw.get(m, 0)
    c_pap = cost_pap.get(m, 0)
    
    data.append({'Model': m, 'Acc_Raw': acc_raw, 'Acc_Pap': acc_pap, 'Cost_Raw': c_raw, 'Cost_Pap': c_pap})

df_plot = pd.DataFrame(data)

# ==========================================
# 3. PLOTAGEM DA TRAJETÓRIA (BEFORE & AFTER)
# ==========================================
plt.figure(figsize=(12, 8))

# Define os limites do quadrante ideal
ideal_cost_limit = 1.4

# Desenha uma área sombreada para a "Zona do Papyrus" (Baixo custo)
plt.axvspan(0.011, ideal_cost_limit, color=color_papyrus, alpha=0.1, zorder=0, label='Papyrus Efficiency Zone')

# Para cada modelo, desenha a bolinha inicial, a final e a seta conectando
for idx, row in df_plot.iterrows():
    # Seta do Raw para o Papyrus
    plt.annotate('', 
                 xy=(row['Cost_Pap'], row['Acc_Pap']), # Destino (Ponta da seta)
                 xytext=(row['Cost_Raw'], row['Acc_Raw']), # Origem
                 arrowprops=dict(arrowstyle="->", color='gray', lw=2, shrinkA=8, shrinkB=8),
                 zorder=1)
    
    # Texto com o nome do modelo perto da bolinha do Papyrus
    plt.text(row['Cost_Raw'], row['Acc_Raw'] - 0.04, row['Model'], 
             fontsize=10, weight='bold', ha='center', color='#333333', zorder=4)
    
    if row['Model'] not in ['Llama 3.1 70B', 'Claude 3.5']:
        plt.text(row['Cost_Pap'], row['Acc_Pap'] - 0.04, f"Papyrus [{row['Model']}]", 
                fontsize=10, weight='bold', ha='center', color='#333333', zorder=4)

    if row['Model'] in ['Llama 3.1 70B', 'Claude 3.5']:
        plt.text(row['Cost_Pap'], row['Acc_Pap'] + 0.025, f"Papyrus [{row['Model']}]", 
                fontsize=10, weight='bold', ha='center', color='#333333', zorder=4)

# Desenha os pontos (Raw)
sns.scatterplot(x='Cost_Raw', y='Acc_Raw', data=df_plot, s=200, color=color_raw, label='Without Papyrus (Raw Context)', zorder=2, edgecolor='black')

# Desenha os pontos (Papyrus)
sns.scatterplot(x='Cost_Pap', y='Acc_Pap', data=df_plot, s=200, color=color_papyrus, label='With Papyrus AI', zorder=3, edgecolor='black', marker='s') # marker='s' faz ser um quadrado para diferenciar

plt.title('Before & After Papyrus AI: Cost vs. Visual Performance Trajectory', weight='bold', fontsize=16)
plt.ylabel('Correctness Score in Visual Queries (1-5)', weight='bold')
plt.xlabel('Total API Processing Cost (USD) - Log Scale', weight='bold')
plt.xscale('log')

# Customiza a legenda
plt.legend(loc='lower right', frameon=True, fontsize=11)
plt.grid(True, which="both", ls="--", alpha=0.4)

plt.tight_layout()
plt.savefig('figures/fig7_trajectory_cost_performance.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Gráfico de Trajetória Matador gerado com sucesso!")