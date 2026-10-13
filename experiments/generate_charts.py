import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ==========================================
# 1. SETUP E CARREGAMENTO DE DADOS
# ==========================================
# Configuração visual para artigos científicos
sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
color_raw = "#E63946"     # Vermelho (Contexto Bruto - Caro/Ineficiente)
color_papyrus = "#2A9D8F" # Verde/Azul (Papyrus - Otimizado/Seguro)

os.makedirs("figures", exist_ok=True)

# Carrega os dados
df_eval = pd.read_csv("./experiments/AVALIACAO_FINAL_MESTRADO.csv")
df_raw_act = pd.read_csv("./experiments/models--openrouter_activity_2026-02-20.csv")
df_pap_act = pd.read_csv("./experiments/papyrus--openrouter_activity_2026-02-21.csv")

# ==========================================
# 2. LIMPEZA E PREPARAÇÃO DOS DADOS
# ==========================================
def clean_model_name(name):
    name = str(name).lower()
    if 'gpt-4o' in name: return 'GPT-4o'
    if 'claude-3.5-sonnet' in name: return 'Claude 3.5'
    if 'deepseek-chat' in name: return 'DeepSeek V3'
    if 'llama-3.1-70b' in name: return 'Llama 3.1 70B'
    if 'gemini' in name: return 'Gemini Flash'
    return name.split('/')[-1]

# Aplica limpeza
df_raw_act['Model'] = df_raw_act['model_permaslug'].apply(clean_model_name)
df_pap_act['Model'] = df_pap_act['model_permaslug'].apply(clean_model_name)

df_eval['Architecture'] = df_eval['model'].apply(lambda x: 'Papyrus AI' if 'Papyrus' in str(x) else 'Raw Context')
df_eval['Base Model'] = df_eval['model'].apply(clean_model_name)

# Normaliza Correctness para escala 1-5 (O DeepEval costuma dar notas 0 a 1 normalizadas)
if df_eval['score_correctness'].max() <= 1.0:
    df_eval['Correctness'] = df_eval['score_correctness'] * 4 + 1
else:
    df_eval['Correctness'] = df_eval['score_correctness']

# Agrupa custos e tokens
cost_raw = df_raw_act.groupby('Model')['cost_total'].sum()
cost_pap = df_pap_act.groupby('Model')['cost_total'].sum()
tokens_raw = df_raw_act.groupby('Model')['tokens_prompt'].sum()
tokens_pap = df_pap_act.groupby('Model')['tokens_prompt'].sum()

df_eff = pd.DataFrame({
    'Raw Cost ($)': cost_raw,
    'Papyrus Cost ($)': cost_pap,
    'Raw Tokens': tokens_raw,
    'Papyrus Tokens': tokens_pap
}).reset_index().fillna(0)


# ==========================================
# 3. GERAÇÃO DOS GRÁFICOS
# ==========================================

# ---------------------------------------------------------
# FIGURA 1: Redução de Custo (Escala Logarítmica para ver o Papyrus)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
df_melt_cost = df_eff.melt(id_vars='Model', value_vars=['Raw Cost ($)', 'Papyrus Cost ($)'], var_name='Approach', value_name='Cost (USD)')
df_melt_cost['Approach'] = df_melt_cost['Approach'].map({'Raw Cost ($)': 'Raw Context', 'Papyrus Cost ($)': 'Papyrus AI'})

ax1 = sns.barplot(data=df_melt_cost, x='Model', y='Cost (USD)', hue='Approach', palette=[color_raw, color_papyrus])
plt.yscale('log') # ESCALA LOGARÍTMICA: Fundamental para enxergar a barra do Papyrus comparada ao Claude/GPT
plt.title('API Cost Comparison (Log Scale)', weight='bold')
plt.ylabel('Total Cost in USD (Log Scale)')
plt.xlabel('')
plt.xticks(rotation=15)
plt.legend(title='Architecture')

# Adicionando os valores exatos em cima das barras para clareza
for p in ax1.patches:
    if p.get_height() > 0:
        ax1.annotate(f'${p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

plt.tight_layout()
plt.savefig('figures/fig1_cost_log_scale.png', dpi=300)
plt.close()

# ---------------------------------------------------------
# FIGURA 2: Redução de Tokens (Gráfico de Barras Agrupadas)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
df_melt_tok = df_eff.melt(id_vars='Model', value_vars=['Raw Tokens', 'Papyrus Tokens'], var_name='Approach', value_name='Tokens')
df_melt_tok['Approach'] = df_melt_tok['Approach'].map({'Raw Tokens': 'Raw Context', 'Papyrus Tokens': 'Papyrus AI'})

ax2 = sns.barplot(data=df_melt_tok, x='Model', y='Tokens', hue='Approach', palette=[color_raw, color_papyrus])
plt.title('Prompt Tokens Consumption', weight='bold')
plt.ylabel('Number of Tokens (Millions)')
plt.xlabel('')
plt.xticks(rotation=15)

# Formata o eixo Y para milhões
ticks_y = ax2.get_yticks()
ax2.set_yticklabels([f'{int(tick/1000000)}M' if tick >= 1000000 else f'{int(tick/1000)}k' for tick in ticks_y])
plt.legend(title='Architecture')
plt.tight_layout()
plt.savefig('figures/fig2_token_reduction.png', dpi=300)
plt.close()

# ---------------------------------------------------------
# FIGURA 3: Impacto do Papyrus em Modelos Menores (O Caso Llama 3) - Categoria Visual
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
df_visual = df_eval[df_eval['category'] == 'visual']

ax3 = sns.barplot(data=df_visual, x='Base Model', y='Correctness', hue='Architecture', 
                  palette=[color_papyrus, color_raw], errorbar=None)
plt.title('Visual Category Correctness: The Papyrus Impact', weight='bold')
plt.ylabel('Correctness Score (1-5)')
plt.xlabel('Model')
plt.ylim(1, 5)
plt.axhline(y=df_visual[df_visual['Architecture']=='Papyrus AI']['Correctness'].mean(), color=color_papyrus, linestyle='--', alpha=0.5, label='Papyrus Avg')
plt.xticks(rotation=15)
plt.legend(title='Architecture')

# Destaca o Llama 3.1 70B
llama_raw = df_visual[(df_visual['Base Model'] == 'Llama 3.1 70B') & (df_visual['Architecture'] == 'Raw Context')]['Correctness'].mean()
llama_pap = df_visual[(df_visual['Base Model'] == 'Llama 3.1 70B') & (df_visual['Architecture'] == 'Papyrus AI')]['Correctness'].mean()
if not np.isnan(llama_raw) and not np.isnan(llama_pap):
    gain = ((llama_pap - llama_raw) / llama_raw) * 100
    plt.annotate(f'+{gain:.0f}% Gain\nin Accuracy!', xy=(3.1, llama_pap), xytext=(3.5, 4.0),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=11, weight='bold', color='#1D3557', ha='center')

plt.tight_layout()
plt.savefig('figures/fig3_visual_llama_impact.png', dpi=300)
plt.close()

# ---------------------------------------------------------
# FIGURA 4: Fidelidade (Faithfulness) - Ausência de Alucinação
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))
df_papyrus_only = df_eval[df_eval['Architecture'] == 'Papyrus AI'].copy()
df_papyrus_only['Faithfulness (%)'] = df_papyrus_only['score_faithfulness'] * 100

ax4 = sns.barplot(data=df_papyrus_only, x='Base Model', y='Faithfulness (%)', color=color_papyrus, errorbar=('ci', 95), capsize=.1)
plt.title('Papyrus AI Faithfulness (Anti-Hallucination Metric)', weight='bold')
plt.ylabel('Faithfulness Score (%)')
plt.xlabel('')
plt.ylim(0, 110)
plt.axhline(y=100, color='gray', linestyle='--', alpha=0.7)
plt.xticks(rotation=15)

for p in ax4.patches:
    ax4.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., 100), 
                 ha='center', va='bottom', fontsize=11, xytext=(0, 5), textcoords='offset points', weight='bold')

plt.tight_layout()
plt.savefig('figures/fig4_faithfulness.png', dpi=300)
plt.close()

# ---------------------------------------------------------
# ANÁLISE TEXTUAL: Rastreabilidade (Citações de Fontes)
# ---------------------------------------------------------
papyrus_answers = df_eval[df_eval['Architecture'] == 'Papyrus AI']['actual_output'].dropna()
# Checa quantas vezes a string "Source:" ou o nome do arquivo "gsgpy" aparece
citations = papyrus_answers.str.contains(r'Source:|gsgpy|page', case=False, regex=True).sum()
total_papyrus = len(papyrus_answers)
citation_rate = (citations / total_papyrus) * 100

with open("figures/citations_report.txt", "w") as f:
    f.write(f"Papyrus Citations Rate: {citation_rate:.2f}% ({citations}/{total_papyrus} answers included explicit source references).\n")

print(f"✅ Gráficos gerados com sucesso na pasta 'figures/'!")
print(f"✅ Relatório de Citações: {citation_rate:.2f}% das respostas do Papyrus incluíram fontes explícitas.")