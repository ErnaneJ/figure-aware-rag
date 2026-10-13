import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. SETUP
# ==========================================
sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
color_raw = "#E63946"     # Vermelho
color_papyrus = "#2A9D8F" # Verde/Azul

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
# EXTRA 1: LLM Generation Latency (Speed)
# ==========================================
plt.figure(figsize=(10, 6))
# Convertendo ms para segundos
time_raw = df_raw_act.groupby('Model')['generation_time_ms'].mean() / 1000
time_pap = df_pap_act.groupby('Model')['generation_time_ms'].mean() / 1000

df_time = pd.DataFrame({'Raw Context': time_raw, 'Papyrus AI': time_pap}).reset_index()
df_melt_time = df_time.melt(id_vars='Model', var_name='Architecture', value_name='Time (s)')

ax1 = sns.barplot(data=df_melt_time, x='Model', y='Time (s)', hue='Architecture', palette=[color_raw, color_papyrus])
plt.title('Average LLM Generation Latency', weight='bold')
plt.ylabel('Generation Time (Seconds)')
plt.xlabel('')
plt.xticks(rotation=15)
plt.legend(title='Architecture')
plt.tight_layout()
plt.savefig('figures/fig5_generation_latency.png', dpi=300)
plt.close()

# ==========================================
# EXTRA 2: Answer Conciseness (Length of Output)
# ==========================================
plt.figure(figsize=(10, 6))
# Calcula o tamanho da resposta em caracteres
df_eval['Answer Length'] = df_eval['actual_output'].astype(str).apply(len)

ax2 = sns.barplot(data=df_eval, x='Base Model', y='Answer Length', hue='Architecture', palette=[color_raw, color_papyrus])
plt.title('Answer Conciseness (Average Character Count)', weight='bold')
plt.ylabel('Number of Characters in Response')
plt.xlabel('')
plt.xticks(rotation=15)
plt.legend(title='Architecture')
plt.tight_layout()
plt.savefig('figures/fig6_answer_conciseness.png', dpi=300)
plt.close()

# ==========================================
# EXTRA 3: Consistency & Variance (Boxplot)
# ==========================================
plt.figure(figsize=(12, 6))
ax3 = sns.boxplot(data=df_eval, x='Base Model', y='Correctness', hue='Architecture', palette=[color_raw, color_papyrus])
plt.title('Correctness Distribution and Consistency', weight='bold')
plt.ylabel('Correctness Score (1-5)')
plt.xlabel('')
plt.ylim(0.5, 5.5)
plt.xticks(rotation=15)
plt.legend(title='Architecture', loc='lower right')
plt.tight_layout()
plt.savefig('figures/fig7_correctness_variance.png', dpi=300)
plt.close()

# ==========================================
# EXTRA 4: Text Category Stability
# ==========================================
plt.figure(figsize=(10, 6))
df_text = df_eval[df_eval['category'] == 'text']

ax4 = sns.barplot(data=df_text, x='Base Model', y='Correctness', hue='Architecture', palette=[color_raw, color_papyrus], errorbar=None)
plt.title('Text Category Correctness: Maintaining Stability', weight='bold')
plt.ylabel('Correctness Score (1-5)')
plt.xlabel('Model')
plt.ylim(1, 5)
plt.xticks(rotation=15)
plt.legend(title='Architecture')
plt.tight_layout()
plt.savefig('figures/fig8_text_stability.png', dpi=300)
plt.close()

print("✅ Mais 4 gráficos complementares gerados com sucesso na pasta 'figures/'!")