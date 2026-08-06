import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import shap

# Configuração da pagina Streamlit
st.set_page_config(page_title="SOC Shield - Detecção de Intrusões", layout="wide")

# Carregamento do modelo e dos dados
@st.cache_resource
def carregar_modelo():
    return joblib.load(os.path.join("app", "modelo_final.pkl"))

@st.cache_data
def carregar_dados():
    return pd.read_csv(os.path.join("data", "processed", "test_processed.csv"))

modelo = carregar_modelo()
df_teste = carregar_dados()

X_test = df_teste.drop(columns=["target"])
y_test = df_teste["target"]

@st.cache_resource
def criar_explainer():
    return shap.TreeExplainer(modelo)

explainer = criar_explainer()

st.title("Sistema de Detecção de Intrusões")
st.write("Aplicação para ajudar a equipe de segurança (SOC) a identificar conexões maliciosas em redes de computadores.")

st.warning("Limitacoes do Modelo: Este sistema funciona como um auxiliar de triagem e nao substitui o analista humano. O modelo pode ter menor eficiencia contra ataques ineditos nao presentes no treino.")

st.divider()

# Visão 1: Panorama geral
st.header("1. Panorama Geral de Desempenho")
st.write("Visão dos resultados do modelo na base de teste.")

c1, c2, c3 = st.columns(3)
c1.metric(label="Taxa de Falsos Alarmes (FPR)", value="2,72%", help="Meta: menor que 5%")
c2.metric(label="Desempenho no Teste (F1-Score)", value="75,33%")
c3.metric(label="Total de Conexões Analisadas", value=f"{len(df_teste):,}")

st.subheader("Distribuição do Tráfego na Base de Teste")
fig1, ax1 = plt.subplots(figsize=(6, 2.5))
contagem = y_test.value_counts()
ax1.bar(["Normal", "Ataque"], [contagem[0], contagem[1]], color=["#2ecc71", "#e74c3c"])
ax1.set_ylabel("Quantidade de Conexões")
plt.tight_layout()
st.pyplot(fig1)

st.divider()

# Visão 2: Predição individual
st.header("2. Análise de Conexão Individual (SHAP Local)")
st.write("Escolha um registro para verificar a predição e o motivo da decisão.")

idx = st.number_input("Selecione o número da conexão (0 a 22543):", min_value=0, max_value=len(X_test)-1, value=42)

amostra = X_test.iloc[[idx]]
y_real = y_test.iloc[idx]

pred = modelo.predict(amostra)[0]
prob = modelo.predict_proba(amostra)[0][1]

col_res, col_graf = st.columns([1, 2])

with col_res:
    st.subheader("Resultado do Modelo:")
    if pred == 1:
        st.error("ALERTA: ATAQUE DETECTADO")
    else:
        st.success("TRÁFEGO NORMAL")
        
    st.write(f"Probabilidade de Ataque: {prob*100:.1f}%")
    st.write(f"Classe Real: {'Ataque' if y_real == 1 else 'Normal'}")

with col_graf:
    st.subheader("Impacto dos Atributos na Decisão")
    
    shap_vals = explainer.shap_values(amostra)
    if isinstance(shap_vals, list):
        vals = shap_vals[1][0]
    elif len(shap_vals.shape) == 3:
        vals = shap_vals[0, :, 1]
    else:
        vals = shap_vals[0]
        
    df_shap = pd.DataFrame({"Atributo": X_test.columns, "Impacto": vals})
    df_shap["Impacto_Abs"] = df_shap["Impacto"].abs()
    top_5 = df_shap.sort_values(by="Impacto_Abs", ascending=False).head(5)
    
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    cores = ["#e74c3c" if v > 0 else "#2ecc71" for v in top_5["Impacto"]]
    ax2.barh(top_5["Atributo"], top_5["Impacto"], color=cores)
    ax2.axvline(0, color="gray", linestyle="--")
    ax2.set_xlabel("Impacto na Decisão (Vermelho = Aumenta Suspeita | Verde = Indica Normal)")
    ax2.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig2)

st.divider()

# Visão 3: Importância global
st.header("3. Interpretabilidade Global (SHAP Global)")
st.write("Atributos de rede que mais influenciam as decisões do modelo no geral.")

amostra_g = X_test.sample(100, random_state=42)
shap_g = explainer.shap_values(amostra_g)
if isinstance(shap_g, list):
    vals_g = shap_g[1]
elif len(shap_g.shape) == 3:
    vals_g = shap_g[:, :, 1]
else:
    vals_g = shap_g

imp = np.abs(vals_g).mean(axis=0)
df_g = pd.DataFrame({"Atributo": X_test.columns, "Importancia": imp}).sort_values(by="Importancia", ascending=False).head(8)

fig3, ax3 = plt.subplots(figsize=(7, 3.5))
ax3.barh(df_g["Atributo"], df_g["Importancia"], color="#3498db")
ax3.set_xlabel("Importância Média (SHAP Value)")
ax3.invert_yaxis()
plt.tight_layout()
st.pyplot(fig3)

st.info("Legenda das principais variáveis: same_srv_rate é a porcentagem de conexões ao mesmo serviço; log_src_bytes é o volume de dados enviados; count é o número de conexões nos últimos 2 segundos.")
