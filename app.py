import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from shap import TreeExplainer, summary_plot
import os

# Configuração da página
st.set_page_config(page_title="Dashboard Varejo - 7 Perguntas", layout="wide")
st.title("📊 Dashboard de Comportamento de Compra – 7 Perguntas de Negócio")
st.markdown("Análise interativa do dataset *Customer Shopping Behavior*.")

# Caminho seguro para o CSV
CSV_PATH = os.path.join(os.path.dirname(__file__), "shopping_behavior_updated.csv")

@st.cache_data
def load_data():
    if not os.path.exists(CSV_PATH):
        st.error(f"❌ Arquivo não encontrado: {CSV_PATH}")
        st.stop()
    return pd.read_csv(CSV_PATH)

df = load_data()

# Sidebar com filtros obrigatórios
with st.sidebar:
    st.header("🎛️ Filtros Obrigatórios")
    categorias = st.multiselect("Categoria", sorted(df["Category"].unique()), default=df["Category"].unique())
    generos = st.multiselect("Gênero", sorted(df["Gender"].unique()), default=df["Gender"].unique())
    faixa_etaria = st.slider("Idade", int(df["Age"].min()), int(df["Age"].max()), (18, 70))
    aplicar = st.button("🔍 Aplicar Filtros")

# Validação de filtros
if not aplicar:
    st.warning("⚠️ Por favor, selecione os filtros desejados e clique em **'Aplicar Filtros'** para visualizar as respostas.")
    st.stop()

df_filtrado = df[
    (df["Category"].isin(categorias)) &
    (df["Gender"].isin(generos)) &
    (df["Age"].between(faixa_etaria[0], faixa_etaria[1]))
]

# ==========================
# 🎯 Respostas Automáticas às 7 Perguntas de Negócio
# ==========================
st.markdown("---")
st.header("🎯 Respostas Automáticas às 7 Perguntas de Negócio")

# Pergunta 1
with st.expander("1️⃣ Probabilidade de um cliente ser Big Spender"):
    bs = df[df["Purchase Amount (USD)"] > df["Purchase Amount (USD)"].quantile(0.8)]
    prob = len(bs) / len(df)
    st.metric("Probabilidade (percentil 80)", f"{prob:.1%}")
    st.write("Base: todo o dataset (sem filtros do usuário)")

# Pergunta 2
with st.expander("2️⃣ Segmentos naturais de consumidores"):
    X = df[["Age", "Purchase Amount (USD)"]].dropna()
    kmeans = KMeans(n_clusters=3, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X)
    st.write("Clusters baseados em Idade × Valor de Compra")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="Age", y="Purchase Amount (USD)", hue="Cluster", palette="Set2", ax=ax)
    st.pyplot(fig)

# Pergunta 3
with st.expander("3️⃣ Estações e locais com vendas mais intensas"):
    season_state = df.groupby(["Season", "Location"])["Purchase Amount (USD)"].sum().reset_index()
    top_season_state = season_state.sort_values("Purchase Amount (USD)", ascending=False).head(10)
    st.write("Top 10 combinações Estação × Localização")
    st.dataframe(top_season_state)

# Pergunta 4
with st.expander("4️⃣ Categorias que geram maior valor médio"):
    cat_avg = df.groupby("Category")["Purchase Amount (USD)"].mean().sort_values(ascending=False)
    st.bar_chart(cat_avg)

# Pergunta 5
with st.expander("5️⃣ Persona ideal para campanhas de alto valor"):
    top_10 = df["Purchase Amount (USD)"].quantile(0.9)
    persona = df[df["Purchase Amount (USD)"] > top_10]
    st.write("Perfil dos 10% maiores gastadores")
    st.dataframe(persona[["Age", "Gender", "Category", "Season", "Purchase Amount (USD)"]].describe())

# Pergunta 6
with st.expander("6️⃣ Relação entre características do cliente e valor gasto"):
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="Age", y="Purchase Amount (USD)", hue="Category", alpha=0.7, ax=ax)
    st.pyplot(fig)

# Pergunta 7
with st.expander("7️⃣ Previsão dos 20% maiores gastadores (LightGBM + SHAP)"):
    df_model = df.copy()
    df_model["BigSpender"] = (df_model["Purchase Amount (USD)"] > df_model["Purchase Amount (USD)"].quantile(0.8)).astype(int)
    X = pd.get_dummies(df_model[["Age", "Gender", "Category", "Season"]], drop_first=True)
    y = df_model["BigSpender"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)

    explainer = TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    st.write("Importância das features (SHAP)")
    summary_plot(shap_values[1], X_test, plot_type="bar", show=False)
    st.pyplot(plt.gcf())

# Rodapé
st.markdown("---")
st.markdown("Desenvolvido com ❤️ usando Streamlit | Dataset: Customer Shopping Behavior")
