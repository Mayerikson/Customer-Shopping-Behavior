import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard Varejo - 7 Perguntas", layout="wide")
st.title("📊 Dashboard de Comportamento de Compra – 7 Perguntas de Negócio")
st.markdown("Respostas interativas com base no dataset *Customer Shopping Behavior*.")

@st.cache_data
def load_data():
    return pd.read_csv("shopping_behavior_updated.csv")

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

# Função auxiliar para exibir gráficos
def plot_box(data, x, y, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=data, x=x, y=y, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

# Pergunta 1: Probabilidade de ser Big Spender
st.header("1️⃣ Probabilidade de ser Big Spender")
big_spenders = df_filtrado[df_filtrado["Purchase Amount (USD)"] > df_filtrado["Purchase Amount (USD)"].quantile(0.8)]
prob = len(big_spenders) / len(df_filtrado)
st.metric("Probabilidade estimada", f"{prob:.1%}")
st.write("Baseado no percentil 80 do valor de compra.")

# Pergunta 2: Segmentos de Clientes
st.header("2️⃣ Segmentos de Clientes (KMeans)")
from sklearn.cluster import KMeans
X = df_filtrado[["Age", "Purchase Amount (USD)"]].dropna()
kmeans = KMeans(n_clusters=3, random_state=42)
df_filtrado = df_filtrado.copy()
df_filtrado["Cluster"] = kmeans.fit_predict(X)
st.write("Clusters baseados em idade e valor de compra:")
plot_box(df_filtrado, "Cluster", "Purchase Amount (USD)", "Valor por Cluster")

# Pergunta 3: Estações com mais vendas
st.header("3️⃣ Estações com Mais Vendas")
vendas_por_estacao = df_filtrado.groupby("Season")["Purchase Amount (USD)"].sum()
st.bar_chart(vendas_por_estacao)

# Pergunta 4: Categorias com maior valor médio
st.header("4️⃣ Categorias com Maior Valor Médio")
valor_medio = df_filtrado.groupby("Category")["Purchase Amount (USD)"].mean().sort_values(ascending=False)
st.bar_chart(valor_medio)

# Pergunta 5: Persona Ideal para Campanhas
st.header("5️⃣ Persona Ideal para Campanhas de Alto Valor")
persona = df_filtrado[df_filtrado["Purchase Amount (USD)"] > df_filtrado["Purchase Amount (USD)"].quantile(0.9)]
st.write("Perfil dos 10% maiores gastadores:")
st.write(persona[["Age", "Gender", "Category", "Season", "Purchase Amount (USD)"]].describe())

# Pergunta 6: Relação entre Idade e Valor Gasto
st.header("6️⃣ Relação entre Idade e Valor Gasto")
fig, ax = plt.subplots()
sns.scatterplot(data=df_filtrado, x="Age", y="Purchase Amount (USD)", hue="Category", ax=ax)
st.pyplot(fig)

# Pergunta 7: Previsão de Big Spenders com LightGBM
st.header("7️⃣ Previsão de Big Spenders (LightGBM + SHAP)")
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from shap import explainer, summary_plot

# Preparação
df_model = df_filtrado.copy()
df_model["BigSpender"] = (df_model["Purchase Amount (USD)"] > df_model["Purchase Amount (USD)"].quantile(0.8)).astype(int)
X = pd.get_dummies(df_model[["Age", "Gender", "Category", "Season"]], drop_first=True)
y = df_model["BigSpender"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelo
model = LGBMClassifier(random_state=42)
model.fit(X_train, y_train)

# SHAP
exp = explainer.TreeExplainer(model)
shap_values = exp.shap_values(X_test)

st.write("Importância das features para prever Big Spenders:")
summary_plot(shap_values[1], X_test, plot_type="bar", show=False)
st.pyplot(plt.gcf())

# Rodapé
st.markdown("---")
st.markdown("Desenvolvido com ❤️ usando Streamlit | Dataset: Customer Shopping Behavior")



# ==========================
# 🎯 ABA: Respostas Automáticas às 7 Perguntas
# ==========================
st.markdown("---")
st.header("🎯 Respostas Automáticas às 7 Perguntas de Negócio")

# Cada pergunta tem seu próprio filtro interno
def big_spenders_auto(df):
    return df[df["Purchase Amount (USD)"] > df["Purchase Amount (USD)"].quantile(0.8)]

def top_category_by_season(df):
    return df.groupby(["Season", "Category"])["Purchase Amount (USD)"].mean().reset_index()

def top_persona(df):
    top_10 = df["Purchase Amount (USD)"].quantile(0.9)
    return df[df["Purchase Amount (USD)"] > top_10]

# Pergunta 1
with st.expander("1️⃣ Probabilidade de um cliente ser Big Spender"):
    bs = big_spenders_auto(df)
    prob = len(bs) / len(df)
    st.metric("Probabilidade (percentil 80)", f"{prob:.1%}")
    st.write("Base: todo o dataset (sem filtros do usuário)")

# Pergunta 2
with st.expander("2️⃣ Segmentos naturais de consumidores"):
    from sklearn.cluster import KMeans
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
    persona = top_persona(df)
    st.write("Perfil dos 10% maiores gastadores")
    st.dataframe(persona[["Age", "Gender", "Category", "Season", "Purchase Amount (USD)"]].describe())

# Pergunta 6
with st.expander("6️⃣ Relação entre características do cliente e valor gasto"):
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="Age", y="Purchase Amount (USD)", hue="Category", alpha=0.7, ax=ax)
    st.pyplot(fig)

# Pergunta 7
with st.expander("7️⃣ Previsão dos 20% maiores gastadores (LightGBM + SHAP)"):
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import train_test_split
    from shap import TreeExplainer, summary_plot

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
