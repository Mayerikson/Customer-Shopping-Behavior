import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(page_title="Dashboard Varejo", layout="wide")
st.title("📊 Dashboard de Comportamento de Compra")
st.markdown("Análise interativa do dataset *Customer Shopping Behavior*")

# Carregar dados
@st.cache_data
def load_data():
    return pd.read_csv("shopping_behavior_updated.csv")

df = load_data()

# Sidebar com filtros
st.sidebar.header("Filtros")
categorias = st.sidebar.multiselect("Categoria", sorted(df["Category"].unique()), default=df["Category"].unique())
generos = st.sidebar.multiselect("Gênero", sorted(df["Gender"].unique()), default=df["Gender"].unique())
faixa_etaria = st.sidebar.slider("Idade", int(df["Age"].min()), int(df["Age"].max()), (18, 70))

# Aplicar filtros
df_filtrado = df[
    (df["Category"].isin(categorias)) &
    (df["Gender"].isin(generos)) &
    (df["Age"].between(faixa_etaria[0], faixa_etaria[1]))
]

# Métricas principais
st.subheader("📈 Métricas Gerais")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total de Compras", len(df_filtrado))
col2.metric("Valor Total (USD)", f"${df_filtrado['Purchase Amount (USD)'].sum():,.2f}")
col3.metric("Ticket Médio", f"${df_filtrado['Purchase Amount (USD)'].mean():,.2f}")
col4.metric("Top Categoria", df_filtrado["Category"].mode()[0])

# Gráficos
st.subheader("📊 Visualizações")
col1, col2 = st.columns(2)

with col1:
    st.write("**Valor de Compra por Categoria**")
    fig, ax = plt.subplots()
    sns.boxplot(data=df_filtrado, x="Category", y="Purchase Amount (USD)", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

with col2:
    st.write("**Distribuição de Compras por Gênero**")
    fig, ax = plt.subplots()
    df_filtrado["Gender"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
    ax.set_ylabel("")
    st.pyplot(fig)

# Top 10 estados
st.subheader("🗺️ Top 10 Estados por Número de Compras")
top_estados = df_filtrado["Location"].value_counts().head(10)
st.bar_chart(top_estados)

# Tabela interativa
st.subheader("🧑‍💻 Visualizar Dados Filtrados")
st.dataframe(df_filtrado)

# Insights
st.subheader("💡 Insights Rápidos")
st.write("- Pessoas acima de 50 anos tendem a gastar mais em **Outerwear**.")
st.write("- **Footwear** tem maior valor médio de compra.")
st.write("- **California** e **New York** lideram em número de compras.")
