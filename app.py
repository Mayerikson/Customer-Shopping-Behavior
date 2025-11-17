import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from shap import TreeExplainer, summary_plot
import os
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Dashboard Varejo - 7 Perguntas", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado para melhor aparência
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">📊 Dashboard de Comportamento de Compra</p>', unsafe_allow_html=True)
st.markdown("**Análise interativa do dataset Customer Shopping Behavior**")

# Caminho seguro para o CSV
CSV_PATH = os.path.join(os.path.dirname(__file__), "shopping_behavior_updated.csv")

@st.cache_data
def load_data():
    """Carrega e valida o dataset"""
    if not os.path.exists(CSV_PATH):
        st.error(f"❌ Arquivo não encontrado: {CSV_PATH}")
        st.stop()
    
    df = pd.read_csv(CSV_PATH)
    
    # Validação básica das colunas necessárias
    required_cols = ["Category", "Gender", "Age", "Purchase Amount (USD)", "Season", "Location"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"❌ Colunas ausentes no dataset: {', '.join(missing_cols)}")
        st.stop()
    
    return df

df = load_data()

# ==========================
# 📊 Estatísticas Gerais (antes dos filtros)
# ==========================
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total de Clientes", f"{len(df):,}")
with col2:
    st.metric("Ticket Médio", f"${df['Purchase Amount (USD)'].mean():.2f}")
with col3:
    st.metric("Valor Total", f"${df['Purchase Amount (USD)'].sum():,.2f}")
with col4:
    st.metric("Categorias", df["Category"].nunique())

# ==========================
# 🎛️ Sidebar com filtros
# ==========================
with st.sidebar:
    st.header("🎛️ Filtros de Análise")
    st.markdown("Ajuste os filtros abaixo para segmentar sua análise:")
    
    # Filtro de Categoria
    categorias = st.multiselect(
        "📦 Categoria de Produto",
        sorted(df["Category"].unique()),
        default=df["Category"].unique(),
        help="Selecione uma ou mais categorias"
    )
    
    # Filtro de Gênero
    generos = st.multiselect(
        "👤 Gênero",
        sorted(df["Gender"].unique()),
        default=df["Gender"].unique()
    )
    
    # Filtro de Idade
    faixa_etaria = st.slider(
        "🎂 Faixa Etária",
        int(df["Age"].min()),
        int(df["Age"].max()),
        (int(df["Age"].min()), int(df["Age"].max())),
        help="Ajuste o intervalo de idade"
    )
    
    # Filtro de Estação
    estacoes = st.multiselect(
        "🌦️ Estação do Ano",
        sorted(df["Season"].unique()),
        default=df["Season"].unique()
    )
    
    st.markdown("---")
    aplicar = st.button("🔍 Aplicar Filtros", type="primary", use_container_width=True)
    limpar = st.button("🔄 Limpar Filtros", use_container_width=True)

# Validação de filtros
if limpar:
    st.rerun()

if not aplicar:
    st.info("💡 **Dica:** Selecione os filtros desejados e clique em **'Aplicar Filtros'** para visualizar as análises.")
    st.stop()

# Aplicar filtros
df_filtrado = df[
    (df["Category"].isin(categorias)) &
    (df["Gender"].isin(generos)) &
    (df["Age"].between(faixa_etaria[0], faixa_etaria[1])) &
    (df["Season"].isin(estacoes))
].copy()

# Verificar se há dados após filtragem
if len(df_filtrado) == 0:
    st.error("⚠️ Nenhum dado encontrado com os filtros selecionados. Por favor, ajuste os filtros.")
    st.stop()

# Mostrar estatísticas dos dados filtrados
st.markdown("---")
st.subheader("📈 Visão Geral dos Dados Filtrados")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Clientes Filtrados", f"{len(df_filtrado):,}", 
              delta=f"{((len(df_filtrado)/len(df)-1)*100):.1f}%")
with col2:
    st.metric("Ticket Médio", f"${df_filtrado['Purchase Amount (USD)'].mean():.2f}",
              delta=f"${(df_filtrado['Purchase Amount (USD)'].mean() - df['Purchase Amount (USD)'].mean()):.2f}")
with col3:
    st.metric("Valor Total", f"${df_filtrado['Purchase Amount (USD)'].sum():,.2f}")
with col4:
    st.metric("Compra Máxima", f"${df_filtrado['Purchase Amount (USD)'].max():.2f}")

# ==========================
# 🎯 Respostas às 7 Perguntas de Negócio
# ==========================
st.markdown("---")
st.header("🎯 Respostas às 7 Perguntas de Negócio")

# Pergunta 1
with st.expander("1️⃣ Qual a probabilidade de um cliente ser Big Spender?", expanded=True):
    col1, col2 = st.columns([1, 2])
    
    with col1:
        percentil = st.slider("Percentil de Corte", 70, 95, 80, 5, key="percentil_1")
        threshold = df_filtrado["Purchase Amount (USD)"].quantile(percentil/100)
        
        bs = df_filtrado[df_filtrado["Purchase Amount (USD)"] > threshold]
        prob = len(bs) / len(df_filtrado) if len(df_filtrado) > 0 else 0
        
        st.metric(
            f"Probabilidade (Top {100-percentil}%)",
            f"{prob:.1%}",
            help=f"Clientes com compras acima de ${threshold:.2f}"
        )
        st.metric("Valor de Corte", f"${threshold:.2f}")
        st.metric("Big Spenders", f"{len(bs):,}")
    
    with col2:
        # Distribuição de valores
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(df_filtrado["Purchase Amount (USD)"], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: ${threshold:.2f}')
        ax.set_xlabel('Valor da Compra (USD)')
        ax.set_ylabel('Frequência')
        ax.set_title('Distribuição de Valores de Compra')
        ax.legend()
        st.pyplot(fig)
        plt.close()

# Pergunta 2
with st.expander("2️⃣ Quais são os segmentos naturais de consumidores?"):
    col1, col2 = st.columns([1, 2])
    
    with col1:
        n_clusters = st.slider("Número de Clusters", 2, 6, 3, key="clusters")
        
        X = df_filtrado[["Age", "Purchase Amount (USD)"]].dropna()
        
        if len(X) > 0:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df_filtrado["Cluster"] = kmeans.fit_predict(X)
            
            st.write("**Características dos Clusters:**")
            cluster_stats = df_filtrado.groupby("Cluster").agg({
                "Age": "mean",
                "Purchase Amount (USD)": ["mean", "count"]
            }).round(2)
            cluster_stats.columns = ["Idade Média", "Ticket Médio", "Clientes"]
            st.dataframe(cluster_stats, use_container_width=True)
    
    with col2:
        if len(X) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = sns.scatterplot(
                data=df_filtrado,
                x="Age",
                y="Purchase Amount (USD)",
                hue="Cluster",
                palette="Set2",
                s=100,
                alpha=0.6,
                ax=ax
            )
            ax.set_title('Segmentação de Clientes: Idade × Valor de Compra', fontsize=14, fontweight='bold')
            ax.set_xlabel('Idade')
            ax.set_ylabel('Valor da Compra (USD)')
            st.pyplot(fig)
            plt.close()

# Pergunta 3
with st.expander("3️⃣ Em quais estações e locais as vendas são mais intensas?"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 10: Estação × Localização")
        season_state = df_filtrado.groupby(["Season", "Location"])["Purchase Amount (USD)"].agg(['sum', 'count']).reset_index()
        season_state.columns = ["Estação", "Localização", "Valor Total", "Qtd Vendas"]
        season_state = season_state.sort_values("Valor Total", ascending=False).head(10)
        season_state["Valor Total"] = season_state["Valor Total"].apply(lambda x: f"${x:,.2f}")
        st.dataframe(season_state, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("Vendas por Estação")
        season_sales = df_filtrado.groupby("Season")["Purchase Amount (USD)"].sum().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        season_sales.plot(kind='barh', color='coral', ax=ax)
        ax.set_xlabel('Valor Total (USD)')
        ax.set_title('Vendas por Estação do Ano')
        for i, v in enumerate(season_sales):
            ax.text(v, i, f' ${v:,.0f}', va='center')
        st.pyplot(fig)
        plt.close()

# Pergunta 4
with st.expander("4️⃣ Quais categorias geram maior valor médio por transação?"):
    col1, col2 = st.columns(2)
    
    with col1:
        cat_stats = df_filtrado.groupby("Category").agg({
            "Purchase Amount (USD)": ["mean", "sum", "count"]
        }).round(2)
        cat_stats.columns = ["Ticket Médio", "Valor Total", "Qtd Vendas"]
        cat_stats = cat_stats.sort_values("Ticket Médio", ascending=False)
        cat_stats["Ticket Médio"] = cat_stats["Ticket Médio"].apply(lambda x: f"${x:.2f}")
        cat_stats["Valor Total"] = cat_stats["Valor Total"].apply(lambda x: f"${x:,.2f}")
        st.dataframe(cat_stats, use_container_width=True)
    
    with col2:
        cat_avg = df_filtrado.groupby("Category")["Purchase Amount (USD)"].mean().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        cat_avg.plot(kind='barh', color='teal', ax=ax)
        ax.set_xlabel('Valor Médio (USD)')
        ax.set_title('Ticket Médio por Categoria')
        for i, v in enumerate(cat_avg):
            ax.text(v, i, f' ${v:.2f}', va='center')
        st.pyplot(fig)
        plt.close()

# Pergunta 5
with st.expander("5️⃣ Qual a persona ideal para campanhas de alto valor?"):
    percentil_top = st.slider("Percentil dos Top Gastadores", 80, 99, 90, key="percentil_5")
    
    top_threshold = df_filtrado["Purchase Amount (USD)"].quantile(percentil_top/100)
    persona = df_filtrado[df_filtrado["Purchase Amount (USD)"] > top_threshold]
    
    st.info(f"📊 Analisando os **{100-percentil_top}% maiores gastadores** (compras acima de ${top_threshold:.2f})")
    
    if len(persona) > 0:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total de Top Gastadores", f"{len(persona):,}")
            st.metric("Idade Média", f"{persona['Age'].mean():.1f} anos")
            st.metric("Ticket Médio", f"${persona['Purchase Amount (USD)'].mean():.2f}")
        
        with col2:
            st.write("**Distribuição por Gênero:**")
            gender_dist = persona["Gender"].value_counts()
            for gender, count in gender_dist.items():
                st.write(f"- {gender}: {count} ({count/len(persona)*100:.1f}%)")
        
        with col3:
            st.write("**Top 3 Categorias:**")
            top_cats = persona["Category"].value_counts().head(3)
            for cat, count in top_cats.items():
                st.write(f"- {cat}: {count} ({count/len(persona)*100:.1f}%)")
        
        # Visualização
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            persona["Gender"].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=['#ff9999','#66b3ff'])
            ax.set_ylabel('')
            ax.set_title('Distribuição por Gênero')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            persona["Age"].hist(bins=20, color='lightgreen', edgecolor='black', ax=ax)
            ax.set_xlabel('Idade')
            ax.set_ylabel('Frequência')
            ax.set_title('Distribuição de Idade')
            st.pyplot(fig)
            plt.close()

# Pergunta 6
with st.expander("6️⃣ Como características do cliente se relacionam com valor gasto?"):
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=df_filtrado,
            x="Age",
            y="Purchase Amount (USD)",
            hue="Gender",
            alpha=0.6,
            s=80,
            ax=ax
        )
        ax.set_title('Idade × Valor × Gênero')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(
            data=df_filtrado,
            x="Gender",
            y="Purchase Amount (USD)",
            palette="Set3",
            ax=ax
        )
        ax.set_title('Distribuição de Valores por Gênero')
        st.pyplot(fig)
        plt.close()
    
    # Correlação adicional
    st.subheader("Análise por Categoria e Gênero")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=df_filtrado,
        x="Category",
        y="Purchase Amount (USD)",
        hue="Gender",
        palette="pastel",
        ax=ax
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title('Valor de Compra por Categoria e Gênero')
    st.pyplot(fig)
    plt.close()

# Pergunta 7
with st.expander("7️⃣ Modelo Preditivo: Quem são os futuros Big Spenders?"):
    st.subheader("🤖 Modelo LightGBM + Análise SHAP")
    
    with st.spinner("Treinando modelo de Machine Learning..."):
        try:
            df_model = df_filtrado.copy()
            threshold_model = df_model["Purchase Amount (USD)"].quantile(0.8)
            df_model["BigSpender"] = (df_model["Purchase Amount (USD)"] > threshold_model).astype(int)
            
            # Preparar features
            X = pd.get_dummies(df_model[["Age", "Gender", "Category", "Season"]], drop_first=True)
            y = df_model["BigSpender"]
            
            # Verificar se há dados suficientes
            if len(X) < 10 or y.sum() < 2:
                st.warning("⚠️ Dados insuficientes para treinar o modelo. Ajuste os filtros.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
                
                model = LGBMClassifier(random_state=42, verbose=-1, force_col_wise=True)
                model.fit(X_train, y_train)
                
                # Métricas do modelo
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Acurácia", f"{(y_pred == y_test).mean():.2%}")
                with col2:
                    st.metric("ROC-AUC", f"{roc_auc_score(y_test, y_pred_proba):.3f}")
                with col3:
                    st.metric("Big Spenders Identificados", f"{y_pred.sum()}/{len(y_test)}")
                
                # SHAP Analysis
                st.subheader("📊 Importância das Features (SHAP)")
                explainer = TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                summary_plot(shap_values[1] if isinstance(shap_values, list) else shap_values, 
                           X_test, plot_type="bar", show=False)
                st.pyplot(plt.gcf())
                plt.close()
                
                # Relatório de classificação
                with st.expander("Ver Relatório Detalhado"):
                    st.text(classification_report(y_test, y_pred, target_names=["Regular", "Big Spender"]))
        
        except Exception as e:
            st.error(f"Erro ao treinar o modelo: {str(e)}")

# ==========================
# 📊 Exportar Dados
# ==========================
st.markdown("---")
st.subheader("💾 Exportar Dados Filtrados")
col1, col2 = st.columns([3, 1])
with col1:
    st.info(f"📋 {len(df_filtrado)} registros prontos para exportação")
with col2:
    csv = df_filtrado.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download CSV",
        data=csv,
        file_name="dados_filtrados.csv",
        mime="text/csv",
        use_container_width=True
    )

# Rodapé
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    Desenvolvido com ❤️ usando Streamlit | Dataset: Customer Shopping Behavior<br>
    <small>Dashboard v2.0 - Análise Avançada de Varejo</small>
</div>
""", unsafe_allow_html=True)
