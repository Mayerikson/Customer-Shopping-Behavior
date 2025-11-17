"""Componentes das 7 perguntas de negócio"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from shap import summary_plot

from config.settings import MODEL_CONFIG, UI_CONFIG
from utils.formatters import formatar_moeda, formatar_percentual, formatar_numero
from utils.data_processor import (
    calcular_big_spenders, 
    preparar_dados_top_gastadores,
    calcular_vendas_por_dimensao
)
from models.clustering import realizar_clustering, calcular_estatisticas_clusters
from models.prediction import (
    preparar_dados_modelo, 
    treinar_modelo_big_spender,
    avaliar_modelo,
    calcular_shap_values
)
from visualizations.charts import (
    criar_grafico_distribuicao,
    criar_grafico_clusters,
    criar_grafico_barras_horizontal,
    criar_grafico_pizza,
    criar_scatter_idade_valor,
    criar_boxplot_genero
)

def pergunta_1_probabilidade_big_spender(df: pd.DataFrame):
    """Pergunta 1: Qual a probabilidade de um cliente ser Big Spender?"""
    with st.expander(
        "1️⃣ Qual a probabilidade de um cliente ser Big Spender?", 
        expanded=True
    ):
        col1, col2 = st.columns([1, 2])
        
        with col1:
            percentil = st.slider(
                "Percentil de Corte", 
                MODEL_CONFIG.PERCENTILE_MIN,
                MODEL_CONFIG.PERCENTILE_MAX,
                MODEL_CONFIG.PERCENTILE_DEFAULT,
                5,
                key="percentil_1",
                help="Define o limite superior de gastos"
            )
            
            bs, threshold = calcular_big_spenders(df, percentil/100)
            prob = len(bs) / len(df) if len(df) > 0 else 0
            
            st.metric(
                f"Probabilidade (Top {100-percentil}%)",
                formatar_percentual(prob),
                help=f"Clientes com compras acima de {formatar_moeda(threshold)}"
            )
            st.metric("Valor de Corte", formatar_moeda(threshold))
            st.metric("Big Spenders", formatar_numero(len(bs)))
        
        with col2:
            fig = criar_grafico_distribuicao(df, threshold)
            st.pyplot(fig)
            plt.close(fig)

def pergunta_2_segmentos_consumidores(df: pd.DataFrame):
    """Pergunta 2: Quais são os segmentos naturais de consumidores?"""
    with st.expander("2️⃣ Quais são os segmentos naturais de consumidores?"):
        col1, col2 = st.columns([1, 2])
        
        with col1:
            n_clusters = st.slider(
                "Número de Clusters",
                MODEL_CONFIG.N_CLUSTERS_MIN,
                MODEL_CONFIG.N_CLUSTERS_MAX,
                MODEL_CONFIG.N_CLUSTERS_DEFAULT,
                key="clusters",
                help="Número de segmentos de clientes"
            )
            
            with st.spinner(UI_CONFIG.SPINNER_TEXT_CLUSTER):
                df_clusters = realizar_clustering(df, n_clusters)
                cluster_stats = calcular_estatisticas_clusters(df_clusters)
            
            if not cluster_stats.empty:
                st.write("**Características dos Clusters:**")
                st.dataframe(cluster_stats, use_container_width=True)
            else:
                st.warning("⚠️ Dados insuficientes para clustering")
        
        with col2:
            if not cluster_stats.empty:
                fig = criar_grafico_clusters(df_clusters)
                st.pyplot(fig)
                plt.close(fig)

def pergunta_3_vendas_intensas(df: pd.DataFrame):
    """Pergunta 3: Em quais estações e locais as vendas são mais intensas?"""
    with st.expander("3️⃣ Em quais estações e locais as vendas são mais intensas?"):
        vendas = calcular_vendas_por_dimensao(df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 10: Estação × Localização")
            season_state = vendas['por_estacao_local'].head(10).copy()
            season_state.columns = ["Estação", "Localização", "Valor Total", "Qtd Vendas"]
            season_state["Valor Total"] = season_state["Valor Total"].apply(formatar_moeda)
            st.dataframe(season_state, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("Vendas por Estação")
            fig = criar_grafico_barras_horizontal(
                vendas['por_estacao'],
                "Vendas por Estação do Ano",
                "Valor Total (USD)"
            )
            st.pyplot(fig)
            plt.close(fig)

def pergunta_4_categorias_maior_valor(df: pd.DataFrame):
    """Pergunta 4: Quais categorias geram maior valor médio por transação?"""
    with st.expander("4️⃣ Quais categorias geram maior valor médio por transação?"):
        vendas = calcular_vendas_por_dimensao(df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            cat_stats = vendas['por_categoria'].copy()
            cat_stats.columns = ["Ticket Médio", "Valor Total", "Qtd Vendas"]
            cat_stats = cat_stats.sort_values("Ticket Médio", ascending=False)
            cat_stats["Ticket Médio"] = cat_stats["Ticket Médio"].apply(formatar_moeda)
            cat_stats["Valor Total"] = cat_stats["Valor Total"].apply(formatar_moeda)
            st.dataframe(cat_stats, use_container_width=True)
        
        with col2:
            cat_avg = df.groupby("Category")["Purchase Amount (USD)"].mean().sort_values(ascending=True)
            fig = criar_grafico_barras_horizontal(
                cat_avg,
                "Ticket Médio por Categoria",
                "Valor Médio (USD)",
                color='teal'
            )
            st.pyplot(fig)
            plt.close(fig)

def pergunta_5_persona_ideal(df: pd.DataFrame):
    """Pergunta 5: Qual a persona ideal para campanhas de alto valor?"""
    with st.expander("5️⃣ Qual a persona ideal para campanhas de alto valor?"):
        percentil_top = st.slider(
            "Percentil dos Top Gastadores", 
            80, 99, 90, 
            key="percentil_5",
            help="Define o percentil dos maiores gastadores"
        )
        
        persona_data = preparar_dados_top_gastadores(df, percentil_top/100)
        
        if persona_data is None:
            st.warning("⚠️ Dados insuficientes para análise de persona")
            return
        
        st.info(
            f"📊 Analisando os **{100-percentil_top}% maiores gastadores** "
            f"(compras acima de {formatar_moeda(persona_data['threshold'])})"
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total de Top Gastadores", formatar_numero(persona_data['total']))
            st.metric("Idade Média", f"{persona_data['idade_media']:.1f} anos")
            st.metric("Ticket Médio", formatar_moeda(persona_data['ticket_medio']))
        
        with col2:
            st.write("**Distribuição por Gênero:**")
            for gender, count in persona_data['genero_dist'].items():
                pct = (count / persona_data['total']) * 100
                st.write(f"- {gender}: {count} ({pct:.1f}%)")
        
        with col3:
            st.write("**Top 3 Categorias:**")
            for cat, count in persona_data['top_categorias'].items():
                pct = (count / persona_data['total']) * 100
                st.write(f"- {cat}: {count} ({pct:.1f}%)")
        
        # Visualizações
        col1, col2 = st.columns(2)
        
        with col1:
            gender_series = pd.Series(persona_data['genero_dist'])
            fig = criar_grafico_pizza(gender_series, 'Distribuição por Gênero')
            st.pyplot(fig)
            plt.close(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            persona_data['df']["Age"].hist(
                bins=20, 
                color='lightgreen', 
                edgecolor='black', 
                ax=ax
            )
            ax.set_xlabel('Idade')
            ax.set_ylabel('Frequência')
            ax.set_title('Distribuição de Idade', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)


def pergunta_6_relacao_caracteristicas(df: pd.DataFrame):
    """Pergunta 6: Como características do cliente se relacionam com valor gasto?"""
    with st.expander("6️⃣ Como características do cliente se relacionam com valor gasto?"):
        col1, col2 = st.columns(2)
        
        with col1:
            fig = criar_scatter_idade_valor(df)
            st.pyplot(fig)
            plt.close(fig)
        
        with col2:
            fig = criar_boxplot_genero(df)
            st.pyplot(fig)
            plt.close(fig)
        
        # Análise adicional por categoria e gênero
        st.subheader("Análise por Categoria e Gênero")
        fig, ax = plt.subplots(figsize=(12, 6))
        import seaborn as sns
        sns.boxplot(
            data=df,
            x="Category",
            y="Purchase Amount (USD)",
            hue="Gender",
            palette="pastel",
            ax=ax
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title('Valor de Compra por Categoria e Gênero', fontweight='bold')
        ax.set_xlabel('Categoria')
        ax.set_ylabel('Valor da Compra (USD)')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

def pergunta_7_modelo_preditivo(df: pd.DataFrame):
    """Pergunta 7: Modelo Preditivo - Quem são os futuros Big Spenders?"""
    with st.expander("7️⃣ Modelo Preditivo: Quem são os futuros Big Spenders?"):
        st.subheader("🤖 Modelo LightGBM + Análise SHAP")
        
        with st.spinner(UI_CONFIG.SPINNER_TEXT_MODEL):
            try:
                # Preparar dados
                threshold_model = df["Purchase Amount (USD)"].quantile(
                    MODEL_CONFIG.BIG_SPENDER_PERCENTILE
                )
                
                dados = preparar_dados_modelo(
                    df, 
                    threshold_model,
                    MODEL_CONFIG.TEST_SIZE,
                    MODEL_CONFIG.RANDOM_STATE
                )
                
                if dados is None:
                    st.warning(
                        "⚠️ Dados insuficientes para treinar o modelo. "
                        "Ajuste os filtros para incluir mais dados."
                    )
                    return
                
                # Treinar modelo
                model = treinar_modelo_big_spender(
                    dados['X_train'], 
                    dados['y_train'],
                    n_estimators=MODEL_CONFIG.LGBM_N_ESTIMATORS,
                    max_depth=MODEL_CONFIG.LGBM_MAX_DEPTH,
                    learning_rate=MODEL_CONFIG.LGBM_LEARNING_RATE
                )
                
                # Avaliar modelo
                metricas = avaliar_modelo(
                    model, 
                    dados['X_test'], 
                    dados['y_test']
                )
                
                # Exibir métricas
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Acurácia", 
                        f"{metricas['acuracia']:.2%}",
                        help="Proporção de predições corretas"
                    )
                
                with col2:
                    st.metric(
                        "ROC-AUC", 
                        f"{metricas['roc_auc']:.3f}",
                        help="Área sob a curva ROC (0.5 = aleatório, 1.0 = perfeito)"
                    )
                
                with col3:
                    st.metric(
                        "Big Spenders Identificados", 
                        f"{metricas['big_spenders_pred']}/{metricas['total_test']}",
                        help="Quantidade de Big Spenders preditos no conjunto de teste"
                    )
                
                # SHAP Analysis
                st.subheader("📊 Importância das Features (SHAP)")
                
                shap_values = calcular_shap_values(model, dados['X_test'])
                
                fig, ax = plt.subplots(figsize=(10, 6))
                summary_plot(
                    shap_values, 
                    dados['X_test'], 
                    plot_type="bar", 
                    show=False
                )
                st.pyplot(plt.gcf())
                plt.close()
                
                # Relatório detalhado
                with st.expander("📄 Ver Relatório Detalhado"):
                    st.text(metricas['report'])
                    
                    st.markdown("### 💡 Interpretação")
                    st.info("""
                    **Como interpretar os resultados:**
                    
                    - **Acurácia**: Percentual de predições corretas
                    - **ROC-AUC**: Capacidade do modelo de distinguir entre classes (>0.7 é bom)
                    - **SHAP**: Features mais importantes têm barras maiores
                    
                    **Próximos passos:**
                    1. Identifique as features mais importantes
                    2. Foque estratégias de marketing nessas características
                    3. Monitore a evolução das predições ao longo do tempo
                    """)
            
            except ValueError as e:
                st.error(f"❌ Erro nos dados: {str(e)}")
            except Exception as e:
                st.error(f"❌ Erro inesperado ao treinar o modelo: {str(e)}")
                import traceback
                with st.expander("🔍 Detalhes do erro"):
                    st.code(traceback.format_exc())
