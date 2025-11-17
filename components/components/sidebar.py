"""Componente de sidebar com filtros"""
import streamlit as st
import pandas as pd
from typing import Tuple, List

def criar_sidebar(df: pd.DataFrame) -> dict:
    """
    Cria sidebar com filtros interativos.
    
    Args:
        df: DataFrame com os dados
        
    Returns:
        Dicionário com valores dos filtros selecionados
    """
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
        
        # Botões de controle
        col1, col2 = st.columns(2)
        with col1:
            aplicar = st.button(
                "🔍 Aplicar", 
                type="primary", 
                use_container_width=True
            )
        with col2:
            limpar = st.button(
                "🔄 Limpar", 
                use_container_width=True
            )
        
        # Informações adicionais
        st.markdown("---")
        st.markdown("### 📊 Sobre os Dados")
        st.info(f"""
        **Dataset:** Customer Shopping Behavior
        
        **Total de registros:** {len(df):,}
        
        **Categorias:** {df["Category"].nunique()}
        
        **Período:** Todas as estações
        """)
    
    return {
        'categorias': categorias,
        'generos': generos,
        'faixa_etaria': faixa_etaria,
        'estacoes': estacoes,
        'aplicar': aplicar,
        'limpar': limpar
    }
