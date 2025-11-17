"""Processamento e transformação de dados"""
import pandas as pd
import streamlit as st
from typing import Tuple, List
from utils.data_loader import calcular_estatisticas_gerais

def aplicar_filtros(
    df: pd.DataFrame,
    categorias: List[str],
    generos: List[str],
    faixa_etaria: Tuple[int, int],
    estacoes: List[str]
) -> pd.DataFrame:
    """
    Aplica filtros ao dataframe.
    
    Args:
        df: DataFrame original
        categorias: Lista de categorias selecionadas
        generos: Lista de gêneros selecionados
        faixa_etaria: Tupla (min, max) de idade
        estacoes: Lista de estações selecionadas
        
    Returns:
        DataFrame filtrado
    """
    return df[
        (df["Category"].isin(categorias)) &
        (df["Gender"].isin(generos)) &
        (df["Age"].between(faixa_etaria[0], faixa_etaria[1])) &
        (df["Season"].isin(estacoes))
    ].copy()

@st.cache_data
def calcular_big_spenders(
    df: pd.DataFrame, 
    percentile: float
) -> Tuple[pd.DataFrame, float]:
    """
    Identifica big spenders baseado em percentil.
    
    Args:
        df: DataFrame com os dados
        percentile: Percentil de corte (0.0 a 1.0)
        
    Returns:
        Tuple de (dataframe de big spenders, threshold usado)
    """
    threshold = df["Purchase Amount (USD)"].quantile(percentile)
    big_spenders = df[df["Purchase Amount (USD)"] > threshold].copy()
    return big_spenders, threshold

def calcular_estatisticas_filtradas(
    df_original: pd.DataFrame,
    df_filtrado: pd.DataFrame
) -> dict:
    """
    Calcula estatísticas dos dados filtrados com comparação.
    
    Args:
        df_original: DataFrame original
        df_filtrado: DataFrame filtrado
        
    Returns:
        Dicionário com estatísticas e deltas
    """
    stats_original = calcular_estatisticas_gerais(df_original)
    stats_filtrado = calcular_estatisticas_gerais(df_filtrado)
    
    # Evitar divisão por zero
    total_clientes_original = stats_original['total_clientes'] if stats_original['total_clientes'] > 0 else 1
    
    return {
        'clientes': stats_filtrado['total_clientes'],
        'delta_clientes_pct': (
            (stats_filtrado['total_clientes'] / total_clientes_original - 1) * 100
        ),
        'ticket_medio': stats_filtrado['ticket_medio'],
        'delta_ticket': stats_filtrado['ticket_medio'] - stats_original['ticket_medio'],
        'valor_total': stats_filtrado['valor_total'],
        'valor_maximo': stats_filtrado['valor_maximo']
    }

@st.cache_data
def preparar_dados_top_gastadores(
    df: pd.DataFrame,
    percentil: float
) -> dict:
    """
    Prepara análise de persona dos top gastadores.
    
    Args:
        df: DataFrame com os dados
        percentil: Percentil de corte
        
    Returns:
        Dicionário com análise de persona
    """
    threshold = df["Purchase Amount (USD)"].quantile(percentil)
    persona = df[df["Purchase Amount (USD)"] > threshold].copy()
    
    if len(persona) == 0:
        return None
    
    return {
        'df': persona,
        'threshold': threshold,
        'total': len(persona),
        'idade_media': persona['Age'].mean(),
        'ticket_medio': persona['Purchase Amount (USD)'].mean(),
        'genero_dist': persona["Gender"].value_counts().to_dict(),
        'top_categorias': persona["Category"].value_counts().head(3).to_dict()
    }

@st.cache_data
def calcular_vendas_por_dimensao(df: pd.DataFrame) -> dict:
    """
    Calcula vendas agregadas por diferentes dimensões.
    
    Args:
        df: DataFrame com os dados
        
    Returns:
        Dicionário com agregações
    """
    return {
        'por_estacao_local': df.groupby(["Season", "Location"])["Purchase Amount (USD)"]
            .agg(['sum', 'count']).reset_index()
            .sort_values('sum', ascending=False),
        
        'por_estacao': df.groupby("Season")["Purchase Amount (USD)"]
            .sum().sort_values(ascending=True),
        
        'por_categoria': df.groupby("Category").agg({
            "Purchase Amount (USD)": ["mean", "sum", "count"]
        }).round(2)
    }
