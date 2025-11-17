"""Carregamento e validação de dados"""
import os
import pandas as pd
import streamlit as st
from typing import List

@st.cache_data(ttl=3600)  # Cache por 1 hora
def load_and_validate_data(
    csv_path: str, 
    required_cols: List[str]
) -> pd.DataFrame:
    """
    Carrega e valida o dataset.
    
    Args:
        csv_path: Caminho para o arquivo CSV
        required_cols: Lista de colunas obrigatórias
        
    Returns:
        DataFrame validado
        
    Raises:
        FileNotFoundError: Se arquivo não existir
        ValueError: Se colunas obrigatórias estiverem ausentes
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ Arquivo não encontrado: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"❌ Erro ao ler CSV: {str(e)}")
    
    # Validação de colunas
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(
            f"❌ Colunas ausentes no dataset: {', '.join(missing_cols)}"
        )
    
    return df

@st.cache_data
def calcular_estatisticas_gerais(df: pd.DataFrame) -> dict:
    """
    Calcula estatísticas gerais do dataset.
    
    Args:
        df: DataFrame com os dados
        
    Returns:
        Dicionário com estatísticas
    """
    return {
        'total_clientes': len(df),
        'ticket_medio': df['Purchase Amount (USD)'].mean(),
        'valor_total': df['Purchase Amount (USD)'].sum(),
        'valor_maximo': df['Purchase Amount (USD)'].max(),
        'categorias': df["Category"].nunique()
    }
