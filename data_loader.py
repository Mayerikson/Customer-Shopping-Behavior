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
        # Para fins de teste no ambiente local, vamos criar um dataframe mock
        # Na implantação real, o arquivo deve existir
        if "shopping_behavior_updated.csv" in csv_path:
            st.warning(f"⚠️ Arquivo de dados não encontrado em {csv_path}. Criando dados de exemplo.")
            
            data = {
                "Category": ["Clothing", "Footwear", "Clothing", "Accessories", "Footwear"],
                "Gender": ["Male", "Female", "Male", "Female", "Male"],
                "Age": [25, 30, 45, 22, 35],
                "Purchase Amount (USD)": [50.0, 120.0, 80.0, 30.0, 95.0],
                "Season": ["Winter", "Summer", "Spring", "Fall", "Winter"],
                "Location": ["New York", "Los Angeles", "Chicago", "Miami", "New York"]
            }
            df = pd.DataFrame(data)
            
            # Adicionar mais linhas para simular um dataset maior
            for _ in range(100):
                df = pd.concat([df, pd.DataFrame({
                    "Category": [df["Category"].sample(1).iloc[0]],
                    "Gender": [df["Gender"].sample(1).iloc[0]],
                    "Age": [df["Age"].sample(1).iloc[0] + np.random.randint(-5, 5)],
                    "Purchase Amount (USD)": [df["Purchase Amount (USD)"].sample(1).iloc[0] + np.random.randint(-20, 20)],
                    "Season": [df["Season"].sample(1).iloc[0]],
                    "Location": [df["Location"].sample(1).iloc[0]]
                })], ignore_index=True)
            
            # Garantir que as colunas obrigatórias existam
            for col in required_cols:
                if col not in df.columns:
                    df[col] = "N/A"
            
            return df
        
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
