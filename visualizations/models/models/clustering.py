"""Modelos de clustering"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import streamlit as st

@st.cache_data
def realizar_clustering(
    df: pd.DataFrame, 
    n_clusters: int,
    features: list = None
) -> pd.DataFrame:
    """
    Realiza clustering KMeans nos dados.
    
    Args:
        df: DataFrame com os dados
        n_clusters: Número de clusters
        features: Lista de features para clustering
        
    Returns:
        DataFrame com coluna 'Cluster' adicionada
    """
    if features is None:
        features = ["Age", "Purchase Amount (USD)"]
    
    # Remover NaN
    X = df[features].dropna()
    
    if len(X) == 0:
        return df
    
    # Clustering
    kmeans = KMeans(
        n_clusters=n_clusters, 
        random_state=42, 
        n_init=10
    )
    
    # Criar cópia do dataframe
    df_result = df.copy()
    
    # Adicionar clusters apenas para linhas sem NaN
    clusters = np.full(len(df), -1)  # -1 para valores ausentes
    valid_indices = df[features].dropna().index
    clusters[valid_indices] = kmeans.fit_predict(X)
    
    df_result["Cluster"] = clusters
    
    return df_result

def calcular_estatisticas_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula estatísticas de cada cluster.
    
    Args:
        df: DataFrame com coluna 'Cluster'
        
    Returns:
        DataFrame com estatísticas por cluster
    """
    # Filtrar apenas clusters válidos
    df_valido = df[df["Cluster"] != -1]
    
    if len(df_valido) == 0:
        return pd.DataFrame()
    
    stats = df_valido.groupby("Cluster").agg({
        "Age": "mean",
        "Purchase Amount (USD)": ["mean", "count"]
    }).round(2)
    
    stats.columns = ["Idade Média", "Ticket Médio", "Clientes"]
    stats = stats.sort_values("Ticket Médio", ascending=False)
    
    return stats
