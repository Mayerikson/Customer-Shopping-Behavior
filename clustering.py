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
    
    # Pré-processamento: Selecionar features e remover NaNs
    df_cluster = df[features].dropna().copy()
    
    if len(df_cluster) < n_clusters:
        st.warning("Dados insuficientes para o número de clusters solicitado.")
        df['Cluster'] = -1
        return df
    
    # Normalização (opcional, mas recomendado para KMeans)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster)
    
    # Aplicação do KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_cluster['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Mapear clusters de volta ao DataFrame original
    df = df.merge(
        df_cluster[['Cluster']], 
        left_index=True, 
        right_index=True, 
        how='left'
    )
    df['Cluster'] = df['Cluster'].fillna(-1).astype(int)
    
    return df

@st.cache_data
def calcular_estatisticas_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula estatísticas descritivas para cada cluster.
    
    Args:
        df: DataFrame com coluna 'Cluster'
        
    Returns:
        DataFrame com estatísticas por cluster
    """
    # Excluir dados não clusterizados (-1)
    df_valid = df[df['Cluster'] != -1]
    
    if df_valid.empty:
        return pd.DataFrame()
    
    stats = df_valid.groupby('Cluster').agg(
        Clientes=('Cluster', 'size'),
        Idade_Media=('Age', 'mean'),
        Ticket_Medio=('Purchase Amount (USD)', 'mean'),
        Max_Compra=('Purchase Amount (USD)', 'max'),
        Genero_Mais_Comum=('Gender', lambda x: x.mode()[0] if not x.mode().empty else 'N/A'),
        Categoria_Mais_Comum=('Category', lambda x: x.mode()[0] if not x.mode().empty else 'N/A')
    ).reset_index()
    
    stats['Idade_Media'] = stats['Idade_Media'].round(1)
    stats['Ticket_Medio'] = stats['Ticket_Medio'].round(2)
    stats['Max_Compra'] = stats['Max_Compra'].round(2)
    
    return stats
