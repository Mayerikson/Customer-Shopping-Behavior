"""Funções para criar gráficos"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Tuple

def configurar_estilo_plots():
    """Configura estilo global dos plots"""
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'

def criar_grafico_distribuicao(
    df: pd.DataFrame, 
    threshold: float,
    figsize: Tuple[int, int] = (10, 4)
) -> plt.Figure:
    """
    Cria gráfico de distribuição de valores de compra.
    
    Args:
        df: DataFrame com os dados
        threshold: Linha de corte
        figsize: Tamanho da figura
        
    Returns:
        Figura matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(
        df["Purchase Amount (USD)"], 
        bins=50, 
        color='skyblue', 
        edgecolor='black', 
        alpha=0.7
    )
    
    ax.axvline(
        threshold, 
        color='red', 
        linestyle='--', 
        linewidth=2, 
        label=f'Threshold: ${threshold:.2f}'
    )
    
    ax.set_xlabel('Valor da Compra (USD)', fontsize=11)
    ax.set_ylabel('Frequência', fontsize=11)
    ax.set_title('Distribuição de Valores de Compra', fontsize=13, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    return fig

def criar_grafico_clusters(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Cria gráfico de segmentação por clusters.
    
    Args:
        df: DataFrame com coluna 'Cluster'
        figsize: Tamanho da figura
        
    Returns:
        Figura matplotlib
    """
    # Filtrar clusters válidos
    df_plot = df[df["Cluster"] != -1]
    
    if df_plot.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Dados insuficientes para visualização de clusters", 
                ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.scatterplot(
        data=df_plot,
        x="Age",
        y="Purchase Amount (USD)",
        hue="Cluster",
        palette="Set2",
        s=100,
        alpha=0.6,
        ax=ax
    )
    
    ax.set_title(
        'Segmentação de Clientes: Idade × Valor de Compra', 
        fontsize=14, 
        fontweight='bold'
    )
    ax.set_xlabel('Idade', fontsize=11)
    ax.set_ylabel('Valor da Compra (USD)', fontsize=11)
    
    plt.tight_layout()
    return fig

def criar_grafico_barras_horizontal(
    series: pd.Series,
    titulo: str,
    xlabel: str = 'Valor',
    color: str = 'coral',
    figsize: Tuple[int, int] = (8, 5)
) -> plt.Figure:
    """
    Cria gráfico de barras horizontal.
    
    Args:
        series: Pandas Series com os dados
        titulo: Título do gráfico
        xlabel: Label do eixo X
        color: Cor das barras
        figsize: Tamanho da figura
        
    Returns:
        Figura matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    series.plot(kind='barh', color=color, ax=ax)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(titulo, fontsize=13, fontweight='bold')
    
    # Adicionar valores nas barras
    for i, v in enumerate(series):
        ax.text(v, i, f' {v:,.0f}', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig

def criar_grafico_pizza(
    series: pd.Series,
    titulo: str,
    colors: list = None,
    figsize: Tuple[int, int] = (6, 4)
) -> plt.Figure:
    """
    Cria gráfico de pizza.
    
    Args:
        series: Pandas Series com os dados
        titulo: Título do gráfico
        colors: Lista de cores
        figsize: Tamanho da figura
        
    Returns:
        Figura matplotlib
    """
    if colors is None:
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    series.plot(
        kind='pie', 
        autopct='%1.1f%%', 
        ax=ax, 
        colors=colors,
        startangle=90
    )
    
    ax.set_ylabel('')
    ax.set_title(titulo, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig

def criar_scatter_idade_valor(
    df: pd.DataFrame,
    hue_col: str = "Gender",
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Cria scatter plot de idade vs valor.
    
    Args:
        df: DataFrame com os dados
        hue_col: Coluna para colorir pontos
        figsize: Tamanho da figura
        
    Returns:
        Figura matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.scatterplot(
        data=df,
        x="Age",
        y="Purchase Amount (USD)",
        hue=hue_col,
        alpha=0.6,
        s=80,
        ax=ax
    )
    
    ax.set_title(
        f'Idade × Valor × {hue_col}', 
        fontsize=13, 
        fontweight='bold'
    )
    ax.set_xlabel('Idade', fontsize=11)
    ax.set_ylabel('Valor da Compra (USD)', fontsize=11)
    
    plt.tight_layout()
    return fig

def criar_boxplot_genero(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Cria boxplot de valores por gênero.
    
    Args:
        df: DataFrame com os dados
        figsize: Tamanho da figura
        
    Returns:
        Figura matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.boxplot(
        data=df,
        x="Gender",
        y="Purchase Amount (USD)",
        palette="Set3",
        ax=ax
    )
    
    ax.set_title(
        'Distribuição de Valores por Gênero', 
        fontsize=13, 
        fontweight='bold'
    )
    ax.set_xlabel('Gênero', fontsize=11)
    ax.set_ylabel('Valor da Compra (USD)', fontsize=11)
    
    plt.tight_layout()
    return fig

# Configurar estilo ao importar
configurar_estilo_plots()
