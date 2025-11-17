"""Configurações centralizadas do dashboard"""
from dataclasses import dataclass
import os

@dataclass
class DashboardConfig:
    """Configurações gerais do dashboard"""
    
    # Paths
    # O arquivo CSV será colocado na raiz do projeto para simplificar a implantação
    CSV_PATH: str = "shopping_behavior_updated.csv"
    
    # Títulos
    PAGE_TITLE: str = "Dashboard Varejo - 7 Perguntas"
    PAGE_ICON: str = "📊"
    
    # Colunas obrigatórias
    REQUIRED_COLUMNS: list = None
    
    def __post_init__(self):
        self.REQUIRED_COLUMNS = [
            "Category", "Gender", "Age", 
            "Purchase Amount (USD)", "Season", "Location"
        ]

@dataclass
class ModelConfig:
    """Configurações para modelos de ML"""
    
    # Big Spenders
    BIG_SPENDER_PERCENTILE: float = 0.8
    PERCENTILE_MIN: int = 70
    PERCENTILE_MAX: int = 95
    PERCENTILE_DEFAULT: int = 80
    
    # Clustering
    N_CLUSTERS_MIN: int = 2
    N_CLUSTERS_MAX: int = 6
    N_CLUSTERS_DEFAULT: int = 3
    
    # Machine Learning
    TEST_SIZE: float = 0.3
    RANDOM_STATE: int = 42
    MIN_SAMPLES_FOR_MODEL: int = 10
    MIN_POSITIVE_SAMPLES: int = 2
    
    # LightGBM
    LGBM_N_ESTIMATORS: int = 100
    LGBM_MAX_DEPTH: int = 5
    LGBM_LEARNING_RATE: float = 0.1

@dataclass
class UIConfig:
    """Configurações de interface"""
    
    # Cores
    PRIMARY_COLOR: str = "#1f77b4"
    
    # Plot sizes
    PLOT_SIZE_SMALL: tuple = (8, 4)
    PLOT_SIZE_MEDIUM: tuple = (10, 6)
    PLOT_SIZE_LARGE: tuple = (12, 6)
    
    # Animações
    SPINNER_TEXT_MODEL: str = "🤖 Treinando modelo de Machine Learning..."
    SPINNER_TEXT_CLUSTER: str = "📊 Realizando segmentação..."

# Instâncias globais
DASHBOARD_CONFIG = DashboardConfig()
MODEL_CONFIG = ModelConfig()
UI_CONFIG = UIConfig()
