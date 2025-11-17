from dataclasses import dataclass

@dataclass
class Config:
    # Paths
    CSV_PATH: str = "shopping_behavior_updated.csv"
    
    # UI
    PAGE_TITLE: str = "Dashboard Varejo - 7 Perguntas"
    PAGE_ICON: str = ""
    PRIMARY_COLOR: str = "#1f77b4"
    SPINNER_MODEL: str = " Treinando modelo de Machine Learning..."
    SPINNER_CLUSTER: str = " Realizando segmentação..."
    
    # Data
    REQUIRED_COLS: list = None
    
    def __post_init__(self):
        self.REQUIRED_COLS = [
            "Category", "Gender", "Age", 
            "Purchase Amount (USD)", "Season", "Location"
        ]

@dataclass
class ModelParams:
    # Big Spenders
    BIG_SPENDER_PERCENTILE: float = 0.8
    PERCENTILE_MIN: int = 70
    PERCENTILE_MAX: int = 95
    PERCENTILE_DEFAULT: int = 80
    
    # Clustering
    N_CLUSTERS_MIN: int = 2
    N_CLUSTERS_MAX: int = 6
    N_CLUSTERS_DEFAULT: int = 3
    
    # ML
    TEST_SIZE: float = 0.3
    RANDOM_STATE: int = 42
    MIN_SAMPLES_FOR_MODEL: int = 10
    MIN_POSITIVE_SAMPLES: int = 2
    LGBM_N_ESTIMATORS: int = 100
    LGBM_MAX_DEPTH: int = 5
    LGBM_LEARNING_RATE: float = 0.1

CFG = Config()
MODEL_CFG = ModelParams()

