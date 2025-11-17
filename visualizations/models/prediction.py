"""Modelo preditivo de Big Spenders"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from lightgbm import LGBMClassifier
from shap import TreeExplainer
import streamlit as st
from typing import Optional, Tuple

@st.cache_resource
def treinar_modelo_big_spender(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 5,
    learning_rate: float = 0.1
) -> LGBMClassifier:
    """
    Treina modelo LightGBM para predição de Big Spenders.
    
    Args:
        X_train: Features de treino
        y_train: Target de treino
        n_estimators: Número de árvores
        max_depth: Profundidade máxima
        learning_rate: Taxa de aprendizado
        
    Returns:
        Modelo treinado
    """
    model = LGBMClassifier(
        random_state=42,
        verbose=-1,
        force_col_wise=True,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate
    )
    
    model.fit(X_train, y_train)
    return model

def preparar_dados_modelo(
    df: pd.DataFrame,
    threshold: float,
    test_size: float = 0.3,
    random_state: int = 42
) -> Optional[dict]:
    """
    Prepara dados para treinamento do modelo.
    
    Args:
        df: DataFrame com os dados
        threshold: Threshold para definir Big Spender
        test_size: Proporção de teste
        random_state: Seed para reprodutibilidade
        
    Returns:
        Dicionário com dados preparados ou None se insuficiente
    """
    df_model = df.copy()
    df_model["BigSpender"] = (
        df_model["Purchase Amount (USD)"] > threshold
    ).astype(int)
    
    # Preparar features
    X = pd.get_dummies(
        df_model[["Age", "Gender", "Category", "Season"]], 
        drop_first=True
    )
    y = df_model["BigSpender"]
    
    # Verificar se há dados suficientes
    if len(X) < 10 or y.sum() < 2:
        return None
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

def avaliar_modelo(
    model: LGBMClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """
    Avalia performance do modelo.
    
    Args:
        model: Modelo treinado
        X_test: Features de teste
        y_test: Target de teste
        
    Returns:
        Dicionário com métricas
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    return {
        'acuracia': (y_pred == y_test).mean(),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'big_spenders_pred': y_pred.sum(),
        'total_test': len(y_test),
        'y_pred': y_pred,
        'y_test': y_test,
        'report': classification_report(
            y_test, y_pred, 
            target_names=["Regular", "Big Spender"]
        )
    }

def calcular_shap_values(
    model: LGBMClassifier,
    X_test: pd.DataFrame
):
    """
    Calcula SHAP values para interpretabilidade.
    
    Args:
        model: Modelo treinado
        X_test: Features de teste
        
    Returns:
        SHAP values
    """
    explainer = TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # LightGBM pode retornar lista ou array
    if isinstance(shap_values, list):
        return shap_values[1]  # Classe positiva
    return shap_values
