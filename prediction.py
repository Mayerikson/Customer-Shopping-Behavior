"""Modelos de predição (LightGBM)"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, classification_report
from shap import TreeExplainer
import streamlit as st
from typing import Tuple

def preparar_dados_modelo(
    df: pd.DataFrame, 
    threshold: float,
    test_size: float,
    random_state: int
) -> dict:
    """
    Prepara os dados para o treinamento do modelo.
    
    Args:
        df: DataFrame com os dados
        threshold: Limite de compra para definir Big Spender
        test_size: Proporção do conjunto de teste
        random_state: Seed para reprodutibilidade
        
    Returns:
        Dicionário com X_train, X_test, y_train, y_test
    """
    # 1. Feature Engineering e Target
    df['is_big_spender'] = (df['Purchase Amount (USD)'] > threshold).astype(int)
    
    # 2. Seleção de Features
    features = ['Age', 'Gender', 'Category', 'Season', 'Location']
    
    # 3. One-Hot Encoding para variáveis categóricas
    df_model = pd.get_dummies(df[features], drop_first=True)
    
    # 4. Target
    y = df['is_big_spender']
    X = df_model
    
    # 5. Validação de dados
    if len(y) < 10:
        return None
    if y.sum() < 2:
        return None # Mínimo de 2 amostras positivas
    
    # 6. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y # Garante proporção de classes
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

def treinar_modelo_big_spender(
    X_train: pd.DataFrame, 
    y_train: pd.Series,
    n_estimators: int,
    max_depth: int,
    learning_rate: float
) -> LGBMClassifier:
    """
    Treina o modelo LightGBM.
    
    Args:
        X_train: Features de treino
        y_train: Target de treino
        n_estimators: Número de estimadores
        max_depth: Profundidade máxima
        learning_rate: Taxa de aprendizado
        
    Returns:
        Modelo LGBMClassifier treinado
    """
    model = LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        verbose=-1 # Desliga logs
    )
    model.fit(X_train, y_train)
    return model

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
