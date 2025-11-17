import streamlit as st
import pandas as pd

st.title("Dashboard de Análise de Comportamento de Compra")

# Arquivo está na raiz do repo
df = pd.read_csv("shopping_behavior_updated.csv")

st.dataframe(df.head())
