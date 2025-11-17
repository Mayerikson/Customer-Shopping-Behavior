import streamlit as st
import pandas as pd

st.title("Dashboard de Análise de Comportamento de Compra")
st.write("Bem-vindo ao dashboard interativo do projeto de varejo.")

# Carregar dataset
df = pd.read_csv("data/shopping_behavior_updated.csv")
st.dataframe(df.head())
