"""
Dashboard de Análise de Comportamento de Compra
Versão 2.0 - Refatorada e Otimizada
"""
import streamlit as st
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

# Imports dos módulos
import settings
DASHBOARD_CONFIG = settings.DASHBOARD_CONFIG
MODEL_CONFIG = settings.MODEL_CONFIG
from data_loader import load_and_validate_data, calcular_estatisticas_gerais
from data_processor import aplicar_filtros, calcular_estatisticas_filtradas
from formatters import formatar_moeda, formatar_numero, formatar_percentual
from sidebar import criar_sidebar
from questions import render_questions

# ===========================
# CONFIGURAÇÃO DA PÁGINA
# ===========================
st.set_page_config(
    page_title=DASHBOARD_CONFIG.PAGE_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=DASHBOARD_CONFIG.PAGE_ICON
)

# ===========================
# CSS CUSTOMIZADO
# ===========================
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stButton>button {
        border-radius: 0.5rem;
        font-weight: 600;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ===========================
# HEADER
# ===========================
st.markdown(
    f'<p class="main-header">{DASHBOARD_CONFIG.PAGE_ICON} Dashboard de Comportamento de Compra</p>', 
    unsafe_allow_html=True
)
st.markdown("**Análise interativa do dataset Customer Shopping Behavior**")

# ===========================
# CARREGAMENTO DE DADOS
# ===========================
try:
    df = load_and_validate_data(
        DASHBOARD_CONFIG.CSV_PATH,
        DASHBOARD_CONFIG.REQUIRED_COLUMNS
    )
except (FileNotFoundError, ValueError) as e:
    st.error(str(e))
    st.stop()

# ===========================
# ESTATÍSTICAS GERAIS
# ===========================
stats_gerais = calcular_estatisticas_gerais(df)

st.markdown("###  Visão Geral do Dataset")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total de Clientes", 
        formatar_numero(stats_gerais['total_clientes']),
        help="Número total de registros no dataset"
    )

with col2:
    st.metric(
        "Ticket Médio", 
        formatar_moeda(stats_gerais['ticket_medio']),
        help="Valor médio gasto por transação"
    )

with col3:
    st.metric(
        "Valor Total", 
        formatar_moeda(stats_gerais['valor_total'], decimais=0),
        help="Soma total de todas as compras"
    )

with col4:
    st.metric(
        "Categorias", 
        stats_gerais['categorias'],
        help="Número de categorias de produtos"
    )

# ===========================
# SIDEBAR COM FILTROS
# ===========================
filtros = criar_sidebar(df)

# Validação de ação
if 'aplicar_filtro' not in st.session_state:
    st.session_state.aplicar_filtro = False

if not st.session_state.aplicar_filtro:
    st.info(
        " **Dica:** Selecione os filtros desejados na barra lateral "
        "e clique em **'Aplicar'** para visualizar as análises."
    )
    st.stop()

# ===========================
# APLICAR FILTROS
# ===========================
df_filtrado = aplicar_filtros(
    df,
    filtros['categorias'],
    filtros['generos'],
    filtros['faixa_etaria'],
    filtros['estacoes']
)

# Verificação de dados
if len(df_filtrado) == 0:
    st.error(
        "⚠️ Nenhum dado encontrado com os filtros selecionados. "
        "Por favor, ajuste os filtros na barra lateral."
    )
    st.stop()

# ===========================
# ESTATÍSTICAS FILTRADAS
# ===========================
stats_filtradas = calcular_estatisticas_filtradas(df, df_filtrado)

st.markdown("---")
st.subheader(" Visão Geral dos Dados Filtrados")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Clientes Filtrados",
        formatar_numero(stats_filtradas['clientes']),
        delta=f"{stats_filtradas['delta_clientes_pct']:.1f}%",
        help="Quantidade de clientes após aplicar os filtros"
    )

with col2:
    st.metric(
        "Ticket Médio",
        formatar_moeda(stats_filtradas['ticket_medio']),
        delta=formatar_moeda(stats_filtradas['delta_ticket']),
        help="Valor médio gasto por transação (filtrado)"
    )

with col3:
    st.metric(
        "Valor Total",
        formatar_moeda(stats_filtradas['valor_total'], decimais=0),
        help="Soma total das compras filtradas"
    )

with col4:
    st.metric(
        "Compra Máxima",
        formatar_moeda(stats_filtradas['valor_maximo']),
        help="Maior valor de compra nos dados filtrados"
    )

# ===========================
# 7 PERGUNTAS DE NEGÓCIO
# ===========================
st.markdown("---")
st.header(" Respostas às 7 Perguntas de Negócio")

# Renderizar todas as perguntas
render_questions(df_filtrado)

# ===========================
# EXPORTAR DADOS
# ===========================
st.markdown("---")
st.subheader(" Exportar Dados Filtrados")

col1, col2 = st.columns([3, 1])

with col1:
    st.info(
        f" {formatar_numero(len(df_filtrado))} registros prontos para exportação"
    )

with col2:
    csv = df_filtrado.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download CSV",
        data=csv,
        file_name="dados_filtrados.csv",
        mime="text/csv",
        use_container_width=True
    )

# ===========================
# RODAPÉ
# ===========================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    Desenvolvido com  usando Streamlit | Dataset: Customer Shopping Behavior<br>
    <small>Dashboard v2.0 - Análise Avançada de Varejo | Código Refatorado e Otimizado</small>
</div>
""", unsafe_allow_html=True)
