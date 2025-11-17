import streamlit as st
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

# Importações diretas (assumindo que todos os arquivos estão na raiz)
import settings
from data_loader import load_data, get_general_stats
from data_processor import apply_filters, get_filtered_stats
from formatters import fmt_currency, fmt_number
from sidebar import render_sidebar
from questions import render_questions

# Variáveis de configuração simplificadas
CFG = settings.CFG
MODEL_CFG = settings.MODEL_CFG

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title=CFG.PAGE_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=CFG.PAGE_ICON
)

# --- CSS CUSTOMIZADO ---
st.markdown(f"""
    <style>
    .main-header {{
        font-size: 3.5rem; /* AUMENTADO DE 2.5rem PARA 3.5rem */
        font-weight: bold;
        color: {CFG.PRIMARY_COLOR};
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 0.5rem;
    }}
    .stButton>button {{
        border-radius: 0.5rem;
        font-weight: 600;
    }}
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
# Ícone removido da string de formatação
st.markdown(f'<p class="main-header">Dashboard de Comportamento de Compra</p>', unsafe_allow_html=True)
st.markdown("**Análise interativa do dataset Customer Shopping Behavior**")

# --- CARREGAMENTO DE DADOS ---
try:
    df = load_data(CFG.CSV_PATH, CFG.REQUIRED_COLS)
except (FileNotFoundError, ValueError) as e:
    st.error(str(e))
    st.stop()

# --- ESTATÍSTICAS GERAIS ---
stats_gerais = get_general_stats(df)

st.markdown("### 📈 Visão Geral do Dataset")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total de Clientes", fmt_number(stats_gerais['total_clientes']))
with col2:
    st.metric("Ticket Médio", fmt_currency(stats_gerais['ticket_medio']))
with col3:
    st.metric("Valor Total", fmt_currency(stats_gerais['valor_total'], decimals=0))
with col4:
    st.metric("Categorias", stats_gerais['categorias'])

# --- SIDEBAR COM FILTROS ---
filtros = render_sidebar(df)

if not st.session_state.aplicar_filtro:
    st.info("💡 **Dica:** Selecione os filtros desejados na barra lateral e clique em **'Aplicar'** para visualizar as análises.")
    st.stop()

# --- APLICAR FILTROS ---
df_filtrado = apply_filters(df, filtros['categorias'], filtros['generos'], filtros['faixa_etaria'], filtros['estacoes'])

if len(df_filtrado) == 0:
    st.error("⚠️ Nenhum dado encontrado com os filtros selecionados. Por favor, ajuste os filtros na barra lateral.")
    st.stop()

# --- ESTATÍSTICAS FILTRADAS ---
stats_filtradas = get_filtered_stats(df, df_filtrado)

st.markdown("---")
st.subheader("📊 Visão Geral dos Dados Filtrados")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Clientes Filtrados", fmt_number(stats_filtradas['clientes']), delta=f"{stats_filtradas['delta_clientes_pct']:.1f}%")
with col2:
    st.metric("Ticket Médio", fmt_currency(stats_filtradas['ticket_medio']), delta=fmt_currency(stats_filtradas['delta_ticket']))
with col3:
    st.metric("Valor Total", fmt_currency(stats_filtradas['valor_total'], decimals=0))
with col4:
    st.metric("Compra Máxima", fmt_currency(stats_filtradas['valor_maximo']))

# --- 7 PERGUNTAS DE NEGÓCIO ---
st.markdown("---")
st.header("🎯 Respostas às 7 Perguntas de Negócio")

render_questions(df_filtrado)

# --- EXPORTAR DADOS ---
st.markdown("---")
st.subheader("💾 Exportar Dados Filtrados")

col1, col2 = st.columns([3, 1])
with col1:
    st.info(f"📋 {fmt_number(len(df_filtrado))} registros prontos para exportação")

with col2:
    csv = df_filtrado.to_csv(index=False).encode('utf-8')
    st.download_button(label="⬇️ Download CSV", data=csv, file_name="dados_filtrados.csv", mime="text/csv", use_container_width=True)

# --- RODAPÉ ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    Desenvolvido com ❤️ usando Streamlit | Dataset: Customer Shopping Behavior  

    <small>Dashboard v2.0 - Análise Avançada de Varejo</small>
</div>
""", unsafe_allow_html=True)
