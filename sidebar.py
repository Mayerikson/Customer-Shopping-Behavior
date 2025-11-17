import streamlit as st

def render_sidebar(df):
    """
    Renderiza a barra lateral com filtros e perguntas de negócio
    """
    st.sidebar.title("🎯 Dashboard de Análise")
    st.sidebar.markdown("---")
    
    # Seção de Perguntas de Negócio - FIXA E SEMPRE VISÍVEL
    st.sidebar.markdown("### 📊 Respostas às 7 Perguntas de Negócio")
    
    st.sidebar.markdown("""
    <div style='background-color: #f0f8ff; padding: 15px; border-radius: 10px; border-left: 4px solid #1f77b4;'>
    
    <strong>1️⃣</strong> Qual a probabilidade de um cliente ser Big Spender?
    
    <strong>2️⃣</strong> Quais são os segmentos naturais de consumidores?
    
    <strong>3️⃣</strong> Em quais estações e locais as vendas são mais intensas?
    
    <strong>4️⃣</strong> Quais categorias geram maior valor médio por transação?
    
    <strong>5️⃣</strong> Qual a persona ideal para campanhas de alto valor?
    
    <strong>6️⃣</strong> Como características do cliente se relacionam com valor gasto?
    
    <strong>7️⃣</strong> Modelo Preditivo: Quem são os futuros Big Spenders?
    
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Filtros de Análise")
    st.sidebar.caption("Ajuste os filtros para segmentar sua análise:")
    
    # Filtro de Categoria
    categories = sorted(df['Category'].unique().tolist())
    selected_categories = st.sidebar.multiselect(
        '🏷️ Categoria de Produto',
        options=categories,
        default=categories,
        help="Selecione uma ou mais categorias"
    )
    
    # Filtro de Gênero
    genders = sorted(df['Gender'].unique().tolist())
    selected_genders = st.sidebar.multiselect(
        '👤 Gênero',
        options=genders,
        default=genders,
        help="Selecione um ou mais gêneros"
    )
    
    # Filtro de Faixa Etária
    min_age = int(df['Age'].min())
    max_age = int(df['Age'].max())
    age_range = st.sidebar.slider(
        '📅 Faixa Etária',
        min_value=min_age,
        max_value=max_age,
        value=(min_age, max_age),
        help="Ajuste a faixa etária desejada"
    )
    
    # Filtro de Estação do Ano
    if 'Season' in df.columns:
        seasons = sorted(df['Season'].unique().tolist())
        selected_seasons = st.sidebar.multiselect(
            '🌦️ Estação do Ano',
            options=seasons,
            default=seasons,
            help="Selecione uma ou mais estações"
        )
    else:
        selected_seasons = None
    
    # Filtro de Localização
    if 'Location' in df.columns:
        locations = sorted(df['Location'].unique().tolist())
        # Limitar a 10 localizações por padrão para não sobrecarregar
        default_locations = locations[:min(10, len(locations))]
        selected_locations = st.sidebar.multiselect(
            '📍 Localização',
            options=locations,
            default=default_locations,
            help="Selecione uma ou mais localizações"
        )
    else:
        selected_locations = None
    
    # Filtro de Método de Pagamento
    if 'Payment Method' in df.columns:
        payment_methods = sorted(df['Payment Method'].unique().tolist())
        selected_payments = st.sidebar.multiselect(
            '💳 Método de Pagamento',
            options=payment_methods,
            default=payment_methods,
            help="Selecione um ou mais métodos"
        )
    else:
        selected_payments = None
    
    st.sidebar.markdown("---")
    
    # Botões de ação
    col1, col2 = st.sidebar.columns(2)
    with col1:
        apply_filters = st.button('✅ Aplicar', use_container_width=True, type="primary")
    with col2:
        clear_filters = st.button('🔄 Limpar', use_container_width=True)
    
    # Informações do dataset
    st.sidebar.markdown("---")
    total_revenue = df['Purchase Amount (USD)'].sum() if 'Purchase Amount (USD)' in df.columns else 0
    avg_ticket = df['Purchase Amount (USD)'].mean() if 'Purchase Amount (USD)' in df.columns else 0
    
    st.sidebar.info(f"""
    📊 **Estatísticas Gerais:**
    
    📈 Registros: **{len(df):,}**
    
    👥 Clientes: **{df['Customer ID'].nunique() if 'Customer ID' in df.columns else 'N/A'}**
    
    💰 Receita Total: **${total_revenue:,.0f}**
    
    💵 Ticket Médio: **${avg_ticket:.2f}**
    
    🏷️ Categorias: **{df['Category'].nunique()}**
    """)
    
    # Retornar filtros aplicados
    filters = {
        'categories': selected_categories if selected_categories else categories,
        'genders': selected_genders if selected_genders else genders,
        'age_range': age_range,
        'seasons': selected_seasons if selected_seasons else (seasons if 'Season' in df.columns else None),
        'locations': selected_locations if selected_locations else (locations if 'Location' in df.columns else None),
        'payments': selected_payments if selected_payments else (payment_methods if 'Payment Method' in df.columns else None),
        'apply': apply_filters,
        'clear': clear_filters
    }
    
    return filters


def apply_filters_to_dataframe(df, filters):
    """
    Aplica os filtros selecionados ao dataframe
    """
    filtered_df = df.copy()
    
    # Aplicar filtro de categoria
    if filters['categories']:
        filtered_df = filtered_df[filtered_df['Category'].isin(filters['categories'])]
    
    # Aplicar filtro de gênero
    if filters['genders']:
        filtered_df = filtered_df[filtered_df['Gender'].isin(filters['genders'])]
    
    # Aplicar filtro de idade
    filtered_df = filtered_df[
        (filtered_df['Age'] >= filters['age_range'][0]) & 
        (filtered_df['Age'] <= filters['age_range'][1])
    ]
    
    # Aplicar filtro de estação
    if filters['seasons'] and 'Season' in df.columns:
        filtered_df = filtered_df[filtered_df['Season'].isin(filters['seasons'])]
    
    # Aplicar filtro de localização
    if filters['locations'] and 'Location' in df.columns:
        filtered_df = filtered_df[filtered_df['Location'].isin(filters['locations'])]
    
    # Aplicar filtro de método de pagamento
    if filters['payments'] and 'Payment Method' in df.columns:
        filtered_df = filtered_df[filtered_df['Payment Method'].isin(filters['payments'])]
    
    return filtered_df
