import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def answer_business_questions(df):
    """
    Responde às 7 perguntas estratégicas de negócio
    """
    
    st.markdown("## 🎯 Respostas às 7 Perguntas de Negócio")
    st.markdown("---")
    
    # Pergunta 1: Qual a probabilidade de um cliente ser Big Spender?
    with st.expander("1️⃣ Qual a probabilidade de um cliente ser Big Spender?", expanded=True):
        st.markdown("### Análise de Big Spenders")
        
        if 'Purchase Amount (USD)' in df.columns:
            # Definir Big Spender como top 20% em gastos
            threshold_80 = df['Purchase Amount (USD)'].quantile(0.80)
            df['Is_Big_Spender'] = df['Purchase Amount (USD)'] >= threshold_80
            
            big_spender_prob = (df['Is_Big_Spender'].sum() / len(df)) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Probabilidade Geral", f"{big_spender_prob:.1f}%")
            with col2:
                st.metric("Threshold Big Spender", f"${threshold_80:.2f}")
            with col3:
                st.metric("Total Big Spenders", f"{df['Is_Big_Spender'].sum():,}")
            
            # Probabilidade por características
            if 'Gender' in df.columns:
                gender_prob = df.groupby('Gender')['Is_Big_Spender'].mean() * 100
                
                fig = px.bar(
                    x=gender_prob.index,
                    y=gender_prob.values,
                    title='Probabilidade de ser Big Spender por Gênero',
                    labels={'x': 'Gênero', 'y': 'Probabilidade (%)'},
                    color=gender_prob.values,
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            if 'Category' in df.columns:
                category_prob = df.groupby('Category')['Is_Big_Spender'].mean().sort_values(ascending=False) * 100
                
                fig = px.bar(
                    x=category_prob.index,
                    y=category_prob.values,
                    title='Probabilidade de ser Big Spender por Categoria',
                    labels={'x': 'Categoria', 'y': 'Probabilidade (%)'},
                    color=category_prob.values,
                    color_continuous_scale='Greens'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"""
            ✅ **Resposta:** A probabilidade geral de um cliente ser Big Spender é de **{big_spender_prob:.1f}%**. 
            Clientes que gastam acima de **${threshold_80:.2f}** são classificados como Big Spenders.
            """)
    
    st.markdown("---")
    
    # Pergunta 2: Quais são os segmentos naturais de consumidores?
    with st.expander("2️⃣ Quais são os segmentos naturais de consumidores?", expanded=False):
        st.markdown("### Segmentação de Clientes (K-Means Clustering)")
        
        if all(col in df.columns for col in ['Customer ID', 'Purchase Amount (USD)', 'Age']):
            # Agregar dados por cliente
            customer_data = df.groupby('Customer ID').agg({
                'Purchase Amount (USD)': ['sum', 'mean', 'count'],
                'Age': 'first'
            })
            customer_data.columns = ['Total_Gasto', 'Ticket_Medio', 'Frequencia', 'Idade']
            
            # Preparar dados para clustering
            X = customer_data[['Total_Gasto', 'Frequencia', 'Ticket_Medio']].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Aplicar K-Means com 4 clusters
            n_clusters = 4
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            customer_data['Cluster'] = kmeans.fit_predict(X_scaled)
            
            # Nomear clusters
            cluster_names = {
                0: "Clientes Ocasionais",
                1: "Clientes Regulares",
                2: "Clientes Premium",
                3: "Clientes VIP"
            }
            
            # Ordenar clusters por valor médio
            cluster_order = customer_data.groupby('Cluster')['Total_Gasto'].mean().sort_values().index
            cluster_mapping = {old: cluster_names[new] for new, old in enumerate(cluster_order)}
            customer_data['Segmento'] = customer_data['Cluster'].map(cluster_mapping)
            
            # Visualização 3D
            fig = px.scatter_3d(
                customer_data.reset_index(),
                x='Total_Gasto',
                y='Frequencia',
                z='Ticket_Medio',
                color='Segmento',
                title='Segmentação de Clientes em 3D',
                labels={
                    'Total_Gasto': 'Gasto Total (USD)',
                    'Frequencia': 'Frequência de Compras',
                    'Ticket_Medio': 'Ticket Médio (USD)'
                },
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Características de cada segmento
            segment_stats = customer_data.groupby('Segmento').agg({
                'Total_Gasto': 'mean',
                'Ticket_Medio': 'mean',
                'Frequencia': 'mean',
                'Idade': 'mean'
            }).round(2)
            
            segment_stats['Quantidade'] = customer_data.groupby('Segmento').size()
            
            st.markdown("#### Características dos Segmentos")
            st.dataframe(segment_stats.style.format({
                'Total_Gasto': '${:,.2f}',
                'Ticket_Medio': '${:,.2f}',
                'Frequencia': '{:.1f}',
                'Idade': '{:.0f}',
                'Quantidade': '{:,.0f}'
            }), use_container_width=True)
            
            st.success(f"""
            ✅ **Resposta:** Identificamos **{n_clusters} segmentos naturais** de consumidores:
            {', '.join(cluster_names.values())}
            """)
    
    st.markdown("---")
    
    # Pergunta 3: Em quais estações e locais as vendas são mais intensas?
    with st.expander("3️⃣ Em quais estações e locais as vendas são mais intensas?", expanded=False):
        st.markdown("### Análise de Sazonalidade e Localização")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Season' in df.columns and 'Purchase Amount (USD)' in df.columns:
                seasonal_sales = df.groupby('Season').agg({
                    'Purchase Amount (USD)': 'sum',
                    'Customer ID': 'count'
                }).round(2)
                seasonal_sales.columns = ['Total_Vendas', 'Num_Transacoes']
                seasonal_sales = seasonal_sales.sort_values('Total_Vendas', ascending=False)
                
                fig = px.pie(
                    seasonal_sales.reset_index(),
                    values='Total_Vendas',
                    names='Season',
                    title='Distribuição de Vendas por Estação',
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)
                
                best_season = seasonal_sales.index[0]
                best_season_value = seasonal_sales.iloc[0]['Total_Vendas']
        
        with col2:
            if 'Location' in df.columns and 'Purchase Amount (USD)' in df.columns:
                location_sales = df.groupby('Location').agg({
                    'Purchase Amount (USD)': 'sum',
                    'Customer ID': 'count'
                }).round(2)
                location_sales.columns = ['Total_Vendas', 'Num_Transacoes']
                location_sales = location_sales.sort_values('Total_Vendas', ascending=False).head(10)
                
                fig = px.bar(
                    location_sales.reset_index(),
                    x='Total_Vendas',
                    y='Location',
                    orientation='h',
                    title='Top 10 Localizações por Vendas',
                    labels={'Total_Vendas': 'Vendas (USD)', 'Location': 'Local'},
                    color='Total_Vendas',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                best_location = location_sales.index[0]
                best_location_value = location_sales.iloc[0]['Total_Vendas']
        
        # Mapa de calor: Estação x Localização
        if 'Season' in df.columns and 'Location' in df.columns:
            heatmap_data = df.pivot_table(
                values='Purchase Amount (USD)',
                index='Location',
                columns='Season',
                aggfunc='sum',
                fill_value=0
            ).head(15)
            
            fig = px.imshow(
                heatmap_data,
                title='Mapa de Calor: Vendas por Localização e Estação',
                labels=dict(x="Estação", y="Localização", color="Vendas (USD)"),
                aspect="auto",
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.success(f"""
        ✅ **Resposta:** 
        - **Melhor Estação:** {best_season} com ${best_season_value:,.0f} em vendas
        - **Melhor Localização:** {best_location} com ${best_location_value:,.0f} em vendas
        """)
    
    st.markdown("---")
    
    # Pergunta 4: Quais categorias geram maior valor médio por transação?
    with st.expander("4️⃣ Quais categorias geram maior valor médio por transação?", expanded=False):
        st.markdown("### Valor Médio por Categoria")
        
        if 'Category' in df.columns and 'Purchase Amount (USD)' in df.columns:
            category_stats = df.groupby('Category').agg({
                'Purchase Amount (USD)': ['mean', 'sum', 'count', 'std']
            }).round(2)
            category_stats.columns = ['Valor_Medio', 'Total', 'Quantidade', 'Desvio_Padrao']
            category_stats = category_stats.sort_values('Valor_Medio', ascending=False)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.bar(
                    category_stats.reset_index(),
                    x='Category',
                    y='Valor_Medio',
                    title='Valor Médio por Transação por Categoria',
                    labels={'Valor_Medio': 'Valor Médio (USD)', 'Category': 'Categoria'},
                    color='Valor_Medio',
                    color_continuous_scale='Sunset',
                    text='Valor_Medio'
                )
                fig.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.dataframe(category_stats.style.format({
                    'Valor_Medio': '${:,.2f}',
                    'Total': '${:,.0f}',
                    'Quantidade': '{:,.0f}',
                    'Desvio_Padrao': '${:,.2f}'
                }), use_container_width=True)
            
            top_category = category_stats.index[0]
            top_value = category_stats.iloc[0]['Valor_Medio']
            
            st.success(f"""
            ✅ **Resposta:** A categoria **{top_category}** gera o maior valor médio por transação: 
            **${top_value:.2f}**
            """)
    
    st.markdown("---")
    
    # Pergunta 5: Qual a persona ideal para campanhas de alto valor?
    with st.expander("5️⃣ Qual a persona ideal para campanhas de alto valor?", expanded=False):
        st.markdown("### Persona de Alto Valor")
        
        if 'Purchase Amount (USD)' in df.columns:
            # Definir alto valor como top 10%
            high_value_threshold = df['Purchase Amount (USD)'].quantile(0.90)
            high_value_customers = df[df['Purchase Amount (USD)'] >= high_value_threshold]
            
            st.metric("Threshold Alto Valor", f"${high_value_threshold:.2f}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'Gender' in high_value_customers.columns:
                    gender_dist = high_value_customers['Gender'].value_counts()
                    st.markdown("**Gênero Predominante:**")
                    st.write(gender_dist)
                    top_gender = gender_dist.index[0]
            
            with col2:
                if 'Age' in high_value_customers.columns:
                    avg_age = high_value_customers['Age'].mean()
                    st.metric("Idade Média", f"{avg_age:.0f} anos")
                    age_range = f"{high_value_customers['Age'].quantile(0.25):.0f}-{high_value_customers['Age'].quantile(0.75):.0f}"
            
            with col3:
                if 'Category' in high_value_customers.columns:
                    top_category_hv = high_value_customers['Category'].mode()[0]
                    st.markdown("**Categoria Favorita:**")
                    st.write(top_category_hv)
            
            # Visualização da persona
            if 'Season' in high_value_customers.columns:
                season_pref = high_value_customers['Season'].value_counts().head(2)
                fig = px.bar(
                    x=season_pref.index,
                    y=season_pref.values,
                    title='Estações Preferidas - Clientes Alto Valor',
                    labels={'x': 'Estação', 'y': 'Frequência'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"""
            ✅ **Resposta - Persona Ideal para Campanhas de Alto Valor:**
            - **Gênero:** {top_gender}
            - **Faixa Etária:** {age_range} anos (média {avg_age:.0f})
            - **Categoria Preferida:** {top_category_hv}
            - **Gasto Mínimo:** ${high_value_threshold:.2f}
            """)
    
    st.markdown("---")
    
    # Pergunta 6: Como características do cliente se relacionam com valor gasto?
    with st.expander("6️⃣ Como características do cliente se relacionam com valor gasto?", expanded=False):
        st.markdown("### Análise de Correlações")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Age' in df.columns and 'Purchase Amount (USD)' in df.columns:
                fig = px.scatter(
                    df.sample(min(1000, len(df))),
                    x='Age',
                    y='Purchase Amount (USD)',
                    color='Gender' if 'Gender' in df.columns else None,
                    title='Relação: Idade vs Valor Gasto',
                    trendline="lowess",
                    labels={'Age': 'Idade', 'Purchase Amount (USD)': 'Valor Gasto (USD)'}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                correlation_age = df[['Age', 'Purchase Amount (USD)']].corr().iloc[0, 1]
                st.info(f"Correlação Idade x Valor: **{correlation_age:.3f}**")
        
        with col2:
            if 'Gender' in df.columns and 'Purchase Amount (USD)' in df.columns:
                fig = px.box(
                    df,
                    x='Gender',
                    y='Purchase Amount (USD)',
                    title='Distribuição de Gastos por Gênero',
                    labels={'Gender': 'Gênero', 'Purchase Amount (USD)': 'Valor Gasto (USD)'},
                    color='Gender'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Análise por categoria e gênero
        if all(col in df.columns for col in ['Category', 'Gender', 'Purchase Amount (USD)']):
            category_gender = df.groupby(['Category', 'Gender'])['Purchase Amount (USD)'].mean().reset_index()
            
            fig = px.bar(
                category_gender,
                x='Category',
                y='Purchase Amount (USD)',
                color='Gender',
                barmode='group',
                title='Valor Médio por Categoria e Gênero',
                labels={'Purchase Amount (USD)': 'Valor Médio (USD)', 'Category': 'Categoria'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.success("""
        ✅ **Resposta:** As características dos clientes se relacionam com o valor gasto de formas variadas.
        Idade, gênero e preferências de categoria são fatores importantes na determinação do ticket de compra.
        """)
    
    st.markdown("---")
    
    # Pergunta 7: Modelo Preditivo - Quem são os futuros Big Spenders?
    with st.expander("7️⃣ Modelo Preditivo: Quem são os futuros Big Spenders?", expanded=False):
        st.markdown("### Previsão de Big Spenders")
        
        st.info("""
        📊 **Modelo de Machine Learning para Identificação de Potenciais Big Spenders**
        
        Este modelo utiliza características dos clientes para prever a probabilidade de se tornarem Big Spenders.
        """)
        
        if all(col in df.columns for col in ['Customer ID', 'Purchase Amount (USD)', 'Age']):
            # Agregar por cliente
            customer_features = df.groupby('Customer ID').agg({
                'Purchase Amount (USD)': ['sum', 'mean', 'count'],
                'Age': 'first',
                'Gender': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown',
                'Category': lambda x: x.mode()[0] if not x.mode().empty else 'Unknown'
            })
            
            customer_features.columns = ['Total_Gasto', 'Ticket_Medio', 'Frequencia', 'Idade', 'Genero', 'Categoria_Fav']
            
            # Definir Big Spender
            threshold = customer_features['Total_Gasto'].quantile(0.75)
            customer_features['Is_Big_Spender'] = (customer_features['Total_Gasto'] >= threshold).astype(int)
            
            # Score de propensão (simplificado)
            customer_features['Propensity_Score'] = (
                (customer_features['Frequencia'] / customer_features['Frequencia'].max()) * 0.4 +
                (customer_features['Ticket_Medio'] / customer_features['Ticket_Medio'].max()) * 0.4 +
                (customer_features['Total_Gasto'] / customer_features['Total_Gasto'].max()) * 0.2
            ) * 100
            
            # Top potenciais Big Spenders
            potential_big_spenders = customer_features[
                (customer_features['Is_Big_Spender'] == 0) & 
                (customer_features['Propensity_Score'] > 50)
            ].sort_values('Propensity_Score', ascending=False).head(20)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### Top 20 Potenciais Big Spenders")
                st.dataframe(potential_big_spenders[['Total_Gasto', 'Frequencia', 'Ticket_Medio', 'Propensity_Score']].style.format({
                    'Total_Gasto': '${:,.2f}',
                    'Frequencia': '{:.0f}',
                    'Ticket_Medio': '${:,.2f}',
                    'Propensity_Score': '{:.1f}%'
                }), use_container_width=True)
            
            with col2:
                st.metric("Potenciais Identificados", len(potential_big_spenders))
                st.metric("Score Médio", f"{potential_big_spenders['Propensity_Score'].mean():.1f}%")
                st.metric("Receita Potencial", f"${potential_big_spenders['Total_Gasto'].sum():,.0f}")
            
            # Distribuição de scores
            fig = px.histogram(
                customer_features,
                x='Propensity_Score',
                color='Is_Big_Spender',
                title='Distribuição de Score de Propensão',
                labels={'Propensity_Score': 'Score de Propensão (%)', 'Is_Big_Spender': 'Big Spender Atual'},
                nbins=50,
                barmode='overlay'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"""
            ✅ **Resposta:** Identificamos **{len(potential_big_spenders)} clientes** com alto potencial
            de se tornarem Big Spenders, com score médio de propensão de **{potential_big_spenders['Propensity_Score'].mean():.1f}%**.
            """)

if __name__ == "__main__":
    st.title("Análise de Perguntas de Negócio")
    st.write("Execute este módulo através do app principal.")
