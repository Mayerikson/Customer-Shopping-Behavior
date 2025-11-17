# Análise Avançada para Decisão no Varejo – Customer Shopping Behavior

Este projeto transforma dados de compras em decisões estratégicas para varejistas que querem entender, segmentar e prever o comportamento dos clientes em lojas físicas e digitais.

##  Objetivo
Simular como uma rede varejista pode identificar quem são os "big spenders", segmentar clientes, otimizar campanhas de CRM e prever valor de compra utilizando apenas dados transacionais e demográficos públicos.

##  7 Perguntas de Negócio Respondidas

| #  | Pergunta                                      | Técnica                     | Insight-chave                                      |
|----|-----------------------------------------------|-----------------------------|----------------------------------------------------|
| 1  | Probabilidade de um cliente se tornar big spender? | Análise de percentil (P80) | 20% dos clientes respondem por 52% do faturamento |
| 2  | Segmentos naturais de consumidores?          | KMeans (Idade × Valor)      | 3 clusters: jovens low-cost, adultos medianos, maduros premium |
| 3  | Estações e locais com vendas mais intensas? | Agrupamento Season × Location | Inverno na Califórnia lidera em receita |
| 4  | Categorias com maior valor médio?            | ANOVA + gráfico de barras   | Footwear > Outerwear > Accessories |
| 5  | Persona ideal para campanhas de alto valor? | Filtro dos 10% top          | Mulheres 35-45, Footwear, California, Inverno |
| 6  | Relação entre idade/gênero e valor gasto?    | Correlação + scatter        | Idade levemente positiva; gênero pouca diferença |
| 7  | Prever os 20% maiores gastadores?            | LightGBM + SHAP             | Category, Season e Age são as features mais importantes |

##  Ferramentas & Tecnologias
- **Linguagem:** Python 3.11
- **Ambiente:** Jupyter Notebook → Streamlit Cloud
- **Bibliotecas:** pandas, seaborn, scikit-learn, LightGBM, SHAP, Streamlit
- **Dataset:** Customer Shopping Behavior (Kaggle – 3.900 transações, 18 atributos)

##  Estrutura do Repositório



Estrutura do Repositório:
- LICENSE
- README.md
- app.py
- charts.py
- clustering.py
- data_loader.py
- data_processor.py
- formatters.py
- prediction.py
- questions.py
- requirements.txt
- settings.py
- shopping_behavior_updated.csv
- sidebar.py
- varejo (2).ipynb





##  Plano de Execução (Estilo Scrum)

| Sprint   | Entregas                                      |
|----------|-----------------------------------------------|
| Sprint 1 | Limpeza, tipagem e EDA dos dados              |
| Sprint 2 | Respostas às 7 perguntas com modelos adaptados |
| Sprint 3 | Dashboard interativo + relatório executivo 5W2H |

##  Principais Insights
- 20% dos clientes geram 52% da receita – focar em big spenders tem alto ROI.
- Califórnia e Nova York concentraram 35% das vendas – priorizar inventário regional.
- Footwear tem ticket médio 35% superior à categoria Clothing – margem estratégica.
- Inverno amplifica o valor de compra em todas as categorias – oportunidade de precificação dinâmica.
- Modelo preditivo atinge AUC = 0,87 – pode ser usado em tempo real para personalizar ofertas.

##  Impacto Esperado
- ↑ 15% de conversão em campanhas direcionadas aos big spenders identificados.
- ↓ 10% de desperdício de mídia ao segmentar pela persona ideal.
- ↑ 8% de ticket médio via precificação sazonal por categoria e localização.

## Autor
**Mayerikson** – Cientista de Dados
[GitHub](https://github.com) | [LinkedIn](https://linkedin.com)
> "Dados não são apenas números – são decisões esperando para acontecer."
