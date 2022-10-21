#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas            as pd
import streamlit         as st
import seaborn           as sns
import matplotlib.pyplot as plt
import numpy as np

from scipy.cluster.hierarchy import linkage, fcluster
from gower import gower_matrix
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import squareform

from PIL                 import Image
from io                  import BytesIO

# Set no tema do seaborn para melhorar o visual dos plots
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

# Função principal da aplicação
def main():
    # Configuração inicial da página da aplicação
    st.set_page_config(page_title = 'Agrupamento hierárquico',         layout="wide",
        initial_sidebar_state='expanded'
    )

    # Título principal da aplicação
    st.write('# Agrupamento hierárquico')
    st.markdown("---")

    st.write('Objetivo: agrupar as sessões de acesso ao portal considerando o comportamento de acesso e informações da data, como a proximidade a uma data especial, fim de semana e o mês.')

    # Botão para carregar arquivo na aplicação
    st.sidebar.write("## Suba o arquivo")
    data_file_1 = st.sidebar.file_uploader("Online shoppers intention data", type = ['csv'])

    # Verifica se há conteúdo carregado na aplicação
    if (data_file_1 is not None):
        df_raw = pd.read_csv(data_file_1)
        df = df_raw.copy()

        st.write('## Informações da base de dados')
        st.write(df.head(2))

        # Variáveis

        variaveis = ['Administrative', 'Administrative_Duration', 'Informational', 
             'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration', 
             'SpecialDay', 'Month', 'Weekend']

        variaveis_qtd = ['Administrative', 'Administrative_Duration', 'Informational', 
             'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration']

        variaveis_cat = ['SpecialDay', 'Month', 'Weekend']

        groups = [0, 2, 3, 4, 5, 6, 7]

        with st.sidebar.form(key='my_form'):

            # SELECIONA AS VARIÁVEIS PARA GRÁFICO E A QUANTIDADE DE GRUPOS PARA O AGRUPAMENTO

            graph_vars = st.multiselect('Deseja ver a distribuição gráfica de quais variáveis?', variaveis)

            groups_vars = st.selectbox('Quantos grupos deseja analisar?', groups)

            st.form_submit_button()
          
        # PLOTS

        if any('Administrative' == s for s in graph_vars):
            fig = plt.figure(figsize=(8, 3))    
            sns.histplot(data=df, x = "Administrative", discrete=True).set_title("Administrative Count")
            st.pyplot(fig)
            
        if any('Administrative_Duration' == s for s in graph_vars):
            fig = plt.figure(figsize=(8, 3))
            sns.histplot(data=df, x = "Administrative_Duration").set_title("Administrative_Duration Count")
            st.pyplot(fig)

        if any('Informational' == s for s in graph_vars):
            fig = plt.figure(figsize=(8, 3))
            sns.histplot(data=df, x = "Informational", discrete=True).set_title("Informational Count")
            st.pyplot(fig)
            
        if any('Informational_Duration' == s for s in graph_vars):
            fig = plt.figure(figsize=(8, 3))
            sns.histplot(data=df, x = "Informational_Duration", bins=50).set_title("Informational_Duration Count")
            st.pyplot(fig)
            
        if any('ProductRelated' == s for s in graph_vars):
            fig = plt.figure(figsize=(8, 3))
            sns.histplot(data=df, x = "ProductRelated", discrete=True).set_title("ProductRelated Count")
            st.pyplot(fig)
            
        if any('ProductRelated_Duration' == s for s in graph_vars):
            fig = plt.figure(figsize=(8, 3))
            sns.histplot(data=df, x = "ProductRelated_Duration").set_title("ProductRelated_Duration Count")
            st.pyplot(fig)
            
        if any('SpecialDay' == s for s in graph_vars):
            fig = plt.figure(figsize=(8, 3))
            sns.histplot(data=df, x = "SpecialDay").set_title("SpecialDay Count")
            st.pyplot(fig)
            
        if any('Month' == s for s in graph_vars):
            fig = plt.figure(figsize=(8, 3))
            sns.histplot(data=df, x = "Month").set_title("Month Count")
            st.pyplot(fig)
            
        if any('Weekend' == s for s in graph_vars):
            fig = plt.figure(figsize=(8, 3))
            sns.countplot(data=df, x = 'Weekend').set_title("Weekend Count")
            st.pyplot(fig)

        # Padronização dos Valores Quantitativos

        df_pad = pd.DataFrame(StandardScaler().fit_transform(df[variaveis_qtd]), columns = df[variaveis_qtd].columns)

        # Criação de Dummires

        df_pad[variaveis_cat] = df[variaveis_cat]
        df2 = pd.get_dummies(df_pad[variaveis].dropna(), columns=variaveis_cat)

        # Cálculo das Distâncias Gower
        
        vars_cat = [True if x in {'SpecialDay_0.0', 'SpecialDay_0.2', 'SpecialDay_0.4', 'SpecialDay_0.6', 'SpecialDay_0.8',
            'SpecialDay_1.0', 'Month_Aug', 'Month_Dec', 'Month_Feb', 'Month_Jul', 'Month_June', 'Month_Mar', 'Month_May', 'Month_Nov',
            'Month_Oct', 'Month_Sep', 'Weekend_False', 'Weekend_True'} else False for x in df2.columns]

    
        distancia_gower = gower_matrix(df2, cat_features=vars_cat)
        gdv = squareform(distancia_gower,force='tovector')

        # Treinar Agrupamento

        Z = linkage(gdv, method='complete')

        if groups_vars == 2:
            df2['grupos_2'] = fcluster(Z, 2, criterion='maxclust')
            df3 = df.join(df2['grupos_2'], how='left')
            df3['grupos_2'].replace({1:"grupo_1", 2:"grupo_2"}, inplace=True)
            st.write('Crosstab da quantidade de valores em cada grupo:')
            st.write(pd.crosstab(df3.Revenue, df3.grupos_2))
            st.write('Crosstab da quantidade normalizada de valores em cada grupo:')
            st.write(pd.crosstab(df3.Revenue, df3.grupos_2, normalize='columns'))
            st.write('Bounce Rates:')
            fig = plt.figure(figsize=(8, 3))
            sns.boxplot(data=df3, y='grupos_2', x='BounceRates')
            st.pyplot(fig) 
            st.write('Agrupamentos em relação a informações de Data:')         
            fig, axis = plt.subplots(3, 1, figsize=(8,12))
            sns.countplot(data=df3, x = "SpecialDay", hue='grupos_2', ax=axis[0])
            sns.countplot(data=df3, x = "Month", hue='grupos_2', ax=axis[1])
            sns.countplot(data=df3, x = "Weekend", hue='grupos_2', ax=axis[2])
            st.pyplot(fig)
        
        if groups_vars == 3:
            df2['grupos_3'] = fcluster(Z, 3, criterion='maxclust')
            df3 = df.join(df2['grupos_3'], how='left')
            df3['grupos_3'].replace({1:"grupo_1", 2:"grupo_2", 3:"grupo_3"}, inplace=True)
            st.write('Crosstab da quantidade de valores em cada grupo:')
            st.write(pd.crosstab(df3.Revenue, df3.grupos_3))
            st.write('Crosstab da quantidade normalizada de valores em cada grupo:')
            st.write(pd.crosstab(df3.Revenue, df3.grupos_3, normalize='columns'))
            st.write('Bounce Rates:')
            fig = plt.figure(figsize=(8, 3))
            sns.boxplot(data=df3, y='grupos_3', x='BounceRates')
            st.pyplot(fig) 
            st.write('Agrupamentos em relação a informações de Data:')         
            fig, axis = plt.subplots(3, 1, figsize=(8,12))
            sns.countplot(data=df3, x = "SpecialDay", hue='grupos_3', ax=axis[0])
            sns.countplot(data=df3, x = "Month", hue='grupos_3', ax=axis[1])
            sns.countplot(data=df3, x = "Weekend", hue='grupos_3', ax=axis[2])
            st.pyplot(fig)

        if groups_vars == 4:
            df2['grupos_4'] = fcluster(Z, 4, criterion='maxclust')
            df3 = df.join(df2['grupos_4'], how='left')
            df3['grupos_4'].replace({1:"grupo_1", 2:"grupo_2", 3:"grupo_3", 4:"grupo_4"}, inplace=True)
            st.write('Crosstab da quantidade de valores em cada grupo:')
            st.write(pd.crosstab(df3.Revenue, df3.grupos_4))
            st.write('Crosstab da quantidade normalizada de valores em cada grupo:')
            st.write(pd.crosstab(df3.Revenue, df3.grupos_4, normalize='columns'))
            st.write('Bounce Rates:')
            fig = plt.figure(figsize=(8, 3))
            sns.boxplot(data=df3, y='grupos_4', x='BounceRates')
            st.pyplot(fig) 
            st.write('Agrupamentos em relação a informações de Data:')         
            fig, axis = plt.subplots(3, 1, figsize=(8,12))
            sns.countplot(data=df3, x = "SpecialDay", hue='grupos_4', ax=axis[0])
            sns.countplot(data=df3, x = "Month", hue='grupos_4', ax=axis[1])
            sns.countplot(data=df3, x = "Weekend", hue='grupos_4', ax=axis[2])
            st.pyplot(fig)

        if groups_vars == 5:
            df2['grupos_5'] = fcluster(Z, 5, criterion='maxclust')
            df3 = df.join(df2['grupos_5'], how='left')
            df3['grupos_5'].replace({1:"grupo_1", 2:"grupo_2", 3:"grupo_3", 4:"grupo_4", 5:"grupo_5"}, inplace=True)
            st.write('Crosstab da quantidade de valores em cada grupo:')
            st.write(pd.crosstab(df3.Revenue, df3.grupos_5))
            st.write('Crosstab da quantidade normalizada de valores em cada grupo:')
            st.write(pd.crosstab(df3.Revenue, df3.grupos_5, normalize='columns'))
            st.write('Bounce Rates:')
            fig = plt.figure(figsize=(8, 3))
            sns.boxplot(data=df3, y='grupos_5', x='BounceRates')
            st.pyplot(fig) 
            st.write('Agrupamentos em relação a informações de Data:')         
            fig, axis = plt.subplots(3, 1, figsize=(8,12))
            sns.countplot(data=df3, x = "SpecialDay", hue='grupos_5', ax=axis[0])
            sns.countplot(data=df3, x = "Month", hue='grupos_5', ax=axis[1])
            sns.countplot(data=df3, x = "Weekend", hue='grupos_5', ax=axis[2])
            st.pyplot(fig)

        if groups_vars == 6:
            df2['grupos_6'] = fcluster(Z, 6, criterion='maxclust')
            df3 = df.join(df2['grupos_6'], how='left')
            df3['grupos_6'].replace({1:"grupo_1", 2:"grupo_2", 3:"grupo_3", 4:"grupo_4", 5:"grupo_5", 6:"grupo_6"}, inplace=True)
            st.write('Crosstab da quantidade de valores em cada grupo:')
            st.write(pd.crosstab(df3.Revenue, df3.grupos_6))
            st.write('Crosstab da quantidade normalizada de valores em cada grupo:')
            st.write(pd.crosstab(df3.Revenue, df3.grupos_6, normalize='columns'))
            st.write('Bounce Rates:')
            fig = plt.figure(figsize=(8, 3))
            sns.boxplot(data=df3, y='grupos_6', x='BounceRates')
            st.pyplot(fig) 
            st.write('Agrupamentos em relação a informações de Data:')         
            fig, axis = plt.subplots(3, 1, figsize=(8,12))
            sns.countplot(data=df3, x = "SpecialDay", hue='grupos_6', ax=axis[0])
            sns.countplot(data=df3, x = "Month", hue='grupos_6', ax=axis[1])
            sns.countplot(data=df3, x = "Weekend", hue='grupos_6', ax=axis[2])
            st.pyplot(fig)

        if groups_vars == 7:
            df2['grupos_7'] = fcluster(Z, 7, criterion='maxclust')
            df3 = df.join(df2['grupos_7'], how='left')
            df3['grupos_7'].replace({1:"grupo_1", 2:"grupo_2", 3:"grupo_3", 4:"grupo_4", 5:"grupo_5", 6:"grupo_6", 7:"grupo_7"}, inplace=True)
            st.write('Crosstab da quantidade de valores em cada grupo:')
            st.write(pd.crosstab(df3.Revenue, df3.grupos_7))
            st.write('Crosstab da quantidade normalizada de valores em cada grupo:')
            st.write(pd.crosstab(df3.Revenue, df3.grupos_7, normalize='columns'))
            st.write('Bounce Rates:')
            fig = plt.figure(figsize=(8, 3))
            sns.boxplot(data=df3, y='grupos_7', x='BounceRates')
            st.pyplot(fig) 
            st.write('Agrupamentos em relação a informações de Data:')         
            fig, axis = plt.subplots(3, 1, figsize=(8,12))
            sns.countplot(data=df3, x = "SpecialDay", hue='grupos_7', ax=axis[0])
            sns.countplot(data=df3, x = "Month", hue='grupos_7', ax=axis[1])
            sns.countplot(data=df3, x = "Weekend", hue='grupos_7', ax=axis[2])
            st.pyplot(fig)
                 
        

if __name__ == '__main__':
	main()


# In[ ]:




