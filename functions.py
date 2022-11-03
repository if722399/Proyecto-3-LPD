"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Proyecto de aplicación III: Preparación y feature engineering de variables numéricas       -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: @if722399                                                                                   -- #
# -- repository: https://github.com/if722399/Proyecto-3-LPD.git                                          -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
import os
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def dqr(df):
    return pd.DataFrame(
    {
    '% of nulls':[round(df[i].isna().sum()/len(df),4) for i in df.columns],
    'unique_values':[df[i].nunique() for i in df.columns],
    'type':df.dtypes.tolist()
    },
    index=df.columns
    )

def get_numeric_stats(df):
    """
    Esta magia sacará estadísticas básicas DE LAS VARIABLES NUMÉRICAS.

    Parámetros
    ----------
    df: pandas.DataFrame
        Tabla con variables limpias.

    Regresa
    -------
    stats: diccionario
        Dict de la forma {columna: {nombre de la métrica: valor de la métrica}}
    """

    numeric_df = df.select_dtypes(include=['number'])
    
    stats = {}
    
    for numeric_columns in numeric_df.columns:
        mean = numeric_df[numeric_columns].mean()
    
        median = numeric_df[numeric_columns].median()
        
        std = numeric_df[numeric_columns].std()
        
        quantile25, quantile75 = numeric_df[numeric_columns].quantile(
            q = [.25, .75]
        )
    
        null_count = 100 * (
            numeric_df[numeric_columns].isnull().sum() / len(numeric_df)
        )
        
        stats[numeric_columns] = {'mean':mean,
                                  'median':median,
                                  'std':std,
                                  'q25':quantile25,
                                  'q75':quantile75,
                                  'nulls': null_count}
    
    return stats


def cat_proportion(df,cat_df,target_v):
    categorical_temp = df[list(cat_df.columns) + [target_v]]
    a = pd.DataFrame()
    for predictor in cat_df.columns:
        b = categorical_temp.groupby(predictor).mean(target_v)
        a = pd.concat([a,b])
        display(b)
        print('\n')
    a['diff'] = np.abs(a-df['client_stayed'].mean())
    return a.sort_values('diff',ascending=False).iloc[0:10,:]


def magic(df,categoric,numeric,metric=None,show_conditions:bool=False):
    conditions = df.groupby(categoric).mean()[[numeric]].to_dict()[numeric]
    if metric=='median':
        conditions = df.groupby(categoric).median()[[numeric]].to_dict()[numeric]
    if show_conditions:
        print(conditions)
    new_variable = []
    for index in df.index:
        key = df.loc[index,categoric]
        value = df.loc[index,numeric]
        if  value < conditions[key]:
            new_variable.append(0)
        else:
            new_variable.append(1)
    return new_variable



class V_selection:

    def __init__(self,train,test,values,y):
        self.train = train
        self.test = test
        self.values = values
        self.y = y


    def _get_steps(self):
        min_v, max_v = self.values.min(), self.values.max()
        step = (max_v - min_v) / len(self.values)

        return np.arange(min_v, max_v + step, step)


    # Seleccion de variables
    def seleccionados(self):
        index_vistos = []
        for index, threshold in enumerate(self._get_steps()):
            seleccionados = self.values.loc[self.values > threshold].index.tolist()
            
            if seleccionados in index_vistos:
                continue

            index_vistos.append(seleccionados)

        return index_vistos

    # Generación de modelos
    def generate_models(self):
        index_vistos = []
        evaluaciones = {}
        index = 0

        for threshold in self._get_steps():
            seleccionados = self.values.loc[self.values > threshold].index.tolist()

            if len(seleccionados) == 0:
                continue

            if seleccionados in index_vistos:
                continue

            index_vistos.append(seleccionados)

            # modelo

            modelo = LogisticRegression()
            modelo.fit(self.train[seleccionados], self.train[self.y])
            scores = modelo.predict_proba(self.test[seleccionados])[:,1]

            performance = roc_auc_score(y_score=scores, y_true=self.test[self.y])

            #guardar
            evaluaciones[index] = {'threshold': threshold, 'variables': seleccionados, 'performance': performance}

            index += 1
        return evaluaciones

    def visualize_selections(self,method):
        
        fig, (izq, der) = plt.subplots(nrows=1, ncols=2,figsize=(10,7))
        etiquetas = []

        for index, informacion in self.generate_models().items():
            # sacar la info
            seleccionados = informacion['variables']
            threshold = informacion['threshold']
            performance = informacion['performance']

            variables= str(len(seleccionados))
            etiquetas.append(variables)

            # Graficar perfonance
            label = f'{np.round(performance, 2)}'
            izq.scatter(index, performance, label=label)

            # Graficar threshold
            label = f'{np.round(threshold, 2)}'
            der.scatter(index, threshold, label=label)

            #etiquetas de eje
        izq.set_xticks(range(len(etiquetas)))
        izq.set_xticklabels(etiquetas)
        der.set_xticks(range(len(etiquetas)))
        der.set_xticklabels(etiquetas)

        #Nombres de ejes
        izq.set_xlabel('vars elegidas')
        izq.set_ylabel('AUC')
        der.set_xlabel('vas elegidas')
        der.set_ylabel('Valor threshold')
        izq.legend()
        der.legend()

        # Titulos
        fig.suptitle(f'Cortes con {method}')
        izq.set_title('Performance')
        der.set_title('Thresholds')

        fig.tight_layout()