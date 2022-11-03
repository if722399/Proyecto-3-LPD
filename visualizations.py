"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Proyecto de aplicación III: Preparación y feature engineering de variables numéricas       -- #
# -- script: functions.py : python script with general visualizations                                    -- #
# -- author: @if722399                                                                                   -- #
# -- repository: https://github.com/if722399/Proyecto-3-LPD.git                                          -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')


def nulls_substitution(df,variable,metric):
    if metric=='mode':
        metric = df[variable].value_counts().index[0]
    else:
        metric = metric = df[variable].mean()
    plt.figure(figsize=(10,7))
    sns.histplot(df[variable], color='blue')
    sns.histplot(df[variable].fillna(metric), color='red', alpha=.4)
    plt.title('Comparing distributions');





def _get_colors_to_use(variables):
    """ Función para asignarle colores crecientes a una lista de elements
    
    Parámetros
    ----------
    variables: Lista de elementos a los cuales les queremos asignar color


    Regresa
    -------
    Dictionario de la forma: {element: color}
    """
    colors = plt.cm.jet(np.linspace(0, 1, len(variables)))
    return dict(zip(variables, colors))

def autolabel(rects, ax):
    """
    Método auxiliar para agregarle el númerito correspondiente a su valor 
    a la barra en una gráfica de barras.
    
    Esta función no la hice yo (aunque sí la modifiqué). La origi está en:
    https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html
    
    rects: La figura de la gráfica guardada en una variable
    ax: El eje donde se está graficando.
    """
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.,
                1.05*height,
                '%d'%int(height),
                ha='center', va='bottom')


def plot_numeric(df, numeric_stats):
    corr = df.select_dtypes(exclude=['object']).corr()
    corr.style.background_gradient(cmap='coolwarm')

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.matshow(corr, cmap='Blues')

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)

    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)

    ax.grid(False)
    
    metrics = ['mean', 'median', 'std', 'q25', 'q75', 'nulls']
    colors = _get_colors_to_use(metrics)

    for index, variable in enumerate(sorted(numeric_stats.keys())):

        # Plotting basic metrics
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))

        bar_position = -1
        for metric, value in numeric_stats[variable].items():
            bar_position += 1

            if value is None or np.isnan(value):
                value = -1

            # Plotting bars
            bar_plot = ax[0].bar(bar_position, value,
                                 label=metric, color=colors[metric])
            autolabel(bar_plot, ax[0])

            # Plotting histogram
            df[variable].plot(kind='hist', color='blue',
                              alpha=0.4, ax=ax[1])

            # Plotting boxplot
            df.boxplot(ax=ax[2], column=variable)

            ax[0].set_xticks(range(len(metrics)))
            ax[0].set_xticklabels(metrics, rotation=90)
            ax[2].set_xticklabels('', rotation=90)

            ax[0].set_title('\n Basic metrics \n', fontsize=10)
            ax[1].set_title('\n Data histogram \n', fontsize=10)
            ax[2].set_title('\n Data boxplot \n', fontsize=10)
            fig.suptitle(f'Variable: {variable} \n\n\n', fontsize=15)

            fig.tight_layout()
    return


def prop_by_quint(df:pd.DataFrame,
                  predictor:pd.Series,
                  target:pd.Series,
                  n:int):
    
    df[predictor+'_quintile'] = pd.qcut(df[predictor],n,duplicates='drop')
    temp_df = df.groupby(predictor+'_quintile').mean()[target]
    n_proportions=df[predictor+'_quintile'].nunique()

    plt.figure(figsize=(9,6.5))
    temp_df.plot(
        title=f'Slicing {predictor} into {n_proportions} equal proportions',
    )
    plt.xlabel('')
    plt.ylabel('Proportion of clients who stayed')
    plt.xticks(rotation=45, ha='right');
    if n_proportions!=n:
        print(f'Se ajustó el número de intervalos en la variable {predictor} debido a que existieron duplicados')