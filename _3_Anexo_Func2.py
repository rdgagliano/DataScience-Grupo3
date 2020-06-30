#FUNCIONES AUXILIARES

#CONTIENE FUNCIONES PARA ENTRENAR MODELOS KNN, REG LOGSTICA Y NAIVE BAYES SIN PIPELINES

#TAMBIEN CONTIENE FUNCIONES PARA EVALUAR RESULTADOS, COMO POR EJ. PARA GRAFICAR METRICES DE CONFUSION


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.simplefilter("ignore")


# In[2]:


from sys import maxsize #para imprimir arrays completos
import numpy as np
import pandas as pd

from sklearn import preprocessing #para normalizar datos
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate, StratifiedKFold
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import roc_curve, auc, precision_score, recall_score

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


#Método para imprimir información básica de las columnas del dataframe
def getInfoByColumn(df):
    
    for column in df:
        
        InfoBasica = df[column].describe()
        
        uniqueValuesCount = len(df[column].unique())
        
        #si la columna tiene menos de 10 valores, los imprimimos sin problemas
        #si tiene más truncamos el texto para simplificar la lectura
        if (uniqueValuesCount < 10):
            
            ShowUnique = 'Show Unique  ' + str(df[column].unique()).strip('[]')
        else:
            ShowUnique = 'Show Unique  ' + str(df[column].unique()[0:30]).strip('[]') + ',etc...'
        
        print('Información columna: {} \n''---------------\n{}'.format(column, InfoBasica))
        print('{}''\n'.format(ShowUnique))


# In[4]:


#Método para graficar un histograma por cada columna del dataset
def getHistogramByColumn(df):
    for column in df:

        #Gráfica Histograma:
        Histograma = df[column].hist(grid=False, color='indigo', bins=10, xlabelsize=10, xrot=45)
        
        #Título y nombre de ejes: 
        plt.xlabel(column, fontsize= 13, color='green')
        plt.ylabel('Freq.',fontsize= 13, color='green')
        plt.title('Columna: ' + column, fontsize= 20, color='mediumslateblue')
        
        plt.legend(labels=df[column],  loc='upper right', fontsize='small',bbox_to_anchor=(1.3, 1))
        plt.show()
        print (Histograma)


# In[5]:


#Método para obtener datos estadísticos de cada columna del dataframe
def getStatisticForEachColumn(df):
    
    for column in df:
        
        STD = df[column].std()
        
        MEAN = df[column].mean()
        
        VAR =  df[column].var()
        
        print('Statistics mesures from:{}\n-----------------------------\nSTD:{}\nVAR: {}\nMean: {}\n'.format(column, STD, VAR, MEAN))


# In[6]:


#Método para generar un gráfico de correlación de las variables del data frame
def plot_corr(df,size=10):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    corr = df
    fig, ax = plt.subplots(figsize=(size, size),)
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);


# # Métodos relacionados al modelo KNN

# In[7]:


#Método para obtener el hiperparámetro K más óptimo
#useStandarization determina si se aplica o no la estandarización a los valores de los datos 
#retorna un dataframe con el score para todos los K's, cuya cantidad es determinada por quantityK
def getScoresForHyperparameterK(quantityK, stepK, model_X_train, model_y_train, kFold_N_Splits=5, KFold_shuffle=True, useStandarization=False):
    kf = StratifiedKFold(n_splits=kFold_N_Splits, shuffle=KFold_shuffle, random_state=12)

    scores_para_df = []
    
    #si viene configurado, se realiza la estandarización de los valores
    if(useStandarization):
        scaler = StandardScaler()
        model_X_train = scaler.fit_transform(model_X_train)

    for i in range(1, quantityK+1, stepK):

        # En cada iteración instanciamos el modelo con un hiperparámetro distinto
        model = KNeighborsClassifier(n_neighbors=i)

        # cross_val_scores nos devuelve un array de 5 resultados,
        # uno por cada partición que hizo automáticamente CV
        cv_scores = cross_val_score(model, model_X_train, model_y_train, cv=kf)

        # Para cada valor de n_neighbours, creo un diccionario con el valor
        # de n_neighbours y la media y el desvío de los scores.
        dict_row_score = {'score_medio':np.mean(cv_scores),                          'score_std':np.std(cv_scores), 'n_neighbours':i}

        # Guardo cada uno en la lista de diccionarios
        scores_para_df.append(dict_row_score)
    
    dfResult = pd.DataFrame(scores_para_df)
    dfResult['limite_inferior'] = dfResult['score_medio'] - dfResult['score_std']
    dfResult['limite_superior'] = dfResult['score_medio'] + dfResult['score_std']
    
    return dfResult
    


# In[8]:


def getKNNPredictions(X_train, y_train, X_test, y_test, K, useStandarization = False):
    
    if(useStandarization):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train) 
        X_test = scaler.transform(X_test) 
    
    model = KNeighborsClassifier(n_neighbors=K)   
    
    print(model)
    
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    return y_pred, model.score(X_test, y_test)


# # Metodos relacionados al modelo Regresion Logistica

# In[9]:


#Método para obtener el hiperparámetro C más óptimo
#useStandarization determina si se aplica o no la estandarización a los valores de los datos
#retorna un dataframe con el score para todos los C's del array valoresPosiblesC
def getScoresForHyperparameterC(valoresPosiblesC, model_X_train, model_y_train, kFold_N_Splits=5, KFold_shuffle=True, useStandarization=False):
    kf = KFold(n_splits=kFold_N_Splits, shuffle=KFold_shuffle, random_state=12)

    scores_para_df = []
    
    #si viene configurado, se realiza la estandarización de los valores
    if(useStandarization):
        scaler = StandardScaler()
        model_X_train = scaler.fit_transform(model_X_train)

    for i in valoresPosiblesC:
        
        #para evitar el warning
        #C:\Users\User\Anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
        #se usa el parametro solver='lbfgs'
        model = linear_model.LogisticRegression(C=i, solver='lbfgs', class_weight='balanced')        
        cv_scores = cross_val_score(model, model_X_train, model_y_train, cv=kf)
        
        dict_row_score = {'score_medio':np.mean(cv_scores), 'score_std':np.std(cv_scores), 'C':i}        
        scores_para_df.append(dict_row_score)

    dfResult = pd.DataFrame(scores_para_df)
    dfResult['limite_inferior'] = dfResult['score_medio'] - dfResult['score_std']
    dfResult['limite_superior'] = dfResult['score_medio'] + dfResult['score_std']
    
    return dfResult


# In[10]:


def getLogisticRegressionPredictions(X_train, y_train, X_test, y_test, C, useStandarization = False):
    
    if(useStandarization):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train) 
        X_test = scaler.transform(X_test) 
    
    #para evitar el warning
    #C:\Users\User\Anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
    #se usa el parametro solver='lbfgs'
    model = linear_model.LogisticRegression(C=C, solver='lbfgs', class_weight='balanced')   
    
    print(model)
    
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    
    return y_pred, model.score(X_test, y_test), model.coef_


# # Metodos para Barnoulli Naive Bayes

# In[11]:


#metodo para obtener el hiperparamatro alfa con el que mejor resulta el modelo Bernouilli Naive Bayes
#useStandarization determina si se aplica o no la estandarización a los valores de los datos
def getScoresForHypermarameterAlphaNB(valoresPosiblesAlpha, model_X_train, model_y_train, kFold_N_Splits=5, KFold_shuffle=True, useStandarization=False):
    kf = KFold(n_splits=kFold_N_Splits, shuffle=KFold_shuffle, random_state=12)

    scores_para_df = []
    
    #si viene configurado, se realiza la estandarización de los valores
    if(useStandarization):
        scaler = StandardScaler()
        model_X_train = scaler.fit_transform(model_X_train)

    for alpha in valoresPosiblesAlpha:
        
        
        model = BernoulliNB(alpha=alpha)
        cv_scores = cross_val_score(model, model_X_train, model_y_train, cv=kf)
        
        dict_row_score = {'score_medio':np.mean(cv_scores), 'score_std':np.std(cv_scores), 'Alpha':alpha}        
        scores_para_df.append(dict_row_score)

    dfResult = pd.DataFrame(scores_para_df)
    dfResult['limite_inferior'] = dfResult['score_medio'] - dfResult['score_std']
    dfResult['limite_superior'] = dfResult['score_medio'] + dfResult['score_std']
    
    return dfResult


# In[12]:


def getBernoulliNaiveBayesPredictions(bestAlpha, X_train, y_train, X_test, y_test):
    model = BernoulliNB(alpha = bestAlpha)
    model.fit(X_train, y_train)
    
    print(model)
    
    y_pred = model.predict(X_test)
    
    return y_pred, model.score(X_test, y_test)


# # Metodos para Multinomial Naive Bayes

# In[13]:


#metodo para obtener el hiperparamatro alfa con el que mejor resulta el modelo Bernouilli Naive Bayes
#useStandarization determina si se aplica o no la estandarización a los valores de los datos
def getScoresForHypermarameterAlphaMNB(valoresPosiblesAlpha, model_X_train, model_y_train, kFold_N_Splits=5, KFold_shuffle=True, useStandarization=False):
    kf = KFold(n_splits=kFold_N_Splits, shuffle=KFold_shuffle, random_state=12)

    scores_para_df = []
    
    #si viene configurado, se realiza la estandarización de los valores
    if(useStandarization):
        scaler = StandardScaler()
        model_X_train = scaler.fit_transform(model_X_train)

    for alpha in valoresPosiblesAlpha:
        
        
        model = MultinomialNB(alpha=alpha, fit_prior=True, class_prior=None)
        cv_scores = cross_val_score(model, model_X_train, model_y_train, cv=kf)
        
        dict_row_score = {'score_medio':np.mean(cv_scores), 'score_std':np.std(cv_scores), 'Alpha':alpha}        
        scores_para_df.append(dict_row_score)

    dfResult = pd.DataFrame(scores_para_df)
    dfResult['limite_inferior'] = dfResult['score_medio'] - dfResult['score_std']
    dfResult['limite_superior'] = dfResult['score_medio'] + dfResult['score_std']
    
    return dfResult


# In[14]:


def getMultinomialNaiveBayesPredictions(bestAlpha, X_train, y_train, X_test, y_test):
    model = MultinomialNB(alpha = bestAlpha, fit_prior=True, class_prior=None)
    model.fit(X_train, y_train)
    
    print(model)
    
    y_pred = model.predict(X_test)
    
    return y_pred, model.score(X_test, y_test)


# # Matriz de confusion

# <table>
#   <thead>
#     <tr>
#       <th>PREDICHOS</th>
#       <th>0 (F)</th>
#       <th>1 (V)</th>
#     </tr>
#       <tr>
#       <th>REALES</th>
#       <th></th>
#       <th><th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <td>0 (F)</td>
#       <td>TN</td>
#       <td>FN</td>
#     </tr>
#     <tr>
#       <td>1 (V)</td>
#       <td>FP</td>
#       <td>TN</td>
#     </tr>
#   </tbody>
# </table>
# 
# TN: True Negative
# FN: False Negative
# FP: False Positive
# TN: True Positive

# In[15]:


#grafica la matriz de confusion y retorna los valores tn, fp, fn, tp de la misma
def getConfusionMatrix(y_test, y_pred, size=5, labels = []):
    confusionMatrix = {}
    
    if(labels == []):
        confusionMatrix = confusion_matrix(y_test, y_pred)
    else:
        confusionMatrix = confusion_matrix(y_test, y_pred, labels)
    
    fig, ax = plt.subplots(figsize=(size,size))   
    sns.heatmap(confusionMatrix, annot=True, fmt='d',linewidths=.5,cmap="Blues")
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)   
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.ylabel('Valores verdaderos')
    plt.xlabel('Valores predichos');
    
    return confusionMatrix.ravel()


# # Métodos relacionados a métricas

# ##### Accuracy, para problemas que estén equiibrados y no sesgados

# In[16]:


def getModelAccuracy(tn, fp, fn, tp):
    return (tp + tn) / (tn + fp + fn + tp)


# ##### Para tratar de capturar la mayor cantidad de positivos posibles (cuando para el modelo es necesario capturar también los falsos positivos)

# In[17]:


def getModelRecall(tn, fp, fn, tp):
    return tp / (tp + fn)


# In[18]:


def getModelRecall(y_test, y_pred):
    return recall_score(y_test, y_pred)


# ##### Para cuando se quiere estar muy seguro de una predicción positiva. Mide la capacidad del clasificador de no etiquetar como positiva una muestra que es negativa (1 es el mejor valor)

# In[19]:


def getModelPrecision(tn, fp, fn, tp):
    return tp / (tp + fp)


# In[20]:


def getModelPrecision(y_test, y_pred):
    return precision_score(y_test, y_pred)


# ##### Metrica F1, cuanto mayor es su valor, mejor es el modelo

# In[21]:


def getModelMetricF1(tn, fp, fn, tp):
    precision = getModelPrecision(tn, fp, fn, tp)
    recall = getModelRecall(tn, fp, fn, tp)
    
    return (2 * precision * recall) / (precision + recall)


# ##### Metrica F beta, similar a F1, pero puede regularse la importancia de cada termino mediante el valor de beta. SI beta > 1 => favorece al recall. Si beta < 1 => favorece a la precision

# In[22]:


def getModelMetricFBeta(betaCoeficient, tn, fp, fn, tp):
   
    precision = getModelPrecision(tn, fp, fn, tp)
    recall = getModelRecall(tn, fp, fn, tp)
    
    return (1 + betaCoeficient * betaCoeficient) * (precision * recall) / (betaCoeficient * betaCoeficient * precision + recall)


# ### Curva ROC

# ##### Sensitivity (True Positive Rate)

# In[23]:


def getTruePositiveRatio(tn, fp, fn, tp):
    return tp / (tp + fn)


# ##### Specificity (False Positive Rate)

# In[24]:


def getFalsePositiveRatio(tn, fp, fn, tp):
    return fp / (fp + tn)


# ##### Ploteo de curvas ROC

# In[25]:


#imprime la curva ROC para un modelo en particular
def plotSingleROC_Curve(modelName, y_test, y_prob):
    
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    
    plt.figure(figsize=(5,5))
    plt.title(modelName + 'Receiver Operating Characteristic Curve')
    plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],linestyle='--')
    plt.axis('tight')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')


# In[26]:


#imprime las curvas ROC para varios modelos
#modelsDataDictionaryArray es un array de diccionarios con la siguiente forma
#{"y_prob":<datos predichos por el modelo>, "modelLabel":<etiqueta para identificar el modelo en el gráfico>}
def plotMultipleROC_Curve(y_test, modelsDataDictionaryArray):
    
    plt.figure(figsize=(10,10))
    plt.title('Receiver Operating Characteristic Curve')
   
           
    for dictionary in modelsDataDictionaryArray:
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, dictionary["y_prob"])
        roc_auc = auc(false_positive_rate, true_positive_rate)
        label = dictionary["modelLabel"] + ' - AUC = %0.2f '
        plt.plot(false_positive_rate, true_positive_rate, label = label % roc_auc)    

    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],linestyle='--')
    plt.axis('tight')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

