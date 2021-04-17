# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 19:55:04 2021

@author: Gabriel A. Morales Ruiz
"""
import pandas as pd
import numpy as np

# Realizar código en Python que reciba una tabla de datos numéricos organizados
# en 2 columnas (x,y) y devuelva los centroides de k>=2 clases usando una red
# de Kohonen.
# Inicialice los centroides con valores aleatorios (de forma uniforme en el
# rango de valores de cada columna), y detenga el proceso cuando no haya
# centroide que presente un cambio significativo (a su nivel de precisión).
# Luego, realice código que reciba una pareja de valores (x,y) y devuelva a
# qué clase pertenece de acuerdo a la clasificación obtenida anteriormente.

def kohonen(data, k = 2, centroids = None, max_iters = 100, E = 1e-1) :
    """
    K-means algorithm for n dimensions and k clusters/centroids.
    Parameters
    ----------
    data : np.matrix
        Data to classify.
    k : int, optional
        Amount of clusters/centroids. The default is 2.
    centroids : list of lists / np.matrix, optional
        DESCRIPTION. Starting centroids.
    max_iters : int, optional
        Maximum iterations. The default is 100.

    Returns
    -------
    centroids : np.matrix
        Centroids after the kohonen algorithm finishes.

    """
    m = np.size(data, axis=0)
    if centroids is None :
        centroids = get_random_centroids(data, k)
        
    elif len(centroids) != k :
        AssertionError("Número de centroides no equivale a k")
        
    for i in range(max_iters) :
        old_centroids = centroids.copy()
        rate = 1/(i+2)
        for j in range(m) : # for each sample in the data variable
            assigned_centroid = assign_centroid(data[j,:], centroids)
            centroids[assigned_centroid] = centroids[assigned_centroid]*rate +\
                                           (1-rate)*data[j,:] 
        error = sum([np.linalg.norm(centroids[d] - old_centroids[d]) for d in range(k)])
        print("Generation " + str(i+1) + ". Error: " + str(error))
        if(error < E) : break
    
    return centroids

def get_random_centroids(data, k) :
    """
    Function generates random starting centroids according to the input data's 
    factors ranges.
    Parameters
    ----------
    data : numpy.matrix
        Input data
    k : int
        Amount of clusters/centroids.

    Returns
    -------
    np.matrix
        Randomly generated centroids.

    """
    centroids = []
    columns = np.size(data, axis=1)
    ranges = []
    for i in range(columns) :
        ranges.append([np.min(data[:,i]), np.max(data[:,i])])
    
    for i in range(k) :
        centroid = []
        for span in ranges :
            centroid.append(np.random.uniform(span[0], span[1]))
        centroids.append(centroid)
        
    return np.matrix(centroids)
        
def assign_centroid(data, centroids) :
    """
    Function will calculate the data's distance to the centroids and return
    the index of the pertinent centroid.
    Parameters
    ----------
    data : np.matrix
        Data to classify.
    centroids : np.matrix
        Centroids to use in calculations.

    Returns
    -------
    assigned_centroids : int
        Index of pertinent centroid.

    """    
    dist = []
    for centroid in centroids :
        dist.append((np.linalg.norm(data - centroid)))
    return np.argmin(dist)

# Separé la función en singular y plural. Singular para ahorra procesamiento
# en Kohonen, y plural por si quieres meter un vector ya con los clusters
# formados para predecir su lugar.
def assign_centroids(data, centroids) :
    m = np.size(data, axis=0)
    assigned_centroids = []
    for i in range(m) :
        assigned_centroids.append(assign_centroid(data[i,:], centroids))
    return assigned_centroids
    

#%%
data = pd.read_excel("kohonen_test.xlsx")
# El archivo tiene columnnas con nombres en la primera fila (1)
# e.g.
#        A     B     C     D
# 1    x_1   x_2   x_3   x_4
# 2      0    15    21    54
# 3     90    18    74    23
# ...
data = np.matrix(data.iloc[:,:])

centroids = np.matrix([[10, 35], [90, 65]], dtype = float)
centroids = kohonen(data, centroids = centroids)
# centroids = kohonen(data, k = 2)
#%%
data_to_classify = np.matrix([[1, 2], [90, 90]])
print(assign_centroids(data_to_classify, centroids))
