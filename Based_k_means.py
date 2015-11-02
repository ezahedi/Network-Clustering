# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 22:29:06 2015

@author: Emad Zahedi

Input : A connected graph G
Output : n clusters 
"""
import math
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.io
import sys
from networkx.utils import dict_to_numpy_array
from collections import defaultdict
import time
import operator
import matplotlib.pyplot as pyplot
import copy
import pandas
random.seed(0)

G = nx.fast_gnp_random_graph(8,.4)
n_clusters = 3


def _kmeans_init(G, n_clusters, method='balanced'):
    """ Initialize k=n_clusters centroids randomly
    """
    Centroids = []
    H = G.copy()
    for pop in range(n_clusters):           
              i = random.choice(H.nodes())
              H.remove_node(i)
              Centroids.append(i)
    return (Centroids)
D_Matrix = nx.floyd_warshall_numpy(G)    



centeroids = _kmeans_init(G, n_clusters)
def _cal_dist2center(G, Centeroids):
    """ Calculate the distances to cluster centers
    """   
    Dict = {}
    for i in Centeroids:
        Dict[i] = []
        for j in range(len(G.nodes())):
            Dict[i].append(D_Matrix[i,j])
    return(Dict) 



Dict = _cal_dist2center(G, centeroids)
def Dict2Matr(Dict):
    """ Change the dictionary to a matrix 
    """
    df    = pandas.DataFrame(Dict)
    Matr  = df.values
    Centr = df.columns
    Matr = np.vstack([Matr, Centr])
    return(Matr)



A = Dict2Matr(Dict)

def _assign_clusters(G,A):
    """ Assign each point to the closet cluster center    
    """
    Dict = {}
    D_mat = A[:-1,:]  #distance matrix
    C_list = A[-1,:]  #centroid list
    for i in C_list:
        Dict[i] = []
    for j in range(len(G.nodes())):
        Dict[C_list [D_mat.argmin(1)[j]]].append(j)
    Clusters = []
    for i in C_list:
            Clusters.append(Dict[i])
    return(Clusters)
    


Clusters = _assign_clusters(G,A)








def _kmeans_run(G, n_clusters, centeroids):
    """ Run a single trial of k-means clustering
        on dataset X, and given number of clusters
    """
    
    Dict = _cal_dist2center(G, centeroids)
    A = Dict2Matr(Dict)
    Clusters = _assign_clusters(G,A)
    return(Clusters)


def _update_centers(D_Matrix, Clusters, n_clusters):
    """ Update Cluster Centers:
           assign the centroid with min SSE for each cluster
    """
    New_Centers = []
    for clust in Clusters:
        X =[]        
        for i in clust:
            Sum = 0
            for j in clust:
                Sum += D_Matrix[i,j]            
            X.append(Sum)
        a = X.index(min(X))
        New_Centers.append(clust[a])
        
    return(New_Centers)


New_Centers = _update_centers(D_Matrix, Clusters, n_clusters)




def _kmeans(G, n_clusters):
    """ Run multiple trials of k-means clustering,
        and outputt is the best centers, and cluster labels
    """
    Old_Centroids  = set(_kmeans_init(G, n_clusters, method='balanced'))
    Clusters = _kmeans_run(G, n_clusters, _kmeans_init(G, n_clusters))
    New_Centroids = set(_update_centers(D_Matrix, Clusters, n_clusters))
    while True :
        if Old_Centroids == New_Centroids:
            return(New_Centroids) #,_kmeans_run(G, n_clusters, New_Centroids))
            break
        else:
            #print('OOOO',Old_Centroids , New_Centroids )
            Old_Centroids = New_Centroids
            New_Centroids = list(New_Centroids)
            Clusters = _kmeans_run(G, n_clusters, New_Centroids)
            New_Centroids = set(_update_centers(D_Matrix, Clusters, n_clusters))

            
         




def Best_initial(G, n_clusters, iteration = 10):
    iteration = 10
    Candidates = []
    for i in range(iteration):
        local_centroids = _kmeans(G, n_clusters) 
        for k in local_centroids:
            Candidates.append(k)
    Help_Dict = {}
    for i in Candidates:
        Help_Dict[i] = Candidates.count(i)
        
           
    Best_Centroids = []
    for j in range(n_clusters):
        Max = max(Help_Dict.iteritems(), key=operator.itemgetter(1))[0]
        Best_Centroids.append(Max)
        del Help_Dict[Max]
    return(Best_Centroids)



print('Centroids :', Best_initial(G, n_clusters, iteration = 10))
A = Best_initial(G, n_clusters, iteration = 10)
print('Clusters :', _kmeans_run(G, n_clusters, A))












class KMeans(object):
    """
        KMeans Clustering
        Parameters
        -------
           n_clusters: number of clusters (default = 2)
           n_trials: number of trial random centroid initialization (default = 10)
           max_iter: maximum number of iterations (default = 100)
           tol: tolerance (default = 0.0001)
        Attibutes
        -------
           labels_   :  cluster labels for each data item
           centers_  :  cluster centers
           sse_arr_  :  array of SSE values for each cluster
           n_iter_   :  number of iterations for the best trial
           
        Methods
        ------- 
           fit(X): fit the model
           fit_predict(X): fit the model and return the cluster labels
    """

    def __init__(self, n_clusters=2, n_trials=10, max_iter=100, tol=0.001):
        
        self.n_clusters = n_clusters
        self.n_trials = n_trials
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        """ Apply KMeans Clustering
              X: dataset with feature vectors
        """
        self.centers_, self.labels_, self.sse_arr_, self.n_iter_ = \
              _kmeans(X, self.n_clusters, self.max_iter, self.n_trials, self.tol)


    def fit_predict(self, X):
        """ Apply KMeans Clustering, 
            and return cluster labels
        """
        self.fit(X)
        return(self.labels_)

     
        
        
        
        
        

pos=nx.spring_layout(G) # positions for all nodes
node_colors = ['b','g','r','y','c','k','m'] 
# nodes
Clust = _kmeans_run(G, n_clusters, A)


for item in range(n_clusters):
    for group in Clust[item]:

        nx.draw_networkx_nodes(G,pos,
                               nodelist = Clust[item],
                               node_color=node_colors[item],
                               node_size=80,
                           alpha=0.8)


# edges
nx.draw_networkx_edges(G,pos,width=1.0,alpha=0.5)
nx.draw_networkx_edges(G,pos,
                       edgelist=G.edges(),
                       width=.1,alpha=0.5,edge_color='k')



# some math labels
labels={}
for i in range(len(G.nodes())):
    labels[i]= str(i)

nx.draw_networkx_labels(G,pos,labels,font_size=16)

plt.axis('off')
plt.savefig("labels_and_colors.png") # save as png
plt.show() # display
        

  
nx.write_gml(G,"test.gml")
nx.write_edgelist(G, "test.edgelist")
fh=open("test.edgelist", 'w')

#nx.write_edgelist(G,fh)
#nx.write_edgelist(G,"test.edgelist.gz")


  
        
