# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 22:28:20 2015

@author: andreagoodluck
"""


# -*- coding: utf-8 -*-

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
import pylab
random.seed(0)





'''
G=nx.Graph()

G.add_edge('a','b',weight=0.6)
G.add_edge('a','c',weight=0.2)
G.add_edge('c','d',weight=0.1)
G.add_edge('c','e',weight=0.7)
G.add_edge('c','f',weight=0.9)
G.add_edge('a','d',weight=0.3)
'''



G=nx.Graph()

G.add_edge('1','2',weight=0.24)
G.add_edge('1','3',weight=0.22)
G.add_edge('1','4',weight=0.37)
G.add_edge('1','5',weight=0.34)
G.add_edge('1','6',weight=0.23)

G.add_edge('2','3',weight=0.15)
G.add_edge('2','4',weight=0.20)
G.add_edge('2','5',weight=0.14)
G.add_edge('2','6',weight=0.25)

G.add_edge('3','4',weight=0.15)
G.add_edge('3','5',weight=0.28)
G.add_edge('3','6',weight=0.11)

G.add_edge('4','5',weight=0.29)
G.add_edge('4','6',weight=0.22)

G.add_edge('5','6',weight=0.39)





val_map = {'A': 1.0,
                   'D': 0.5714285714285714,
                              'H': 0.0}
values = [val_map.get(node, 0.45) for node in G.nodes()]
edge_colors = 'k'

edge_labels=dict([((u,v,),d['weight'])
             for u,v,d in G.edges(data=True)])
'''
pos=nx.spring_layout(G) # positions for all nodes                
nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
nx.draw(G,pos, node_color = values, node_size=15,edge_color=edge_colors,edge_cmap=plt.cm.Reds)
pylab.show()

'''











Iterations = 5


D_Matrix = nx.floyd_warshall_numpy(G) 
nodes_label = []
for i in range(len(G.nodes())):
     nodes_label.append(set(G.nodes()[i]))
A = np.vstack([D_Matrix, nodes_label])









for iter in range(Iterations):
    i,j = np.unravel_index(A[:-1,:].argmax(), A[:-1,:].shape)
    for k in range(A.shape[1]):
        A[i,k] = max(A[i,k] , A[j, k])
        A[k,i] = max(A[k,i] , A[k, j])
        A[i,i] = 0
        A[-1,:][0,i] = (A[-1,:][0,i]) | (A[-1,:][0,j]) 
    A = np.delete(A, j, 0)
    A = np.delete(A, j, 1)

    
print(A)








'''
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

     
        
      
        













'''

















     
        

pos=nx.spring_layout(G) # positions for all nodes
node_colors = ['b','g','r','y','c','k','m'] 
# nodes


C_list = A[-1,:] 
for Clust in range(C_list.shape[1]):
        nx.draw_networkx_nodes(G,pos,
                               nodelist = list(C_list[0,Clust]),
                               node_color=node_colors[Clust],
                               node_size=80,
                           alpha=0.8)


# edges
nx.draw_networkx_edges(G,pos,width=1.0,alpha=0.5)
nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)




# some math labels
labels={}
for i in range(len(G.nodes())):
    labels[i]= G.nodes()[i]

nx.draw_networkx_labels(G,pos,labels,font_size=16)

plt.axis('off')
plt.savefig("labels_and_colors.png") # save as png
plt.show() # display
        

  
nx.write_gml(G,"test.gml")
nx.write_edgelist(G, "test.edgelist")
fh=open("test.edgelist", 'w')
nx.write_edgelist(G,fh)
nx.write_edgelist(G,"test.edgelist.gz")


        
