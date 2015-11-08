# -*- coding: utf-8 -*-
"""
@authors: Emad Zahedi & Vahid Mirjalili
Input: A graph G.
Output: Clusters obtained by Average linkage method pluse drawn colored graphs 
Note: If you are looking for specific level, like 'k', of clusters remove the for loop and initiate Iteration = 'k'
"""
import numpy as np
import networkx as nx
import copy
from copy import deepcopy 


def _Index(Matrix, a):
    """Return the index of an specific element, 'a', in the matrix"""
    for i in range(Matrix.shape[0]):
        for j in range(Matrix.shape[1]):
            if Matrix[i,j] == a:
                return(i,j)
   
    
def Dis_Clus(G):
    """adding one row to distance matrix 
    The new row shows nodes clusters
    each node is a cluster at initial"""   
    D_Matrix = nx.floyd_warshall_numpy(G) 
    nodes_label = []
    for i in range(len(G.nodes())):
         nodes_label.append(set(G.nodes()[i]))
         
    A = np.vstack([D_Matrix, nodes_label])  
    return(A)
    
 
def Average_Linkage(G, Iterations):
    """Give the clusters based on Average linkage method"""       
    A = Dis_Clus(G)
    D_matrix = A[:-1,:]
    for iter in range(Iterations):     
        minval = np.min(D_matrix[np.nonzero(D_matrix)])
        i,j    = _Index(D_matrix, minval)
        for k in range(D_matrix.shape[1]):
            A[i,k] = A[i,k] + A[j, k]
            A[k,i] = A[k,i] + A[k, j]
            A[i,i] = 0.0
          
        A[-1,:][0,i] = (A[-1,:][0,i]) | (A[-1,:][0,j]) 
        B = copy.deepcopy(A)
        for k in range(D_matrix.shape[1]):
            for s in range(D_matrix.shape[1]):
                if s > k :
                    B [s,k] = float(B[s,k])/ (len(B[-1,:][0,s])*len(B[-1,:][0,k]))
                    B [k,s] = B [s,k]   
        B = np.delete(B, j, 0)
        B = np.delete(B, j, 1)
        D_matrix    = B[:-1,:] 
        A = np.delete(A, j, 0)
        A = np.delete(A, j, 1)
    Clusters =  A[-1,:]
    return(Clusters)






##
## Class KMeanserror
##
class Average_linkage_Error( ValueError ):
    
    pass

##
## Class KMeans
##
class Average_linkage(object):
    """
        Average_linkage Clustering
        Parameters
        -------
           G         : A connected graph 
           Iterations: number of level clusters (default = 2)          
           
        Attibutes
        -------
           clusters   :  nodes clusters
           
        Methods
        ------- 
           fit(X): fit the model
           fit_predict(X): fit the model and return the centroids and clusters
    """

    def __init__(self, G, Iterations=2):
        
        self.G = G        
        self.Iterations = Iterations


    def fit(self, G):
        """ Apply Average-Linkage Clustering
              G: A weighted graph
        """
        self.clusters = Average_Linkage(self.G, self.Iterations)


    def fit_predict(self, G):
        """ Apply Average-Linkage Clustering, 
            and return cluster labels
        """
        self.fit(G)
        return(self.clusters)


    def __str__(self):
        return 'The clusters are: {0}'.format(self.clusters)