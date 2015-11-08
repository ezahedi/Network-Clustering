"""
@authors: Emad Zahedi & Vahid Mirjalili
Input: A graph G.
Output: Clusters obtained by Single linkage method pluse drawn colored graphs 
"""
import numpy as np
import networkx as nx

  
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
    
 
def Single_Linkage(G, Iterations):
    """Give the clusters based on Average linkage method"""          
    A = Dis_Clus(G)
    for iter in range(Iterations):
        D_matrix = A[:-1,:]
        minval = np.min(D_matrix[np.nonzero(D_matrix)])
        i,j = _Index(D_matrix, minval)
        for k in range(D_matrix.shape[1]):
            A[i,k] = min(A[i,k] , A[j, k])
            A[k,i] = min(A[k,i] , A[k, j])
            A[i,i] = 0.0
            A[-1,:][0,i] = (A[-1,:][0,i]) | (A[-1,:][0,j]) 
        A = np.delete(A, j, 0)
        A = np.delete(A, j, 1)
    return (A[-1,:])
    
    

##
## Class KMeanserror
##
class Single_linkage_Error( ValueError ):
    
    pass

##
## Class KMeans
##
class Single_linkage(object):
    """
        Single_linkage Clustering
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
        """ Apply Single-Linkage Clustering
              G: A weighted graph
        """
        self.clusters = Single_Linkage(self.G, self.Iterations)


    def fit_predict(self, G):
        """ Apply Single-Linkage Clustering, 
            and return clusters 
        """
        self.fit(G)
        return(self.clusters)


    def __str__(self):
        return 'The clusters are: {0}'.format(self.clusters)