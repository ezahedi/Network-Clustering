"""
@authors: Emad Zahedi and Vahid Mirjalili
Input : A connected graph G
Output : n clusters based on k_means
"""
import random
import numpy as np
import networkx as nx
import operator
import pandas


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


def _cal_dist2center(G, Centeroids):
    """ Calculate the distances to cluster centers
    """  
    D_Matrix = nx.floyd_warshall_numpy(G)  
    Dict = {}
    for i in Centeroids:
        Dict[i] = []
        for j in range(len(G.nodes())):
            Dict[i].append(D_Matrix[i,j])
    return(Dict) 


def Dict2Matr(Dict):
    """ Change the dictionary to a matrix 
    """
    df    = pandas.DataFrame(Dict)
    Matr  = df.values
    Centr = df.columns
    Matr = np.vstack([Matr, Centr])
    return(Matr)


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


def _kmeans(G, n_clusters):
    """ Run multiple trials of k-means clustering,
        and outputt is the best centers, and cluster labels
    """
    D_Matrix = nx.floyd_warshall_numpy(G)  
    Old_Centroids  = set(_kmeans_init(G, n_clusters, method='balanced'))
    Clusters = _kmeans_run(G, n_clusters, _kmeans_init(G, n_clusters))
    New_Centroids = set(_update_centers(D_Matrix, Clusters, n_clusters))
    while True :
        if Old_Centroids == New_Centroids:
            return(New_Centroids) #,_kmeans_run(G, n_clusters, New_Centroids))
            break
        
        else:
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



##
## Class KMeanserror
##
class KMeanserror( ValueError ):
    
    pass

##
## Class KMeans
##
class KMeans(object):
    """
        KMeans Clustering
        Parameters
        -------
           G         : A connected graph 
           n_clusters: number of clusters (default = 2)          
           max_iter: maximum number of iterations (default = 100)
           
        Attibutes
        -------
           centers_  :  cluster centers
           clusters_   :  number of iterations for the best trial
           
        Methods
        ------- 
           fit(X): fit the model
           fit_predict(X): fit the model and return the centroids and clusters
    """

    def __init__(self, G, n_clusters=2, max_iter=100):
        
        self.G = G        
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, G):
        """ Apply KMeans Clustering
              X: dataset with feature vectors
        """
        self.centers_ =  Best_initial(self.G, self.n_clusters, self.max_iter)
        self.clusters = _kmeans_run(self.G, self.n_clusters, self.centers_)


    def fit_predict(self, G):
        """ Apply KMeans Clustering, 
            and return cluster labels
        """
        self.fit(G)
        return(self.centers_, self.clusters)


    def __str__(self):
        return 'The centroids are: {0} \nThe clusters are: {1}'.format(self.centers_, self.clusters)