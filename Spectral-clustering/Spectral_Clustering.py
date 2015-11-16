"""
@authors: Emad Zahedi and Vahid Mirjalili
Input : A connected graph G
Output : n clusters based on Spectral Clustering
"""
import scipy.spatial
import numpy as np
import networkx as nx


##
## Class Soectral clustering
##
class SpecClust_error( ValueError ):
    
    pass

##
## Class Spectral Clustering
##
class SpecClust(object):
    """
    Spectral Clustering
    Parameters
    ----------------
           G : A given graph
           n_clusters: number of clusters (default = 2)
           n_trials: number of trial random centroid initialization (default = 10)
           max_iter: maximum number of iterations (default = 100)
           tol: tolerance (default = 0.0001)
    Attibutes
    -------
       Classes_   :  cluster labels for each node
       centers_  :  cluster centers
       sse_arr_  :  array of SSE values for each cluster
       n_iter_   :  number of iterations for the best trial
        Methods
        ------- 
           fit(X): fit the model
           fit_predict(X): fit the model and return the cluster labels
    """
    def __init__(self, G, n_clusters=2, n_trials=10, max_iter=100, tol=0.001):
        self.G         = G
        self.n_cluster = n_clusters
        self.n_trials  = n_trials
        self.max_iter  = max_iter
        self.tol       = tol
  

    def Laplacian(self):
        """Finding the laplacian matrix for a graph G"""
        D = nx.adjacency_matrix(self.G)   
        for i in range(len(self.G.nodes())):
            Sum = 0
            for k in range(len(self.G.nodes())):
                Sum += D[i,k]
            D[i,i] = -1*Sum        
        L = -1* D          
        return(L.todense())
 
   
    
    def Eigen_valu(self, Matrix):
        """Creating a dictionary where
        keys   are eigen values, and
        values are corresponding eigen vector"""
        Eigs = np.linalg.eig(Matrix)  
        eig_values = Eigs[0] 
        eig_vectors = Eigs[1] 
        Dict = {}
        i = 0
        while i < len(self.G.nodes()): 
            Dict[eig_values[i]] = eig_vectors[:,i] 
            i += 1           
        return(Dict)    
    
 
   
    def _assign_clusters(self, X, centers):
        """ Assignment Step:
               assign each point to the closet cluster center
        """
        dist2cents = scipy.spatial.distance.cdist(X, centers, metric='seuclidean')
        membs = np.argmin(dist2cents, axis=1)    
        return(membs)        
    
    
    def _kmeans_init(self, X, method='balanced'):
        """ Initialize k=n_clusters centroids randomly
        """
        n_samples = X.shape[0]
        cent_idx = np.random.choice(n_samples, replace=False, size=self.n_cluster)
        
        centers = X[cent_idx,:]
        mean_X = np.mean(X, axis=0)
        
        if method == 'balanced':
            centers[self.n_cluster-1] = self.n_cluster*mean_X - np.sum(centers[:(self.n_cluster-1)], axis=0)
        
        return (centers)
    
    

    def _cal_dist2center(self, X, center):
        """ Calculate the SSE to the cluster center
        """
        dmemb2cen = scipy.spatial.distance.cdist(X, center.reshape(1,X.shape[1]), metric='seuclidean')
        return(np.sum(dmemb2cen)) 



    def _update_centers(self, X, membs):
        """ Update Cluster Centers:
               calculate the mean of feature vectors for each cluster
        """
        centers = np.empty(shape=(self.n_cluster, X.shape[1]), dtype=float)
        sse = np.empty(shape=self.n_cluster, dtype=float)
        for clust_id in range(self.n_cluster):
            memb_ids = np.where(membs == clust_id)[0]
    
            if memb_ids.shape[0] == 0:
                memb_ids = np.random.choice(X.shape[0], size=1)
                #print("Empty cluster replaced with ", memb_ids)
            centers[clust_id,:] = np.mean(X[memb_ids,:], axis=0)
            
            sse[clust_id] = self._cal_dist2center(X[memb_ids,:], centers[clust_id,:]) 
            
        return(centers, sse)


    def _kmeans_run(self,X):
        """ Run a single trial of k-means clustering
            on dataset X, and given number of clusters
        """
        membs = np.empty(shape=X.shape[0], dtype=int)
        centers = self._kmeans_init(X, self.n_cluster)
    
        sse_last = 9999.9
        n_iter = 0
        for it in range(1,self.max_iter):
            membs = self._assign_clusters(X, centers)
            centers,sse_arr = self._update_centers(X, membs)
            sse_total = np.sum(sse_arr)
            if np.abs(sse_total - sse_last) < self.tol:
                n_iter = it
                break
            sse_last = sse_total
    
        return(centers, membs, sse_total, sse_arr, n_iter)

    
    def _kmeans(self, X):#X, n_clusters, max_iter, n_trials, tol):
        """ Run multiple trials of k-means clustering,
            and outputt he best centers, and cluster labels
        """
        n_samples, n_features = X.shape[0], X.shape[1]
    
        centers_best = np.empty(shape=(self.n_cluster,n_features), dtype=float)
        labels_best  = np.empty(shape=n_samples, dtype=int)
        for i in range(self.n_trials):
            centers, labels, sse_tot, sse_arr, n_iter  = self._kmeans_run(X)
            if i==0:
                sse_tot_best = sse_tot
                sse_arr_best = sse_arr
                n_iter_best = n_iter
                centers_best = centers.copy()
                labels_best  = labels.copy()
            if sse_tot < sse_tot_best:
                sse_tot_best = sse_tot
                sse_arr_best = sse_arr
                n_iter_best = n_iter
                centers_best = centers.copy()
                labels_best  = labels.copy()
                
        return(centers_best, labels_best, sse_arr_best, n_iter_best)
       
    
    def Spectral_clustering(self):
        """classify the least non zero eigen vector based on,
        spectral clustering algorithm"""
        Matrix = self.Laplacian()
        Eigs = np.linalg.eig(Matrix) 
        Sorting = np.sort(Eigs[0])
        least_eig_valu = Sorting[1] # not zero
        least_eig_vec = np.array(self.Eigen_valu(Matrix)[least_eig_valu])
        Classes = list(self._kmeans(least_eig_vec)[1])      
        return Classes    
        
    
    def getting_index(self, list):
        """getting indices to elements in a dictionary""" 
        Dict = {}
        i = 0
        for a in list:
            if a not in Dict:
                Dict[a] = [i]
            else:
                Dict[a].append(i)
            i +=1
        return(Dict)    
      
    
    def fit(self, G):
        """ Apply Spectral Clustering
              G: is a given graph
        """
        self.Classes = self.Spectral_clustering()
        self.Cluster_for_index = self.getting_index(self.Classes)


    def fit_predict(self, G):
        """ Apply Spectral Clustering, 
            and return clusters
        """
        self.fit(G)
        return(self.Cluster_for_index)

        
    def __str__(self):
        return 'The clusters are: {0}'.format(self.Cluster_for_index)