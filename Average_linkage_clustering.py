# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 22:28:20 2015

@author: Emad Zahedi & Vahid Mirjalili
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




Iterations = 4
G=nx.Graph()

G.add_edge('a','b',weight=0.6)
G.add_edge('a','c',weight=0.2)
G.add_edge('c','d',weight=0.1)
G.add_edge('c','e',weight=0.7)
G.add_edge('c','f',weight=0.9)
G.add_edge('a','d',weight=0.3)



val_map = {'A': 1.0,
                   'D': 0.5714285714285714,
                              'H': 0.0}
values = [val_map.get(node, 0.45) for node in G.nodes()]
edge_colors = 'k'

edge_labels=dict([((u,v,),d['weight'])
             for u,v,d in G.edges(data=True)])
pos=nx.spring_layout(G) # positions for all nodes                
nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
nx.draw(G,pos, node_color = values, node_size=15,edge_color=edge_colors,edge_cmap=plt.cm.Reds)
pylab.show()


val_map = {'A': 1.0,
                   'D': 0.5714285714285714,
                              'H': 0.0}
values = [val_map.get(node, 0.45) for node in G.nodes()]
edge_colors = 'k'

edge_labels=dict([((u,v,),d['weight'])
             for u,v,d in G.edges(data=True)])
pos=nx.spring_layout(G) # positions for all nodes                
nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
nx.draw(G,pos, node_color = values, node_size=15,edge_color=edge_colors,edge_cmap=plt.cm.Reds)
pylab.show()




D_Matrix = nx.floyd_warshall_numpy(G) 
nodes_label = []
for i in range(len(G.nodes())):
     nodes_label.append(set(G.nodes()[i]))
A = np.vstack([D_Matrix, nodes_label])


def _Index(Matrix, a):
    for i in range(Matrix.shape[0]):
        for j in range(Matrix.shape[1]):
            if Matrix[i,j] == a:
                #print('(Matrix.shape[0])',(Matrix.shape[0]))
                #print('(Matrix.shape[1])',(Matrix.shape[1]))
                return(i,j)


D_matrix = A[:-1,:]
for iter in range(Iterations):
    
    minval = np.min(D_matrix[np.nonzero(D_matrix)])
    i,j = _Index(D_matrix, minval)
    
    
    print('D_matrix',D_matrix)
    print(minval)
    print( i,j)
    for k in range(D_matrix.shape[1]):
        A[i,k] = A[i,k] + A[j, k]
        A[k,i] = A[k,i] + A[k, j]
        A[i,i] = 0.0

    
    A[-1,:][0,i] = (A[-1,:][0,i]) | (A[-1,:][0,j]) 
    from copy import deepcopy 
    import copy
    B = copy.deepcopy(A)

    for k in range(D_matrix.shape[1]):
      for s in range(D_matrix.shape[1]):
       if s > k :
        B [s,k] = float(B[s,k])/ (len(B[-1,:][0,s])*len(B[-1,:][0,k]))
        B [k,s] = B [s,k]

    
    B = np.delete(B, j, 0)
    B = np.delete(B, j, 1)
    D_matrix = B[:-1,:] 
    print( ' D_matrix ',A )
      
    A = np.delete(A, j, 0)
    A = np.delete(A, j, 1)

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

plt.axis('off')
plt.savefig("labels_and_colors.png") # save as png
plt.show() # display
