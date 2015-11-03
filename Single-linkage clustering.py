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




Iterations = 3


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
'''






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


#print(A)




def _Index(Matrix, a):
    for i in range(Matrix.shape[0]):
        for j in range(Matrix.shape[1]):
            if Matrix[i,j] == a:
                #print('(Matrix.shape[0])',(Matrix.shape[0]))
                #print('(Matrix.shape[1])',(Matrix.shape[1]))
                return(i,j)
            




for iter in range(Iterations):
    D_matrix = A[:-1,:]
    minval = np.min(D_matrix[np.nonzero(D_matrix)])
    i,j = _Index(D_matrix, minval)
    
    print('(Matrix.shape[1])',D_matrix.shape)
    print(D_matrix)
    print(minval)
    print( i,j)
    for k in range(D_matrix.shape[1]):
        A[i,k] = min(A[i,k] , A[j, k])
        A[k,i] = min(A[k,i] , A[k, j])
        A[i,i] = 0.0
        A[-1,:][0,i] = (A[-1,:][0,i]) | (A[-1,:][0,j]) 
    A = np.delete(A, j, 0)
    A = np.delete(A, j, 1)

    
#print(A)












































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

#nx.draw_networkx_labels(G,pos,labels,font_size=16)

plt.axis('off')
plt.savefig("labels_and_colors.png") # save as png
plt.show() # display
'''        

  
nx.write_gml(G,"test.gml")
nx.write_edgelist(G, "test.edgelist")
fh=open("test.edgelist", 'w')
nx.write_edgelist(G,fh)
nx.write_edgelist(G,"test.edgelist.gz")

'''
        
