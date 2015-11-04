# -*- coding: utf-8 -*-
"""
@authors: Emad Zahedi & Vahid Mirjalili
Input: A graph G.
Output: Clusters obtained by Complete linkage method pluse drawn colored graphs 
Note: If you are looking for specific level, like 'k', of clusters remove the for loop and initiate Iteration = 'k'
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pylab

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

for i in range(len(G.nodes())):
    print("in level :",i)  
    Iterations  = i  
    D_Matrix    = nx.floyd_warshall_numpy(G) 
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
    print('Clusters are :',np.array(A[-1,:]))
    
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
     
    plt.axis('off')
    plt.savefig("labels_and_colors.png") # save as png
    plt.show() # display
