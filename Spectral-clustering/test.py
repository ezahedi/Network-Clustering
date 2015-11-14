"""The algorithm is designed to find the min cut if you would like to find more
 clusters you need to change the variable 'n_cluster' iniside the 
 Spectral_Clustering.py as well.
 """

import Spectral_Clustering as SCl
import networkx as nx
import pylab
import matplotlib.pyplot as plt


n_cluster = 2

G=nx.Graph() #petersonii

G.add_edge('a','b',weight=4)
G.add_edge('b','c',weight=4)
G.add_edge('c','d',weight=4)
G.add_edge('d','e',weight=4)
G.add_edge('e','a',weight=4)

G.add_edge('f','g',weight=4)
G.add_edge('g','h',weight=4)
G.add_edge('h','i',weight=4)
G.add_edge('i','j',weight=4)
G.add_edge('j','f',weight=4)

G.add_edge('a','f',weight=.1)
G.add_edge('b','g',weight=.1)
G.add_edge('c','h',weight=.1)
G.add_edge('d','i',weight=.1)
G.add_edge('e','j',weight=.1)


def main(G):
    """draw the input graph and the colored out put graph
       determine the clusters after each level of merging
    """ 
    try:
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
        
        SC = SCl.SpecClust(G)
        pos=nx.spring_layout(G) # positions for all nodes
        node_colors = ['b','g','r','y','c','k','m'] 
        for i in range(len(G)):
                node_colors.append('w')
        Cluster_for_index = SC.fit_predict(G) 
        
        for i in range(n_cluster): 
            Nodes = []
            for j in Cluster_for_index[i]:
                Nodes.append(G.nodes()[j])
                
                
                
            nx.draw_networkx_nodes(G,pos,
                                   nodelist = Nodes,
                                   node_color=node_colors[i],
                                   node_size=80,
                               alpha=0.8)       
        edge_colors = 'k'
        edge_labels=dict([((u,v,),d['weight'])
                     for u,v,d in G.edges(data=True)])               
        nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
        nx.draw(G,pos, node_color = values, node_size=1,edge_color=edge_colors,edge_cmap=plt.cm.Reds)
        pylab.show()
        
        print SC.__str__()


    except SCl.SpecClust_error:
        
        print( "Got an imput error, please change the input and try it again." )

main(G)