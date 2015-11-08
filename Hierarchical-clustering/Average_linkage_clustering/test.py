import Average_linkage_clustering as AVG
import networkx as nx
import pylab
import matplotlib.pyplot as plt
import numpy as np

n_clusters = 3


G=nx.Graph()

G.add_edge('a','b',weight=0.6)
G.add_edge('a','c',weight=0.2)
G.add_edge('c','d',weight=0.1)
G.add_edge('c','e',weight=0.7)
G.add_edge('c','f',weight=0.9)
G.add_edge('a','d',weight=0.3)



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
    
        for i in range(len(G.nodes())):
              
            Iterations = i 
            AL = AVG.Average_linkage(G, Iterations)
            #print(AL.__str__())
            pos=nx.spring_layout(G) # positions for all nodes
            node_colors = ['b','g','r','y','c','k','m'] 
            
            # nodes
            C_list = AL.fit_predict(G)[-1,:]
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
            print "in level :",i 
            print AL.__str__()


    except AVG.Average_linkage_Error:
        
        print( "Got an imput error, please change the input and try it again." )

main(G)