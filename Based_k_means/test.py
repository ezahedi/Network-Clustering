"""
Input: A graph G with integer label's node and the number of clusters, n_clusters
Output: Output : n clusters and the colored graph based on k-means clustering. 
"""
import Based_k_means as BKM
import networkx as nx
import pylab
import matplotlib.pyplot as plt


n_clusters = 3


G=nx.Graph()

G.add_edge(0,1,weight=1)
G.add_edge(0,2,weight=1)
G.add_edge(2,4,weight=1)
G.add_edge(2,5,weight=1)
G.add_edge(0,3,weight=1)
G.add_edge(6,3,weight=1)
G.add_edge(7,3,weight=1)
G.add_edge(8,1,weight=1)
G.add_edge(9,1,weight=1)
G.add_edge(9,8,weight=1)
G.add_edge(6,7,weight=1)




def main(G):
    """draw the input graph and the colored out put graph
       determine the centroides and clusters
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
    
    
        km = BKM.KMeans(G, n_clusters, max_iter=100)
        
        
        pos=nx.spring_layout(G) # positions for all nodes
        node_colors = ['b','g','r','y','c','k','m'] 
        for i in range(len(G)):
                node_colors.append('w')
        # nodes
        Clust = km.fit_predict(G)[1]
        
        
        for item in range(n_clusters):
            for group in Clust[item]:
                nx.draw_networkx_nodes(G,pos,
                                       nodelist = Clust[item],
                                       node_color=node_colors[item],
                                       node_size=80,
                                   alpha=0.8)
        
        edge_colors = 'k'
        edge_labels=dict([((u,v,),d['weight'])
                     for u,v,d in G.edges(data=True)])               
        nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
        nx.draw(G,pos, node_color = values, node_size=1,edge_color=edge_colors,edge_cmap=plt.cm.Reds)
        pylab.show()
    
        print(km.__str__())


    except BKM.KMeanserror:
        
        print( "Got an imput error, please change the input and try it again." )

main(G)

