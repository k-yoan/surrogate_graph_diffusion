
import numpy as np
import networkx as nx

np.random.seed(0)
def graph_perm_matrix(coms): 
  final_index = []
  coms_new = []
  for com in coms:
      coms_new.append(list(com))
      final_index+= [int(i) for i in com]
  initial_index = [i for i in range(len(final_index))]
  P = np.zeros((len(final_index), len(final_index)), dtype=int)

  P[final_index, initial_index] = 1
  return P, coms_new

def initialize_graph(G, dataset, num_coms=2, load=False):

  A = nx.adjacency_matrix(G).todense()
  L = nx.laplacian_matrix(G).toarray()
  activ_ratio = 0.2
  n = G.number_of_nodes()
  x0 = np.ones(n)
  coms = nx.community.asyn_fluidc(G, num_coms, seed=0)
  P, coms = graph_perm_matrix(coms)
  A = P @ A @ P.T
  L = P @ L @ P.T
  activ = np.random.choice(n, int(activ_ratio*n))
  x0[activ] = 100
  sizes = []
  coms_list = []
  for com in coms:
    sizes.append(len(list(com)))
    coms_list.append(com)

  return n, A, sizes, x0