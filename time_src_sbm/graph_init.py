''' Initializes Stochastic Block Model for experiment. '''

import numpy as np
import networkx as nx


def initialize_SBM(K, n_K, prob=0.04, activ_ratio=0.2):
  ### Inputs: # of communities K, # of nodes per community, probs of connections..?, ratio of active nodes for x0
  ### Outputs: graph G, # of nodes, adjacency matrix A and Laplacian L, vector of community sizes, initial conditions x0

  sizes = [n_K for _ in range(K)]
  probs = prob * np.ones((K,K)) + (1-prob) * np.identity(K)
  probs = probs.tolist()

  G = nx.stochastic_block_model(sizes, probs, seed=0)
  A = nx.adjacency_matrix(G).todense()
  L = nx.laplacian_matrix(G).toarray()

  n = G.number_of_nodes()
  x0 = np.zeros(n)
  activ = np.random.choice(n, int(activ_ratio*n))
  x0[activ] = 100

  return G, n, A, L, sizes, x0


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

def initialize_twitter(G, num_coms=3):

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

  return G, n, A, L, coms_list, sizes, x0

