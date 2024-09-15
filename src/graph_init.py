''' Initializes Stochastic Block Model for experiment. '''

import numpy as np
import networkx as nx


def initialize_graph(K, n_K, prob=0.04, activ_ratio=0.2):
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


