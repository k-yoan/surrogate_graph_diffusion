''' Generate diffusion on Twitter dataset. '''

import networkx as nx
import numpy as np
from argparse import ArgumentParser
import sys

sys.path.append('../content/equadratures')
import equadratures as eq

from Diff import *
from graph_init_twitter import *
from utils_poly_app_twitter import *


def main(hparams):

  twitter = nx.read_weighted_edgelist('data/congress_network/congress.edgelist', )
  
  K = hparams.nb_communities # K=2
  simuls = hparams.n_trial
  order = hparams.order

  d = int(K*(K+1)/2) # d=6
  G, n, A, L, coms, sizes, x0 = initialize_graph(twitter, num_coms = K)

### Convergence plot of average RMSE vs. nb of samples, w/ TD index set

  basis = 'total-order'
  name_basis = 'total-degree'
  
  cardinality = eq.basis.Basis(basis, orders=[order for _ in range(d)]).get_cardinality()  #35
  nb_samples = [5*i for i in range(1,13)]
  np.savetxt('nb_samples.txt',nb_samples)

  print('Least squares method')
  mean_ls, std_ls = conv(nb_samples, ['ls',ls], dim=d, simuls=simuls, basis=basis, ord=order)
  print(f'Mean values: {mean_ls}, standard deviations: {std_ls}\n')
  print('Compressed sensing method (QCBP)')
  mean_cs, std_cs = conv(nb_samples, qcbp, dim=d, simuls=simuls, basis=basis, ord=order)
  print(f'Mean values: {mean_cs}, standard deviations: {std_cs}\n')

  np.savetxt('nb_samples.txt',nb_samples)
  np.savetxt('mean_ls.txt', mean_ls)
  np.savetxt('std_ls.txt', std_ls)
  np.savetxt('mean_cs.txt', mean_cs)
  np.savetxt('std_cs.txt', std_cs)

  plt.plot(nb_samples, mean_ls, 'orange', label='least squares')
  plt.fill_between(nb_samples, mean_ls - std_ls, mean_ls + std_ls, color='orange', alpha=0.4)
  plt.plot(nb_samples, mean_cs, 'blue', label='compressed sensing')
  plt.fill_between(nb_samples, mean_cs - std_cs, mean_cs + std_cs, color='blue', alpha=0.4)
  plt.axvline(x=cardinality, color='grey', linestyle='--')
  plt.yscale('log')
  plt.xlabel('# of sample points')
  plt.ylabel('Average RMSE')
  plt.title('Order n={}, basis={}'.format(str(order), name_basis))
  plt.legend()
  plt.savefig('graphic convergence twitter.pdf')
  print('End of the script')


if __name__ == '__main__':

  parser = ArgumentParser()
  parser.add_argument('--nb_communities', type=int, default=2, help='Number of communities in the Twitter dataset')
  parser.add_argument('--order', type=int, default=4, help='Order of the multi-index set')
  parser.add_argument('--n_trial', type=int, default=10, help='Number of rounds of computation for each method')
  
  HPARAMS = parser.parse_args()

  main(HPARAMS)