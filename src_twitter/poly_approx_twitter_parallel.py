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
  conf_vars = initialize_graph(twitter, num_coms = K)

### Convergence plot of average RMSE vs. nb of samples, w/ TD index set

  basis = 'total-order'
  name_basis = 'total-degree'
  
  cardinality = eq.basis.Basis(basis, orders=[order for _ in range(d)]).get_cardinality()  #35
  nb_samples = [5*i for i in range(1,13)]
  np.savetxt('nb_samples.txt',nb_samples)

  print('Least squares method')
  t_ls = conv(nb_samples, ['ls',ls], conf_vars, dim=d, simuls=simuls, basis=basis, ord=order)
  print('Compressed sensing method (QCBP)')
  t_cs = conv(nb_samples, qcbp, conf_vars, dim=d, simuls=simuls, basis=basis, ord=order)


  def get_mu(y, N_trial):
    return 1/N_trial * np.sum(np.log10(y), axis=1)

  def get_sig(y, mu, N_trial):
    return np.sqrt(1/(N_trial - 1) * np.sum((np.log10(y) - np.repeat(mu.reshape(mu.shape[0],1), y.shape[1], axis=1))**2, axis=1))
  
  mu_ls = get_mu(t_ls, 10)
  std_ls = get_sig(t_ls,mu_ls,10)
  mu_cs = get_mu(t_cs, 10)
  std_cs = get_sig(t_cs, mu_cs, 10)

    
  fig, ax = plt.subplots()
  ax.plot(nb_samples, 10**mu_ls, 'orange', label='Least squares')
  ax.plot(nb_samples, 10**mu_cs, 'blue', label='QCBP')
  ax.fill_between(nb_samples, 10**(mu_ls - std_ls), 10**(mu_ls + std_ls), color='papayawhip')
  ax.fill_between(nb_samples, 10**(mu_cs - std_cs), 10**(mu_cs + std_cs), color='lightblue')
  ax.axvline(x=cardinality, color='grey', linestyle='--')
  ax.set_yscale('log')
  ax.set_xlabel('# of sample points')
  ax.set_ylabel('Average RMSE')
  ax.set_title('Order n={}, basis={}'.format(str(order), name_basis))
  ax.legend()
  plt.savefig('twitter_plot.pdf')


if __name__ == '__main__':

  parser = ArgumentParser()
  parser.add_argument('--nb_communities', type=int, default=2, help='Number of communities in the Twitter dataset')
  parser.add_argument('--order', type=int, default=4, help='Order of the multi-index set')
  parser.add_argument('--n_trial', type=int, default=10, help='Number of rounds of computation for each method')
  
  HPARAMS = parser.parse_args()

  main(HPARAMS)