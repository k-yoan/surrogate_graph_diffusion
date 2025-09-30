''' Generate diffusion on Twitter dataset. '''

import networkx as nx
import numpy as np
from argparse import ArgumentParser
import sys

sys.path.append('../content/equadratures')
import equadratures as eq

from Diff import *
from graph_init_twitter import *
from utils_data_generation import *


def main(hparams, dataset, load=False):
  if dataset == 'twitter':
    network = nx.read_weighted_edgelist('data/congress_network/congress.edgelist', )
  elif dataset == 'facebook':
    network = nx.read_edgelist('data/facebook_combined.txt')
  K = hparams.nb_communities # K=2

  d = int(K*(K+1)/2) # d=3

  
  if load==True:
    conf_vars_dict = np.load(f'{dataset}/{dataset}_conf_vars.npz')
    conf_vars = conf_vars_dict['n'], conf_vars_dict['A'], conf_vars_dict['sizes'], conf_vars_dict['x0']
    print(f'{dataset} dataset loaded')
  else:
    start_time = time.time()
    n, A, sizes, x0 = initialize_graph(network, dataset, num_coms=K)
    print(f'Initializing the {dataset} graph takes {time.time()-start_time} seconds\n')
    conf_vars_dict={}
    conf_vars_dict['n'] = n 
    conf_vars_dict['A'] = A
    conf_vars_dict['sizes']= sizes 
    conf_vars_dict['x0'] = x0
    np.savez(f'{dataset}/{dataset}_conf_vars', **conf_vars_dict)
    print(f'{dataset} dataset generated\n')
    conf_vars = n, A, sizes, x0
   

### Convergence plot of average RMSE vs. nb of samples, w/ TD index set

  nb_samples = [3500]#9150= 8800 + 35*10

  conv(nb_samples, conf_vars, dim=d, dataset=dataset)
  print('finished')

if __name__ == '__main__':

  parser = ArgumentParser()
  parser.add_argument('--nb_communities', type=int, default=2, help='Number of communities in the Twitter dataset')
  parser.add_argument('--n_trial', type=int, default=1, help='Number of rounds of computation for each method')
  
  HPARAMS = parser.parse_args()

  main(HPARAMS, 'facebook', load=True)