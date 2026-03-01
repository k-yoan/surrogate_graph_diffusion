''' Visuals. '''

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
#from matplotlib.cm import get_cmap
#from PIL import Image


# Functions for diffusion on graphs


def show_diffusion(data):

  '''
  ARGUMENTS:
    data : 2D Numpy array. Matrix of states of the nodes as time progresses.

  OUTPUT:
    none

  The function plots each iteration of the diffusion solver, so we get a time sequence of the "quantities" in each node of the graph.
  '''

  for i in range(data.shape[1]):
    nx.draw(G, pos=pos, with_labels=labels, node_color = np.array(data[:,i]), node_size=node_size, vmin=vmin, vmax=vmax, cmap = cmap)
    #plt.title('Diffusion at t = ' + str(round(i/10, 2)))
    plt.title('Iteration ' + str(i))
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)))
    plt.show()


def plot_diffusion(data, nodes):

  '''
  ARGUMENTS:
    data : 2D Numpy array. Matrix of states of the nodes as time progresses.
    nodes : list of integers in the range 0 to n. Contains the labels of the nodes we want to see in the plot.

  OUTPUT:
    none

  This function plots the evolutions of the "quantities" in the chosen nodes as a function of time.
  '''

  for node in nodes:
    plt.plot(data[node, :], label=str(node))
    plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize='small')
  plt.xlabel('t')
  plt.ylabel('$u_i(t)$')
  plt.show()





# Visualization for plots in the polynomial approximation section


def get_mu(y, N_trial):
  return 1/N_trial * np.sum(np.log10(y), axis=1)

def get_sig(y, mu, N_trial):
  return np.sqrt(1/(N_trial - 1) * np.sum((np.log10(y) - np.repeat(mu.reshape(mu.shape[0],1), y.shape[1], axis=1))**2, axis=1))

