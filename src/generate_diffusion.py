''' Generate diffusion on Stochastic Block Models. '''


import matplotlib.pyplot as plt
import networkx as nx
from visualization import show_diffusion, plot_diffusion
from Diff import *
from graph_init import *
from argparse import ArgumentParser



def main(hparams):

	# Visualization parameters
	vmin = 0
	vmax = 100
	cmap = plt.cm.get_cmap('jet') # plt.cm.Blues/Reds/etc... OR plt.cm.get_cmap('jet/rainbow/etc...')
	node_size = 500
	labels = hparams.labels

	# Initialize the Stochastic Block Model
	K = hparams.nb_communities
	G, n, A, L, sizes, x0 = initialize_graph(K, hparams.nodes_per_comm)

	# Choice of plot to output
	if hparams.output == 'initial':
		# Static plot of the graph and initial conditions
		pos = nx.spring_layout(G)
		nx.draw(G, pos=pos, with_labels=labels, node_color=x0, cmap=cmap, node_size=node_size, vmin=vmin, vmax=vmax)
		plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)))
		plt.show()
	elif hparams.output == 'diffusion':
		pass
	elif hparams.output == 'curve':
		diff = diffusion(L, x0)
		plot_diffusion(diff, [i for i in range(n)])




if __name__ == '__main__':
	parser = ArgumentParser()

	parser.add_argument('--output', type=str, default='initial', help='Choose to display initial condition, diffusion GIF or curves of diffusion function')
	parser.add_argument('--nb_communities', type=int, default=2, help='Number of communities in the SBM')
	parser.add_argument('--nodes_per_comm', type=int, default=5, help='Number of nodes per community')
	parser.add_argument('--labels', action='store_true', help='Number of nodes per community')

	HPARAMS = parser.parse_args()


	main(HPARAMS)


