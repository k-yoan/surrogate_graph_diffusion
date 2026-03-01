''' A file to generate the convergence plot of average RMSE vs. number of sample points for different sized graphs. '''

import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import sys
sys.path.append('../content/equadratures')
import config
from graph_init import *
from poly_app import *
from visualization import *



def main(hparams):

	K = hparams.nb_communities
	d = int(K*(K+1)/2)  # the dimension is defined as a function of the number of communities

	# Setting the other hyperparameters
	basis = hparams.basis
	order = hparams.order
	start, end, step = hparams.start, hparams.end, hparams.step
	N_trial = hparams.n_trial

	# Create a list for the grid of number of sample points
	nb_samples = [i for i in range(start, end, step)]
	# Cardinality of the multi-index set (will be represented by a dashed line on the graph)
	cardinality = eq.basis.Basis(basis, orders=[order for _ in range(d)]).get_cardinality()

	if basis == 'total-order':
		name_basis = 'TD'
	elif basis == 'hyperbolic-cross':
		name_basis = 'HC'

	# Setting up the plot
	fig, ax = plt.subplots()
	colors = ['orange', 'blue', 'indigo']
	background_colors = ['papayawhip', 'lightblue', 'mediumpurple']

	# We need to initialize SBMs with varying numbers of nodes per community for a given number of communities

	for i in len(hparams.nodes_per_comm):
		conf_vars = initialize_SBM(K, hparams.nodes_per_comm[i])

		# Generate the average RMSE of the polynomial approximation with chosen method
		if hparams.method == 'LS':
			Y = conv(nb_samples, ls, conf_vars, dim=d, simuls=N_trial, basis=basis, ord=order)
		elif hparams.method == 'QCBP':
			Y = conv(nb_samples, qcbp, conf_vars, dim=d, simuls=N_trial, basis=basis, ord=order)
		elif hparams.method == 'wQCBP':
			Y = conv(nb_samples, weighted_qcbp, conf_vars, dim=d, simuls=N_trial, basis=basis, ord=order)

		# Visualize variance of average RMSE on the plot
		N = Y.shape[1]
		mu = get_mu(Y, N)
		sig = get_sig(Y, mu, N)

		ax.plot(nb_samples, 10**mu, colors[i], label='{} nodes per community'.format(hparams.nodes_per_comm[i]))
		ax.fill_between(nb_samples, 10**(mu - sig), 10**(mu + sig), color=background_colors[i])

	ax.axvline(x=cardinality, color='grey', linestyle='--')
	ax.set_yscale('log')
	ax.set_xlabel('# of sample points')
	ax.set_ylabel('Average RMSE')
	ax.set_title('d={}, order={}, basis={}, method={}'.format(d, order, name_basis, hparams.method))
	ax.legend()



# Argument parser to tune hyperparameters from the terminal
if __name__ == '__main__':
	parser = ArgumentParser()

	parser.add_argument('--nb_communities', type=int, default=2, help='Number of communities in the SBM')
	parser.add_argument('--nodes_per_comm', type=list, default=[5, 10, 15], help='List of number of nodes per community')
	parser.add_argument('--basis', type=str, default='total-order', help='Multi-index set to use as basis')
	parser.add_argument('--order', type=int, default=8, help='Order of the multi-index set')
	parser.add_argument('--method', type=str, default='LS', help='Method used to solve for the coefficients')
	parser.add_argument('--n_trial', type=int, default=3, help='Number of rounds of computation for each method')
	parser.add_argument('--start', type=int, default=25, help='Start of the range for the number of sample points')
	parser.add_argument('--end', type=int, default=325, help='End of the range for the number of sample points (non inclusive)')
	parser.add_argument('--step', type=int, default=25, help='Step size of the range for the number of sample points')


	HPARAMS = parser.parse_args()



