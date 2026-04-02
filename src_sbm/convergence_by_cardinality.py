''' A file to generate the convergence plot of average RMSE vs. number of sample points. '''

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

	# First, we need to initialize the Stochastic Block Model we will work with by generating the graph object and other variables 
	# number of nodes, adjacency and Laplacian matrices, initial conditions, and other SBM-related hyperparameters.

	conf_vars = initialize_SBM(K, hparams.nodes_per_comm)
	d = int(K*(K+1)/2)  # the dimension is defined as a function of the number of communities

	# Setting the other hyperparameters
	basis = hparams.basis
	nb_samples = hparams.nb_samples
	start, end, step = hparams.start, hparams.end, hparams.step
	N_trial = hparams.n_trial

	# Create a list for the grid of number of sample points
	nb_samples = 350
	# Cardinality of the multi-index set (will be represented by a dashed line on the graph)
	order_list = [i for i in range(start, end, step)]
	cardinality = [eq.basis.Basis(basis, orders=[order for _ in range(d)]).get_cardinality() for order in order_list]
	print(f'cardinality={cardinality}')
	if basis == 'total-order':
		name_basis = 'TD'
	elif basis == 'hyperbolic-cross':
		name_basis = 'HC'

	# Generate the average RMSE of the polynomial approximation for each method
	y_ls = conv_cardinality(order_list, [ls, 'ls'], conf_vars, dim=d, simuls=N_trial, basis=basis, m=nb_samples)
	y_cs = conv_cardinality(order_list, [qcbp, 'qcbp'], conf_vars, dim=d, simuls=N_trial, basis=basis, m=nb_samples)
	y_wcs = conv_cardinality(order_list, [weighted_qcbp, 'weighted_qcbp'], conf_vars, dim=d, simuls=N_trial, basis=basis, m=nb_samples)


	# Visualize variance of average RMSE on the plot
	N_ls = y_ls.shape[1]
	N_cs = y_cs.shape[1]
	N_wcs = y_wcs.shape[1]

	mu_ls = get_mu(y_ls, N_ls)
	sig_ls = get_sig(y_ls, mu_ls, N_ls)

	mu_cs = get_mu(y_cs, N_cs)
	sig_cs = get_sig(y_cs, mu_cs, N_cs)

	mu_wcs = get_mu(y_wcs, N_wcs)
	sig_wcs = get_sig(y_wcs, mu_wcs, N_wcs)


	# Plot
	fig, ax = plt.subplots()
	ax.plot(cardinality, 10**mu_ls, 'orange', label='Least squares')
	ax.plot(cardinality, 10**mu_cs, 'blue', label='QCBP')
	ax.plot(cardinality, 10**mu_wcs, 'indigo', label='wQCBP')
	ax.fill_between(cardinality, 10**(mu_ls - sig_ls), 10**(mu_ls + sig_ls), color='papayawhip')
	ax.fill_between(cardinality, 10**(mu_cs - sig_cs), 10**(mu_cs + sig_cs), color='lightblue')
	ax.fill_between(cardinality, 10**(mu_wcs - sig_wcs), 10**(mu_wcs + sig_wcs), color='mediumpurple')
	ax.set_yscale('log')
	ax.set_xlabel('Cardinality')
	ax.set_ylabel('Average RMSE')
	ax.set_title('d={}, nb_samples={}, basis={}'.format(d, nb_samples, name_basis))
	ax.legend()
	plt.tight_layout()
	plt.savefig('trials_nodes_per_comm{}_n_samples{}_basis{}.pdf'.format(hparams.nodes_per_comm, nb_samples, name_basis))



# Argument parser to tune hyperparameters from the terminal
if __name__ == '__main__':
	parser = ArgumentParser()

	parser.add_argument('--nb_communities', type=int, default=2, help='Number of communities in the SBM')
	parser.add_argument('--nodes_per_comm', type=int, default=5, help='Number of nodes per community')
	parser.add_argument('--basis', type=str, default='total-order', help='Multi-index set to use as basis')
	parser.add_argument('--nb_samples', type=int, default=8, help='Order of the multi-index set')
	parser.add_argument('--n_trial', type=int, default=20, help='Number of rounds of computation for each method')
	parser.add_argument('--start', type=int, default=3, help='Start of the range for the number of sample points')
	parser.add_argument('--end', type=int, default=15, help='End of the range for the number of sample points (non inclusive)')
	parser.add_argument('--step', type=int, default=1, help='Step size of the range for the number of sample points')


	HPARAMS = parser.parse_args()

	main(HPARAMS)


