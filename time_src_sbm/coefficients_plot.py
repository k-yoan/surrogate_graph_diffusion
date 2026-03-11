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
	order = hparams.order
	start, end, step = hparams.start, hparams.end, hparams.step

	# Create a list for the grid of number of sample points
	nb_samples = 10000
	# Cardinality of the multi-index set (will be represented by a dashed line on the graph)
	cardinality = eq.basis.Basis(basis, orders=[order for _ in range(d)]).get_cardinality()
	print(f'cardinality = {cardinality}')

	if basis == 'total-order':
		name_basis = 'TD'
	elif basis == 'hyperbolic-cross':
		name_basis = 'HC'


	# Generate the average RMSE of the polynomial approximation for each method
	print('Method: Least Squares')
	coeff_ls = get_coefficients(nb_samples, ls, conf_vars, dim=d, basis=basis, ord=order)
	# print('Method: QCBP')
	# coeff_cs = get_coefficients(nb_samples, qcbp, conf_vars, dim=d, basis=basis, ord=order)
	# print('Method: weighted WCBP')
	# coeff_wcs = get_coefficients(nb_samples, weighted_qcbp, conf_vars, dim=d, basis=basis, ord=order)

	# Plot the results
	plt.figure(figsize=(10, 6))
	plt.semilogy(np.abs(coeff_ls), marker='o', linestyle='None', markersize=4, alpha=0.7)

	plt.title('d={}, order n={}, basis={}'.format(d, order, name_basis))
	plt.xlabel("Lexicographic")
	#plt.ylabel("Absolute Coefficient Value (Log Scale)")
	plt.grid(True, which="both", ls="-", alpha=0.5)
	plt.loglog()
	plt.show()
	plt.savefig('coefficients_plot_ls_unsorted.pdf')

	plt.figure(figsize=(10, 6))
	plt.semilogy(np.sort(np.abs(coeff_ls))[::-1], marker='o', linestyle='None', markersize=4, alpha=0.7)

	plt.title('d={}, order n={}, basis={}'.format(d, order, name_basis))
	plt.xlabel("Sorted")
	#plt.ylabel("Absolute Coefficient Value (Log Scale)")
	plt.grid(True, which="both", ls="-", alpha=0.5)
	plt.loglog()

	plt.show()
	plt.savefig('coefficients_plot_ls_sorted.pdf')
 
	# plt.figure(figsize=(10, 6))
	# plt.scatter(np.arange(len(coeff_cs)), coeff_cs, label='QCBP', marker='o')
	# plt.yscale('log')
	# plt.tight_layout()
	# plt.savefig('coefficients_plot_cs.pdf')

 
	# plt.figure(figsize=(10, 6))
	# plt.scatter(np.arange(len(coeff_wcs)), coeff_wcs, label='Weighted QCBP', marker='o')
	# plt.yscale('log')
	# plt.tight_layout()
	# plt.savefig('coefficients_plot_wcs.pdf')
# Argument parser to tune hyperparameters from the terminal
if __name__ == '__main__':
	parser = ArgumentParser()

	parser.add_argument('--nb_communities', type=int, default=2, help='Number of communities in the SBM')
	parser.add_argument('--nodes_per_comm', type=int, default=5, help='Number of nodes per community')
	parser.add_argument('--basis', type=str, default='total-order', help='Multi-index set to use as basis')
	parser.add_argument('--order', type=int, default=8, help='Order of the multi-index set')
	parser.add_argument('--n_trial', type=int, default=20, help='Number of rounds of computation for each method')
	parser.add_argument('--start', type=int, default=25, help='Start of the range for the number of sample points')
	parser.add_argument('--end', type=int, default=325, help='End of the range for the number of sample points (non inclusive)')#325
	parser.add_argument('--step', type=int, default=25, help='Step size of the range for the number of sample points')#25


	HPARAMS = parser.parse_args()

	main(HPARAMS)


