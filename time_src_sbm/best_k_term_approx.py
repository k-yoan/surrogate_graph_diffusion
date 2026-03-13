''' A file to generate the convergence plot of average RMSE vs. number of sample points. '''

import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('../content/equadratures')
import config
from graph_init import *
from poly_app import *
from visualization import *

def best_k_term_approx(coeffs, k):

	# Get indices of the k largest absolute values		
	return np.linalg.norm(np.sort(np.abs(coeffs))[::-1][k:])


def main(hparams):

	K = hparams.nb_communities
	tr = hparams.tr

	# First, we need to initialize the Stochastic Block Model we will work with by generating the graph object and other variables 
	# number of nodes, adjacency and Laplacian matrices, initial conditions, and other SBM-related hyperparameters.

	conf_vars = initialize_SBM(K, hparams.nodes_per_comm)
	d = int(K*(K+1)/2)  # the dimension is defined as a function of the number of communities

	# Setting the other hyperparameters
	basis = hparams.basis
	order = hparams.order

	# Create a list for the grid of number of sample points
	nb_samples = hparams.nb_samples
	# Cardinality of the multi-index set (will be represented by a dashed line on the graph)
	cardinality = eq.basis.Basis(basis, orders=[order for _ in range(d)]).get_cardinality()
	print(f'cardinality = {cardinality}')

	if basis == 'total-order':
		name_basis = 'TD'
	elif basis == 'hyperbolic-cross':
		name_basis = 'HC'


	# Generate the average RMSE of the polynomial approximation for each method
	print('Method: Least Squares')
	if os.path.exists(f'time_coeff_ls_order{order}_{name_basis}.npy'):
		coeff_ls = np.load(f'time_coeff_ls_order{order}_{name_basis}.npy')
	else:
		coeff_ls = get_coefficients(nb_samples, ls, conf_vars, dim=d, basis=basis, ord=order)
		np.save(f'time_coeff_ls_order{order}_{name_basis}.npy', coeff_ls)
	print('Method: QCBP')
	if os.path.exists(f'time_coeff_cs_order{order}_{name_basis}.npy'):
		coeff_cs = np.load(f'time_coeff_cs_order{order}_{name_basis}.npy')
	else:
		coeff_cs = get_coefficients(nb_samples, qcbp, conf_vars, dim=d, basis=basis, ord=order)
		np.save(f'time_coeff_cs_order{order}_{name_basis}.npy', coeff_cs)
	print('Method: weighted WCBP')
	if os.path.exists(f'time_coeff_wcs_order{order}_{name_basis}.npy'):
		coeff_wcs = np.load(f'time_coeff_wcs_order{order}_{name_basis}.npy')
	else:
		coeff_wcs = get_coefficients(nb_samples, weighted_qcbp, conf_vars, dim=d, basis=basis, ord=order)
		np.save(f'time_coeff_wcs_order{order}_{name_basis}.npy', coeff_wcs)

	best_array_ls = np.array([best_k_term_approx(coeff_ls, k) for k in range(1, int(cardinality))])
	best_array_cs = np.array([best_k_term_approx(coeff_cs, k) for k in range(1, int(cardinality))])
	best_array_wcs = np.array([best_k_term_approx(coeff_wcs, k) for k in range(1, int(cardinality))])

	# Plot the results
	plt.figure(figsize=(10, 6))
	plt.semilogy(best_array_ls[:tr], marker='o', linestyle='None', markersize=4, alpha=0.7)
	plt.semilogy(best_array_cs[:tr], marker='s', linestyle='None', markersize=4, alpha=0.7)
	plt.semilogy(best_array_wcs[:tr], marker='^', linestyle='None', markersize=4, alpha=0.7)

	plt.title('d={}, order n={}, basis={}'.format(d, order, name_basis))
	plt.xlabel("Number of nonzero terms $k$") 
	plt.ylabel("Best $k$-term approximation error")
	plt.grid(True, which="both", ls="-", alpha=0.5)
	plt.loglog()
	plt.legend(['Least Squares', 'QCBP', 'Weighted WCBP'])
	plt.tight_layout()
	plt.show()
	plt.savefig(f'time_best_k_term_{order}_{name_basis}.pdf')
 
 
	plt.figure(figsize=(10, 6))
	plt.semilogy(np.abs(coeff_ls)[:tr], marker='o', linestyle='None', markersize=4, alpha=0.7)
	plt.semilogy(np.abs(coeff_cs)[:tr], marker='s', linestyle='None', markersize=4, alpha=0.7)
	plt.semilogy(np.abs(coeff_wcs)[:tr], marker='^', linestyle='None', markersize=4, alpha=0.7)
	plt.title('d={}, order n={}, basis={}'.format(d, order, name_basis))
	plt.xlabel("Lexicographic")
	#plt.ylabel("Absolute Coefficient Value (Log Scale)")
	plt.grid(True, which="both", ls="-", alpha=0.5)
	plt.loglog()
	plt.legend(['Least Squares', 'QCBP', 'Weighted WCBP'])
	plt.show()
	plt.savefig(f'time_coefficients_plot_unsorted_{order}_{name_basis}.pdf')

	plt.figure(figsize=(10, 6))
	plt.semilogy(np.sort(np.abs(coeff_ls))[::-1][:tr], marker='o', linestyle='None', markersize=4, alpha=0.7)
	plt.semilogy(np.sort(np.abs(coeff_cs))[::-1][:tr], marker='s', linestyle='None', markersize=4, alpha=0.7)
	plt.semilogy(np.sort(np.abs(coeff_wcs))[::-1][:tr], marker='^', linestyle='None', markersize=4, alpha=0.7)	
	plt.title('d={}, order n={}, basis={}'.format(d, order, name_basis))
	plt.xlabel("Sorted")
	#plt.ylabel("Absolute Coefficient Value (Log Scale)")
	plt.grid(True, which="both", ls="-", alpha=0.5)
	plt.loglog()
	plt.legend(['Least Squares', 'QCBP', 'Weighted WCBP'])
	plt.show()
	plt.savefig(f'time_coefficients_plot_sorted_{order}_{name_basis}.pdf')
 
# Argument parser to tune hyperparameters from the terminal
if __name__ == '__main__':
	parser = ArgumentParser()

	parser.add_argument('--nb_communities', type=int, default=2, help='Number of communities in the SBM')
	parser.add_argument('--nodes_per_comm', type=int, default=5, help='Number of nodes per community')
	parser.add_argument('--basis', type=str, default='total-order', help='Multi-index set to use as basis')
	parser.add_argument('--order', type=int, default=30, help='Order of the multi-index set')
	parser.add_argument('--nb_samples', type=int, default=20000, help='Number of sample points to use for the polynomial approximation')
	parser.add_argument('--tr', type=int, default=400, help='Number of points to plot for the best k-term approximation error')
	HPARAMS = parser.parse_args()

	main(HPARAMS)


