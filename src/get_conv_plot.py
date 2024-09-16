''' A file to generate the convergence plot of average RMSE vs. number of sample points. '''

import numpy as np
from graph_init import *
from poly_app import *
from visualization import *
from argparse import ArgumentParser



def main(hparams):

	K = hparams.nb_communities

	# First, we need to initialize the Stochastic Block Model we will work with by generating the graph object and other variables 
	# number of nodes, adjacency and Laplacian matrices, initial conditions, etc.

	G, n, A, L, sizes, x0 = initialize_graph(K, hparams.nodes_per_comm)
	d = int(K*(K+1)/2)

	# Setting the other hyperparameters
	basis = hparams.basis
	order = hparams.order
	start, end, step = hparams.start, hparams.end, hparams.step
	N_trial = hparams.n_trial


	nb_samples = [i for i in range(start, end, step)]
	cardinality = eq.basis.Basis(basis, orders=[order for _ in range(d)]).get_cardinality()
	if basis == 'total-order':
		name_basis = 'TD'
	elif basis == 'hyperbolic-cross':
		name_basis = 'HC'


	# Generates the average RMSE of the polynomial approximation for each method
	y_ls = conv(nb_samples, ls, dim=d, simuls=N_trial, basis=basis, ord=order)
	y_cs = conv(nb_samples, qcbp, dim=d, simuls=N_trial, basis=basis, ord=order)
	y_wcs = conv(nb_samples, weighted_qcbp, dim=d, simuls=N_trial, basis=basis, ord=order)


	# Visualize variance of average RMSE on the plot
	N_ls = y_ls.shape[1]
	N_cs = y_ls.shape[1]
	N_wcs = y_ls.shape[1]

	mu_ls = get_mu(y_ls, N_ls)
	sig_ls = get_sig(y_ls, mu_ls, N_ls)

	mu_cs = get_mu(y_cs, N_cs)
	sig_cs = get_sig(y_cs, mu_cs, N_cs)

	mu_wcs = get_mu(y_wcs, N_wcs)
	sig_wcs = get_sig(y_wcs, mu_wcs, N_wcs)


	# Plot
	fig, ax = plt.subplots()
	ax.plot(nb_samples, 10**mu_ls, 'orange', label='Least squares')
	ax.plot(nb_samples, 10**mu_cs, 'blue', label='QCBP')
	ax.plot(nb_samples, 10**mu_wcs, 'indigo', label='wQCBP')
	ax.fill_between(nb_samples, 10**(mu_ls - sig_ls), 10**(mu_ls + sig_ls), color='papayawhip')
	ax.fill_between(nb_samples, 10**(mu_cs - sig_cs), 10**(mu_cs + sig_cs), color='lightblue')
	ax.fill_between(nb_samples, 10**(mu_wcs - sig_wcs), 10**(mu_wcs + sig_wcs), color='mediumpurple')
	ax.axvline(x=cardinality, color='grey', linestyle='--')
	ax.set_yscale('log')
	ax.set_xlabel('# of sample points')
	ax.set_ylabel('Average RMSE')
	ax.set_title('d={}, order n={}, basis={}'.format(d, order, name_basis))
	ax.legend()



if __name__ == '__main__':
	parser = ArgumentParser()

	parser.add_argument('--nb_communities', type=int, default=2, help='Number of communities in the SBM')
	parser.add_argument('--nodes_per_comm', type=int, default=5, help='Number of nodes per community')
	parser.add_argument('--basis', type=str, default='total-order', help='Multi-index set to use as basis')
	parser.add_argument('--order', type=int, default=8, help='Order of the multi-index set')
	parser.add_argument('--n_trial', type=int, default=3, help='Number of rounds of computation for each method')
	parser.add_argument('--start', type=int, default=25, help='Start of the range for the number of sample points')
	parser.add_argument('--end', type=int, default=25, help='End of the range for the number of sample points (non inclusive)')
	parser.add_argument('--step', type=int, default=25, help='Step size of the range for the number of sample points')


	HPARAMS = parser.parse_args()

	main(HPARAMS)


