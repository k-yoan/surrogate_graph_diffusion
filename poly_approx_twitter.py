import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from math import exp, sqrt, log
from scipy.linalg import expm
import itertools
import time
from scipy.integrate import solve_ivp
from IPython.display import clear_output
from Diff import *
import sys
sys.path.append('content/equadratures')

import equadratures as eq
import matplotlib.lines as mlines
import cvxpy as cp
twitter = nx.read_weighted_edgelist('data/congress_network/congress.edgelist', )
pos = nx.spring_layout(twitter, iterations=15, seed=1721)
fig, ax = plt.subplots(figsize=(15, 9))
ax.axis("off")
plot_options = {"node_size": 10, "with_labels": False, "width": 0.15}
nx.draw_networkx(twitter, pos=pos, ax=ax, **plot_options)

def graph_perm_matrix(coms): 
    final_index = []
    coms_new = []
    for com in coms:
        coms_new.append(list(com))
        final_index+= [int(i) for i in com]
    initial_index = [i for i in range(len(final_index))]
    P = np.zeros((len(final_index), len(final_index)), dtype=int)

    P[final_index, initial_index] = 1
    return P, coms_new

def initialize_graph(G, num_coms=3):

  A = nx.adjacency_matrix(G).todense()
  L = nx.laplacian_matrix(G).toarray()

  n = G.number_of_nodes()
  x0 = np.ones(n)
  coms = nx.community.asyn_fluidc(G, num_coms, seed=0)
  P, coms = graph_perm_matrix(coms)
  A = P @ A @ P.T
  L = P @ L @ P.T
  # activ = np.random.choice(n, int(activ_ratio*n))
  # x0[activ] = 100
  sizes = []
  coms_list = []
  for com in coms:
    sizes.append(len(list(com)))
    coms_list.append(com)

  return G, n, A, L, coms_list, sizes, x0

G, n, A, L, coms, sizes, x0 = initialize_graph(twitter)

#no-time&edge dependent
def ed_diff_function(y):
  y = (y+1)/2
  c_bar = construct_cbar(y)
  print(sizes)
  C = construct_C(c_bar, sizes, n)
  D = construct_D(A, C, n)
  M = C*A - D

  diff_2 = ed_diffusion(M, x0, tau=0.01, T=1)

  return diff_2[2,-1]

#time&edge dependent diff equation
def ted_diff_function(y):
  c_bar = construct_cbar(y)

  C = construct_C(c_bar, sizes, n)
  M_t = construct_Mt(A, C, sizes, alpha=[2,3,4])

  diff = backward_euler_NC(x0, 0.01, M_t)

  return diff[2,-1]

# Custom solvers

solver = cp.MOSEK

def qcbp(A, b, eta=1e-6, **kwargs):
  n = A.shape[1]
  z = cp.Variable(n)
  b = np.reshape(b, (b.shape[0],))  # to avoid shape issues when defining the constraint
  objective = cp.Minimize(cp.norm1(z))
  constraints = [cp.norm2(A@z-b) <= eta]
  prob = cp.Problem(objective, constraints)
  result = prob.solve(solver=solver)

  return z.value #.reshape((n,1))


def ls(A, b, **kwargs):
  n = A.shape[1]
  z = cp.Variable(n)
  b = np.reshape(b, (b.shape[0],))  # to avoid shape issues when defining the constraint
  objective = cp.Minimize(cp.norm2(A@z-b))
  prob = cp.Problem(objective)
  result = prob.solve(solver=solver)

  return z.value #.reshape((n,1))

def get_average_rmse(m, my_method, dim=3, simuls=5, basis='total-order', ord=4):
  errors = np.array([])
  my_param_list = [eq.Parameter(distribution='uniform', order=ord, lower=-1.0, upper=1.0) for i in range(dim)]

  #if basis == 'hyperbolic-cross':
  my_basis = eq.Basis(basis, orders=[ord for _ in range(dim)])
  #else:
  #  my_basis = eq.Basis(basis)


  if my_method == qcbp:
    poly = eq.Poly([eq.Parameter(distribution='uniform', order=ord, lower=-1.0, upper=1.0) for _ in range(dim)], my_basis)
    index_set_size = my_basis.get_cardinality()
    M = 10 * index_set_size
    err_grid = np.random.uniform(-1, 1, size=(M, dim))
    A_err_grid = poly.get_poly(err_grid).T/sqrt(M)
    b_err_grid = eq.evaluate_model(err_grid, ed_diff_function)/sqrt(M)
    c_ref, _, _, _ = np.linalg.lstsq(A_err_grid, b_err_grid)

  for j in range(simuls):

    X_train = np.random.uniform(-1, 1, size=(m, dim))
    y_train = eq.evaluate_model(X_train, ed_diff_function)

    if my_method == qcbp:
      my_poly_ref = eq.Poly(my_param_list, my_basis, method='custom-solver',
            sampling_args={'mesh':'user-defined', 'sample-points':X_train, 'sample-outputs':y_train})
      A = my_poly_ref.get_poly(X_train).T
      eta_opt = np.linalg.norm(A@c_ref - y_train)
      my_poly = eq.Poly(my_param_list, my_basis, method='custom-solver',
          sampling_args={'mesh':'user-defined', 'sample-points':X_train, 'sample-outputs':y_train},
            solver_args={'solve':my_method, 'eta':eta_opt, 'verbose':False})
    elif my_method == ls:
      my_poly = eq.Poly(my_param_list, my_basis, method='custom-solver',
          sampling_args={'mesh':'user-defined', 'sample-points':X_train, 'sample-outputs':y_train},
            solver_args={'solve':my_method, 'verbose':False})

    my_poly.set_model()

    test_pts = np.random.uniform(-1, 1, size=(1000, dim))
    test_evals = eq.evaluate_model(test_pts, ed_diff_function)
    train_r2, test_r2 = my_poly.get_polyscore(X_test=test_pts, y_test=test_evals, metric='rmse')
    train_r2, test_r2

    errors = np.append(errors, test_r2)

  return np.mean(errors)


def get_average_rmse_varcoefs(m, my_method, dim=3, simuls=5, basis='total-order', ord=4):
  errors = np.array([])
  my_param_list = [eq.Parameter(distribution='uniform', order=ord, lower=-1.0, upper=1.0) for i in range(dim)]

  for j in range(simuls):
    if basis == 'hyperbolic-cross':
      my_basis = eq.Basis(basis, orders=[ord for _ in range(dim)])
    else:
      my_basis = eq.Basis(basis)
    X_train = np.random.uniform(-1, 1, size=(m, dim))
    y_train = eq.evaluate_model(X_train, ted_diff_function)
    my_poly = eq.Poly(parameters=my_param_list, basis=my_basis, method=my_method, \
                      sampling_args={'mesh':'user-defined', 'sample-points':X_train, \
                                    'sample-outputs':y_train})
    my_poly.set_model()

    test_pts = np.random.uniform(-1, 1, size=(500, dim))
    test_evals = eq.evaluate_model(test_pts, ted_diff_function)
    train_r2, test_r2 = my_poly.get_polyscore(X_test=test_pts, y_test=test_evals, metric='rmse')
    train_r2, test_r2

    errors = np.append(errors, test_r2)

  return np.mean(errors)

def conv(x, method, dim=3, simuls=5, basis='total-order', ord=4, verbose=False):
  Y = []

  for element in x:
    if verbose:
      start = time.time()
    Y.append(get_average_rmse(element, method, dim=dim, simuls=simuls, ord=ord, basis=basis))
    if verbose:
      end = time.time()
      print('m={} w/ {}, done: {} seconds.'.format(element, method, end-start))

  return Y

def conv_varcoefs(x, method, dim=3, simuls=5, basis='total-order', ord=4, verbose=False):
  Y = []

  for element in x:
    if verbose:
      start = time.time()
    Y.append(get_average_rmse_varcoefs(element, method, dim=dim, simuls=simuls, ord=ord, basis=basis))
    if verbose:
      end = time.time()
      print('m={} w/ {}, done: {} seconds.'.format(element, method, end-start))

  return Y

### Convergence plot of average RMSE vs. nb of samples, w/ TD index set

d = 6  # dimension
K = 3  # communities

basis = 'total-order'
name_basis = 'total-degree'
order = 8
cardinality = eq.basis.Basis(basis, orders=[order for _ in range(d)]).get_cardinality()  # 165
nb_samples = [10*i for i in range(5, 51)]

y_ls = conv(nb_samples, ls, dim=d, simuls=1, basis=basis, ord=order)
y_cs = conv(nb_samples, qcbp, dim=d, simuls=1, basis=basis, ord=order)

cutoff = 26

plt.plot(nb_samples[:cutoff], y_ls[:cutoff], 'orange', label='least squares')
plt.plot(nb_samples[:cutoff], y_cs[:cutoff], 'blue', label='compressed sensing')
plt.axvline(x=cardinality, color='grey', linestyle='--')
plt.yscale('log')
plt.xlabel('# of sample points')
plt.ylabel('Average RMSE')
plt.title('Order n={}, basis={}'.format(str(order), name_basis))
plt.legend()
plt.savefig('graphic convergence twitter.pdf')