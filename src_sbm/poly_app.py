''' Functions for polynomial approximation using equadratures. '''

import equadratures as eq
import numpy as np
import time

import config
from Diff import *
from solvers import *



### The following function represents the mapping from the parameter y to the solution of the ED diffusion equation.

def ed_diff_function(y, n, A, sizes, x0):
  '''
  This function takes a diffusion parameter y and returns the solution to the ED diffusion equation
  at the 2nd node (the specific node does not matter, it can be changed) and at time T.
  Here, we take the solution at the last iterated time step.
  '''

  y = (y+1)/2 # transformation on y to take it from [-1,1] to [0,1]
  c_bar = construct_cbar(y)

  C = construct_C(c_bar, sizes, n)
  D = construct_D(A, C, n)
  # print(f'C is of shape {C.shape}')
  # print(f'A is of shape {A.shape}')
  # print(f'D is of shape {D.shape}')
  M = C*A - D
  
  # print(f'M is of shape {M.shape}')


  diff = ed_diffusion(M, x0, 0.1, T=1)

  return diff[2,-1]



### Function used to compute the accuracy of the polynomial approximation (the metric used is the Root Mean Square Error).

def get_average_rmse(m, my_method, conf_vars, dim=3, simuls=5, basis='total-order', ord=4):
  errors = np.array([])
  my_param_list = [eq.Parameter(distribution='uniform', order=ord, lower=-1.0, upper=1.0) for i in range(dim)]

  my_basis = eq.Basis(basis, orders=[ord for _ in range(dim)])

  G, n, A, L, sizes, x0 = conf_vars
  ed_diff = lambda x: ed_diff_function(x,n,A,sizes,x0)
  if my_method == qcbp or my_method == weighted_qcbp:
    poly = eq.Poly([eq.Parameter(distribution='uniform', order=ord, lower=-1.0, upper=1.0) for _ in range(dim)], my_basis)
    index_set_size = my_basis.get_cardinality()
    M = 10 * index_set_size
    err_grid = np.random.uniform(-1, 1, size=(M, dim))
    A_err_grid = poly.get_poly(err_grid).T/sqrt(M)
    b_err_grid = eq.evaluate_model(err_grid, ed_diff)/sqrt(M)
    c_ref, _, _, _ = np.linalg.lstsq(A_err_grid, b_err_grid)

  for j in range(simuls):

    start = time.time()

    X_train = np.random.uniform(-1, 1, size=(m, dim))
    y_train = eq.evaluate_model(X_train, ed_diff)

    end = time.time()
    elapsed = end - start
    print('Training data generated in {} seconds.'.format(elapsed))

    start = time.time()

    if my_method == qcbp:
      # oracle parameter
      my_poly_ref = eq.Poly(my_param_list, my_basis, method='least-squares',
            sampling_args={'mesh':'user-defined', 'sample-points':X_train, 'sample-outputs':y_train})
      S = my_poly_ref.get_poly(X_train).T
      eta_opt = np.linalg.norm(S@c_ref - y_train)
      # define poly
      my_poly = eq.Poly(my_param_list, my_basis, method='custom-solver',
          sampling_args={'mesh':'user-defined', 'sample-points':X_train, 'sample-outputs':y_train},
            solver_args={'solve':my_method, 'eta':eta_opt, 'verbose':False})
    elif my_method == weighted_qcbp:
      # oracle parameter
      my_poly_ref = eq.Poly(my_param_list, my_basis, method='least-squares',
            sampling_args={'mesh':'user-defined', 'sample-points':X_train, 'sample-outputs':y_train})
      S = my_poly_ref.get_poly(X_train).T
      eta_opt = np.linalg.norm(S@c_ref - y_train)
      # weights computation
      I = my_basis.get_elements().T
      size_I = I.shape
      weights = np.prod(np.sqrt(2 * I + np.ones(size_I)), axis=0)
      # define poly
      my_poly = eq.Poly(my_param_list, my_basis, method='custom-solver',
          sampling_args={'mesh':'user-defined', 'sample-points':X_train, 'sample-outputs':y_train},
            solver_args={'solve':my_method, 'eta':eta_opt, 'w':weights, 'verbose':False})
    elif my_method == ls:
      my_poly = eq.Poly(my_param_list, my_basis, method='custom-solver',
          sampling_args={'mesh':'user-defined', 'sample-points':X_train, 'sample-outputs':y_train},
            solver_args={'solve':my_method, 'verbose':False})

    my_poly.set_model()

    end = time.time()
    elapsed_1 = end - start
    print('Coefficients obtained in {} seconds'.format(elapsed_1))
    start_2 = time.time()
    #print(my_poly.get_coefficients())


    test_pts = np.random.uniform(-1, 1, size=(1000, dim))
    test_evals = eq.evaluate_model(test_pts, ed_diff)
    train_r2, test_r2 = my_poly.get_polyscore(X_test=test_pts, y_test=test_evals, metric='rmse')
    train_r2, test_r2

    end_2 = time.time()
    elapsed_2 = end_2 - start_2
    print('RMSE computed in {} seconds. Round {} completed.'.format(elapsed_2, str(j)))


    errors = np.append(errors, test_r2)

  return errors


# Function used to generate the data for the convergence plot "Average RMSE vs. number of sample points".

def conv(x, method, conf_vars, dim=3, simuls=5, basis='total-order', ord=4, verbose=False):
  Y = []

  for element in x:
    if verbose:
      start = time.time()
    Y.append(get_average_rmse(element, method, conf_vars,dim=dim, simuls=simuls, ord=ord, basis=basis))
    if verbose:
      end = time.time()
      print('m={} w/ {}, done: {} seconds.'.format(element, method, end-start))

  return np.array(Y)




