import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from math import sqrt
import time
from multiprocess import Pool
from scipy.integrate import solve_ivp
from IPython.display import clear_output
import matplotlib.lines as mlines
import cvxpy as cp
from Diff import *
import equadratures as eq

#no-time&edge dependent
def ed_diff_function(y):
  y = (y+1)/2
  c_bar = construct_cbar(y)
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



def qcbp(A, b, eta=1e-6, **kwargs):
  n = A.shape[1]
  z = cp.Variable(n)
  b = np.reshape(b, (b.shape[0],))  # to avoid shape issues when defining the constraint
  objective = cp.Minimize(cp.norm1(z))
  constraints = [cp.norm2(A@z-b) <= eta]
  prob = cp.Problem(objective, constraints)
  result = prob.solve(solver=cp.MOSEK)

  return z.value #.reshape((n,1))


def ls(A, b, **kwargs):
  n = A.shape[1]
  z = cp.Variable(n)
  b = np.reshape(b, (b.shape[0],))  # to avoid shape issues when defining the constraint
  objective = cp.Minimize(cp.norm2(A@z-b))
  prob = cp.Problem(objective)
  result = prob.solve(solver=cp.MOSEK)

  return z.value #.reshape((n,1))

def get_average_rmse(m, my_method, dim=3, simuls=5, basis='total-order', ord=4):
  errors = np.array([])
  my_param_list = [eq.Parameter(distribution='uniform', order=ord, lower=-1.0, upper=1.0) for i in range(dim)]

  #if basis == 'hyperbolic-cross':
  my_basis = eq.Basis(basis, orders=[ord for _ in range(dim)])
  #else:
  #  my_basis = eq.Basis(basis)


  if my_method[0] == 'qcbp':
    poly = eq.Poly([eq.Parameter(distribution='uniform', order=ord, lower=-1.0, upper=1.0) for _ in range(dim)], my_basis)
    index_set_size = my_basis.get_cardinality()
    M = 10 * index_set_size
    err_grid = np.random.uniform(-1, 1, size=(M, dim))
    A_err_grid = poly.get_poly(err_grid).T/sqrt(M)
    b_err_grid = eq.evaluate_model(err_grid, ed_diff_function)/sqrt(M)
    c_ref, _, _, _ = np.linalg.lstsq(A_err_grid, b_err_grid)
  #   print(my_method)
  # elif my_method[0] == 'ls':
  #   print(my_method[1])

  def train(simul):
    # print(f'Train and test iteration {simul} initialized')
    local_state = np.random.RandomState(simul)
    X_train = local_state.uniform(-1, 1, size=(m, dim))
    y_train = eq.evaluate_model(X_train, ed_diff_function)
    # print('End of the evaluation')
    if my_method[0] == 'qcbp':
      my_poly_ref = eq.Poly(my_param_list, my_basis, method='custom-solver',
            sampling_args={'mesh':'user-defined', 'sample-points':X_train, 'sample-outputs':y_train})
      A = my_poly_ref.get_poly(X_train).T
      eta_opt = np.linalg.norm(A@c_ref - y_train)
      my_poly = eq.Poly(my_param_list, my_basis, method='custom-solver',
          sampling_args={'mesh':'user-defined', 'sample-points':X_train, 'sample-outputs':y_train},
            solver_args={'solve':my_method[1], 'eta':eta_opt, 'verbose':False})
    elif my_method[0] == 'ls':
      my_poly = eq.Poly(my_param_list, my_basis, method='custom-solver',
            sampling_args={'mesh':'user-defined', 'sample-points':X_train, 'sample-outputs':y_train},
              solver_args={'solve':my_method[1], 'verbose':False})

    my_poly.set_model()
    test_pts = local_state.uniform(-1, 1, size=(100, dim))
    test_evals = eq.evaluate_model(test_pts, ed_diff_function)
    train_r2, test_r2 = my_poly.get_polyscore(X_test=test_pts, y_test=test_evals, metric='rmse')
    print(train_r2, test_r2)
    # print(f'Train and test iteration {simul} completed')
    return test_r2
  with Pool() as pool:
    print(f"{pool.__dict__['_processes']} active threads")
    result = pool.map(train, range(simuls))
  for test_r2 in result:
    errors = np.append(errors, test_r2)
  return np.mean(errors), np.std(errors)


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
    test_pts = np.random.uniform(-1, 1, size=(100, dim))
    test_evals = eq.evaluate_model(test_pts, ted_diff_function)
    train_r2, test_r2 = my_poly.get_polyscore(X_test=test_pts, y_test=test_evals, metric='rmse')
    train_r2, test_r2

    errors = np.append(errors, test_r2)

  return np.mean(errors), np.std(errors)

def conv(x, method, dim=3, simuls=5, basis='total-order', ord=4, verbose=False):
  means = []
  stds = []

  for i, element in enumerate(x):
    print(f'{element} points for training\n')
    if verbose:
      start = time.time()
    m,s = get_average_rmse(element, method, dim=dim, simuls=simuls, ord=ord, basis=basis)
    means.append(m)
    stds.append(s)
    if verbose:
      end = time.time()
      print('m={} w/ {}, done: {} seconds.'.format(element, method, end-start))

  return np.array(means), np.array(stds)

def conv_varcoefs(x, method, dim=3, simuls=5, basis='total-order', ord=4, verbose=False):
  means = []
  stds = []

  for element in x:
    if verbose:
      start = time.time()
    m, s = get_average_rmse_varcoefs(element, method, dim=dim, simuls=simuls, ord=ord, basis=basis)[0]
    means.append(m)
    stds.append(s)
    
    if verbose:
      end = time.time()
      print('m={} w/ {}, done: {} seconds.'.format(element, method, end-start))

  return means, stds