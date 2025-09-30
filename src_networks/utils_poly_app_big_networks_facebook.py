import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from math import sqrt
import time
from multiprocess import Pool
import matplotlib.lines as mlines
import cvxpy as cp
from Diff import *
import equadratures as eq
import pdb

#no-time&edge dependent
def ed_diff_function(y,n, A, sizes, x0):
  y = (y+1)/2
  c_bar = construct_cbar(y)
  C = construct_C(c_bar, sizes, n)
  D = construct_D(A, C, n)
  M = C*A - D

  diff_2 = ed_diffusion(M, x0, tau=0.01, T=1)

  return diff_2[2,-1]

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

def get_average_rmse(m, my_method, conf_vars, dim=3, simuls=5, basis='total-order', ord=4, dataset='twitter'):
  errors = np.array([])
  my_param_list = [eq.Parameter(distribution='uniform', order=ord, lower=-1.0, upper=1.0) for i in range(dim)]

  #if basis == 'hyperbolic-cross':
  my_basis = eq.Basis(basis, orders=[ord for _ in range(dim)])
  #else:
  #  my_basis = eq.Basis(basis)
  n, A, sizes, x0 = conf_vars
  ed_diff = lambda x: ed_diff_function(x,n,A,sizes,x0)
  
  points = np.load(f'{dataset}/graph_diff_12650_samples_{dataset}.npy')
  evals =  np.load(f'{dataset}/graph_diff_12650_evaluations_{dataset}.npy' )

  if my_method[0] == 'qcbp':
    poly = eq.Poly([eq.Parameter(distribution='uniform', order=ord, lower=-1.0, upper=1.0) for _ in range(dim)], my_basis)
    index_set_size = my_basis.get_cardinality()
    M = 10 * index_set_size
    err_grid = points[-M:,:]#np.random.uniform(-1, 1, size=(M, dim))
    A_err_grid = poly.get_poly(err_grid).T/sqrt(M)
    b_err_grid = evals[-M:,:]/sqrt(M)#eq.evaluate_model(err_grid, ed_diff)/sqrt(M)
    c_ref, _, _, _ = np.linalg.lstsq(A_err_grid, b_err_grid)
  else:
    M=0
  #generating test samples (shared across all the seeds)  
  test_pts = points[:1000,:]#np.load(f'{dataset}/graph_diff_1000_samples_{dataset}.npy')#np.random.uniform(-1, 1, size=(1000, dim))
  test_evals = evals[:1000,:] #np.load(f'{dataset}/graph_diff_1000_evaluations_{dataset}.npy' )#eq.evaluate_model(test_pts, ed_diff)

  points_list = [100,300,500,700]
  train_indices_start = 1000+int(np.sum(points_list[:int((m-100)/200)])*simuls)
  print(train_indices_start)
  for j in range(simuls):
    start_time = time.time()

    
    X_train = points[train_indices_start+j*m:train_indices_start+(j+1)*m]#np.load(f'{dataset}/graph_diff_{m}_samples_{dataset}_{j}.npy')#np.random.uniform(-1, 1, size=(m, dim))
    y_train = evals[train_indices_start+j*m:train_indices_start+(j+1)*m]#np.load(f'{dataset}/graph_diff_{m}_evaluations_{dataset}_{j}.npy')#eq.evaluate_model(X_train, ed_diff)

    if my_method[0] == 'qcbp':
      my_poly_ref = eq.Poly(my_param_list, my_basis, method='least-squares',
            sampling_args={'mesh':'user-defined', 'sample-points':X_train, 'sample-outputs':y_train})
      S = my_poly_ref.get_poly(X_train).T
      eta_opt = np.linalg.norm(S@c_ref - y_train)
      my_poly = eq.Poly(my_param_list, my_basis, method='custom-solver',
          sampling_args={'mesh':'user-defined', 'sample-points':X_train, 'sample-outputs':y_train},
            solver_args={'solve':my_method[1], 'eta':eta_opt, 'verbose':False})#eta_opt
    elif my_method[0] == 'ls':
      my_poly = eq.Poly(my_param_list, my_basis, method='custom-solver',
            sampling_args={'mesh':'user-defined', 'sample-points':X_train, 'sample-outputs':y_train},
              solver_args={'solve':my_method[1], 'verbose':False})
    
    my_poly.set_model()

    print('Training ended')
    #print(my_poly.coefficients)
    print('Starting test phase..')
    
    train_r2, test_r2 = my_poly.get_polyscore(X_test=test_pts, y_test=test_evals, metric='rmse')
    print(train_r2, test_r2)
    print(f'Elapsed time:{time.time()-start_time} seconds')
    errors = np.append(errors, test_r2)

  return errors

def conv(x, method, conf_vars, dim=3, simuls=5, basis='total-order', ord=4, dataset='twitter', verbose=False):
  Y = []
  for i, element in enumerate(x):
    print(f'{element} points for training\n')

    Y.append(get_average_rmse(element, method, conf_vars, dim=dim, simuls=simuls, ord=ord, basis=basis, dataset=dataset))

  return np.array(Y)
