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


def get_average_rmse(m, conf_vars, dim=3, dataset='twitter'):

  n, A, sizes, x0 = conf_vars
  def train(i):
    X_train = np.random.uniform(-1, 1, size=(int(m/10), dim))
    y_train = eq.evaluate_model(X_train, ed_diff)
    np.save(f'{dataset}/graph_diff_{m}_samples_{dataset}_{i}', X_train)
    np.save(f'{dataset}/graph_diff_{m}_evaluations_{dataset}_{i}', y_train)


  ed_diff = lambda x: ed_diff_function(x, n, A, sizes, x0)
  with Pool(10) as pool:
    print(f"{pool.__dict__['_processes']} active threads")
    pool.map(train, range(10))
   
  
  return 

def conv(x, conf_vars, dim=3, dataset='twitter'):
  for element in x:
    print(f'{element} points for training\n')
    get_average_rmse(element, conf_vars, dim=dim, dataset=dataset)
  return 
