
###########################################################

import numpy as np
from math import exp, sqrt
import networkx as nx
from scipy.integrate import solve_ivp




def diffusion(L, x0, tau=0.1, T=10, C=1):

  '''
  ARGUMENTS:
    L : 2D Numpy array. Graph Laplacian.
    x0 : 1D Numpy array. Represents the initial conditions of the nodes on the graph, i.e. x(t) at t=0
    tau : Float (0.1 by default). Time step
    T : Integer (10 by default). Maximum time
    C : Float (1 by default). Diffusivity constant?


  OUTPUT:
    data : 2D Numpy array
        Matrix of states of the nodes as time progresses.
  '''

  lmb, v = np.linalg.eig(L)
  data = np.array(x0)

  a_0 = np.linalg.inv(v)@x0

  time = [i*tau for i in range(1, int(T/tau))]

  for t in time:
    E = np.array([exp(-C*i*t) for i in lmb]) # vector of eigenvalues exponentials
    a = np.diag(E)@a_0  # vector of 'a' coefficients
    x = np.sum(np.diag(a)@v.T, axis=0) # psi(t)
    data = np.column_stack((data, x))

  return data


def ed_diffusion(M, x0, tau=0.1, T=10):

  '''
  ARGUMENTS:
    M : 2D Numpy array.
    x0 : 1D Numpy array. Represents the initial conditions of the nodes on the graph, i.e. x(t) at t=0
    tau : Float (0.1 by default). Time step
    T : Integer (10 by default). Maximum time


  OUTPUT:
    data : 2D Numpy array
        Matrix of states of the nodes as time progresses.
  '''

  lmb, v = np.linalg.eig(M)
  data = np.array(x0)

  a_0 = np.linalg.inv(v)@x0

  time = [i*tau for i in range(1, int(T/tau))]

  #TODO: simplify and compute only on the last timestep
  for t in time:
    E = np.array([exp(i*t) for i in lmb]) # vector of eigenvalues exponentials
    a = np.diag(E)@a_0  # vector of 'a' coefficients
    x = np.sum(np.diag(a)@v.T, axis=0) # psi(t)
    data = np.column_stack((data, x))

  return data


def construct_C(c_bar, sizes, n):

  '''
  ARGUMENTS:
    c_bar : 2D Numpy array. KxK matrix of diffusion coeffiecients between and within the K communities.
    sizes : List. Number of nodes per community.
    n : Integer. Total number of nodes.


  OUTPUT:
    C : 2D Numpy array
        Matrix of diffusion coefficients c_{ij} from node i to node j
  '''
  K = c_bar.shape[0]
  C = [[c_bar[i,j]*np.ones((sizes[i],sizes[j])) for j in range(K)] for i in range(K)]
  C = np.block(C)

  return C


def construct_D(A, C, n):

  '''
  ARGUMENTS:
    A : 2D Numpy array. Adjacency matrix.
    C : 2D Numpy array. Matrix of diffusion coefficients c_{ij} from node i to node j.
    n : Integer. Total number of nodes.


  OUTPUT:
    D : 2D Numpy array
        Diagonal matrix D(C)
  '''

  d = [sum([C[i,k]*A[i,k] for k in range(n)]) for i in range(n)]
  D = np.diag(d)

  return D


def RK4_t(x0, tau, Mt, max_iter=100):

  '''
  ARGUMENTS:
    x0 : 1D Numpy array. Represents the initial conditions of the nodes on the graph, i.e. x(t) at t=0
    tau : Float (real number)
    Mt : lambda function to create the matrix M(t) as a 2D Numpy array
    max_iter : Integer (100 by default). Stopping criteria, might use a different one

  OUTPUT:
    data : 2D Numpy array
        Matrix of states of the nodes as time progresses.
  '''

  x = x0
  data = np.array(x)

  time = np.array([i*tau for i in range(2*max_iter)])

  n = x0.shape[0]
  M = [Mt(time[i]/2, A, sizes) for i in range(2*(max_iter-1))]

  for i in range(1, 2*(max_iter-1), 2):
    k_1 = M[i-1]@x
    k_2 = M[i]@(x+k_1*tau/2)
    k_3 = M[i]@(x+k_2*tau/2)
    k_4 = M[i+1]@(x+k_3*tau)
    
    x += tau/6 * (k_1 + 2*k_2 + 2*k_3 + k_4) 
    data = np.column_stack((data, x))

  return data


def make_symmetric(matrix):
  return (matrix + matrix.T)/2


def sigmoid(x, a=1, b=0):
  return 1/(1+exp(-a*(x-b)))


def gaussian(x, a=1, b=0):
  return exp(-a*(x-b)**2)


def construct_cbar(y):

  '''
  ARGUMENTS:
    y : List or 1D Numpy array. Diffusion parameter. 


  OUTPUT:
    c_bar : 2D Numpy array
        KxK matrix of diffusion coefficients between and within the K communities
  '''

  d = len(y)
  K = int(sqrt(2*d + 0.25) - 0.5)
  c_bar = np.zeros((K,K))

  idx = 0
  for i in range(K):
    for j in range(i,K):
      c_bar[i,j] = y[idx]
      idx += 1

  diag = np.diag(c_bar)
  c_bar = c_bar + c_bar.T
  c_bar = c_bar - np.diag(diag)

  return c_bar


def construct_Mt(A, C, sizes, sig=True, alpha=None):

  '''
  ARGUMENTS:
    A : 2D Numpy array. Adjacency matrix.
    C : 2D Numpy array. Matrix of diffusion coefficients c_{ij} from node i to node j.
    sizes : List. Number of nodes per community.
    sig : Boolean (True by default). Time dependency function is a sigmoid function, gaussian otherwise.
    alpha : 1D Numpy array (None by default)...


  OUTPUT:
    M(t) : Lambda function
  '''

  K = len(sizes)
  n = sum(sizes)
  d = int(K*(K+1)/2)
  if sig:
    func = sigmoid
  else:
    func = gaussian

  if alpha==None:
    alpha = [np.random.randint(1, high=9) for i in range(d)]
    
  return lambda t: (C*construct_C(construct_cbar([func(t, a=alpha[i]) for i in range(d)]), sizes, n))*A - construct_D(A, C*construct_C(construct_cbar([func(t, a=alpha[i]) for i in range(d)]), sizes, n), n)





