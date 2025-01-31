
###########################################################

import numpy as np
from math import exp, sqrt
import networkx as nx
from scipy.integrate import solve_ivp



### "Regular" diffusion

def diffusion(L, x0, h=0.1, T=10, C=1):

  '''
  ARGUMENTS:
    L : 2D Numpy array. Graph Laplacian.
    x0 : 1D Numpy array. Represents the initial conditions of the nodes on the graph, i.e. x(t) at t=0
    h : Float (0.1 by default). Time step
    T : Integer (10 by default). Maximum time
    C : Float (1 by default). Diffusivity constant?


  OUTPUT:
    data : 2D Numpy array
        Matrix of states of the nodes as time progresses.
  '''

  lmb, v = np.linalg.eig(L)
  data = np.array(x0)

  a_0 = np.linalg.inv(v)@x0

  time = [i*h for i in range(1, int(T/h))] # time steps

  for t in time:
    E = np.array([exp(-C*i*t) for i in lmb]) # vector of eigenvalues exponentials
    a = np.diag(E)@a_0  # vector of 'a' coefficients (see Newman textbook)
    x = np.sum(np.diag(a)@v.T, axis=0) # psi(t)
    data = np.column_stack((data, x))

  return data




### ED diffusion


def ed_diffusion(M, x0, h=0.1, T=10):

  '''
  Edge-Dependent (ED) diffusion.

  ARGUMENTS:
    M : 2D Numpy array. This matrix should be equivalent to a weighted Laplacian constructed with the matrices C and D(C).
    x0 : 1D Numpy array. Represents the initial conditions of the nodes on the graph, i.e. x(t) at t=0
    h : Float (0.1 by default). Time step
    T : Integer (10 by default). Maximum time


  OUTPUT:
    data : 2D Numpy array
        Matrix of quantities of the nodes as time progresses.
  '''

  lmb, v = np.linalg.eig(M)
  data = np.array(x0)

  '''
  a_0 = np.linalg.inv(v)@x0

  #time = [i*h for i in range(1, int(T/h))] # time steps
  t = int(T/h)

  #for t in time:
  E = np.array([exp(i*t) for i in lmb]) # vector of eigenvalues exponentials
  a = np.diag(E)@a_0  # vector of 'a' coefficients
  x = np.sum(np.diag(a)@v.T, axis=0) # psi(t)
  data = np.column_stack((data, x))
  '''
  

  
  a_0 = np.linalg.inv(v)@x0
  a_0 = a_0.reshape(-1,1)

  #time = [i*h for i in range(1, int(T/h))] # time steps
  t = int(T/h)

  #for t in time:
  E = np.array([exp(i*t) for i in lmb]) # vector of eigenvalues exponentials
  a = np.diag(E)@a_0  # vector of 'a' coefficients
  a = a.A
  #print(type(a))
  #print(a.shape)
  a = a.squeeze()
  #print(a.shape)
  #print(v.T.shape)
  x = np.sum(np.diag(a)@v.T, axis=0)
  x = x.A # psi(t)
  x = x.reshape(-1)
  #print(data.shape)
  #print(x.shape)
  #print(type(x))
  data = np.column_stack((data, x))
  

  return data



def construct_cbar(y):
  '''
  This function takes a d-dimensional diffusion parameter y, and returns a KxK symmetric matrix where
  the entry (i,j) denotes the diffusion coefficient between communities i & j, for all K communities.

  ARGUMENTS:
    y : List or 1D Numpy array. Diffusion parameter. 


  OUTPUT:
    c_bar : 2D Numpy array
        KxK matrix of diffusion coefficients between and within the K communities
  '''

  d = len(y) # dimension of y
  K = int(sqrt(2*d + 0.25) - 0.5) # See report for relationship between dimension and nb of communities
  c_bar = np.zeros((K,K))

  # In this construction, we assume that the diffusion coefficients in the parameter y are ordered a certain way:
  # first, we list the diffusion coefficients for the first community, then the second, then the third, etc.
  # In other words, the matrix c_bar is constructed in such a way that when its upper triangular part is flattened, it gives us y.

  idx = 0
  for i in range(K):
    for j in range(i,K):
      c_bar[i,j] = y[idx]
      idx += 1

  # Makes the upper triangular matrix a symmetric matrix by setting equal the entries (j,i) to their upper triangular equivalent (i,j)
  # This is done by adding the transpose and subtracting the original diagonal entries (since they have been added to themselves)

  diag = np.diag(c_bar)
  c_bar = c_bar + c_bar.T
  c_bar = c_bar - np.diag(diag)

  return c_bar



def construct_C(c_bar, sizes, n):

  '''
  This function creates the block matrix C necessary to construct the "weighted Laplacian"-equivalent, which will then be used for ED diffusion.

  ARGUMENTS:
    c_bar : 2D Numpy array. KxK matrix of diffusion coeffiecients between and within the K communities.
    sizes : List of number of nodes per community.
    n : Integer. Total number of nodes.


  OUTPUT:
    C : 2D Numpy array
        Matrix of diffusion coefficients c_{ij} from node i to node j
  '''

  K = c_bar.shape[0]

  # The block matrix C is constructed the following way:
  # c_bar[i,j]*np.ones((sizes[i],sizes[j])), where c_bar[i,j] denotes the diffusion coefficient between communities i & j
  # That diffusion coefficient is multiplied by a matrix of 1's with the shape (sizes[i], sizes[j]), 
  # sizes[i] & sizes[j] represent the number of nodes in communities i & j respectively
  C = [[c_bar[i,j]*np.ones((sizes[i], sizes[j])) for j in range(K)] for i in range(K)]
  #C = [[c_bar[i,j]*np.ones((sizes[i],sizes[j])) for j in range(c_bar.shape[1])] for i in range(c_bar.shape[0])]
  C = np.block(C)

  return C


def construct_D(A, C, n):

  '''
  This function creates the matrix D(C), analogous to a weighted degree matrix.
  That matrix is necessary to construct the "weighted Laplacian"-equivalent, which will then be used for ED diffusion.

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





### TED diffusion


def ted_diffusion(x0, h, Mt, A, sizes, max_iter=100):

  '''
  Time-and-Edge-Dependent (TED) diffusion.

  This function implements Runge-Kutta 4 to find a numerical solution to the TED diffusion equation.

  ARGUMENTS:
    x0 : 1D Numpy array. Represents the initial conditions of the nodes on the graph, i.e. x(t) at t=0
    h : Float. Time step.
    Mt : lambda function to create the matrix M(t) as a 2D Numpy array
    max_iter : Integer (100 by default). Stopping criteria, might use a different one

  OUTPUT:
    data : 2D Numpy array
        Matrix of quantities of the nodes as time progresses.
  '''

  x = x0
  data = np.array(x)

  time = np.array([i*h for i in range(2*max_iter)])

  #n = x0.shape[0]
  # M is a list of 2D Numpy arrays representing M(t) at each time step.
  # We achieve this by calling the lambda function Mt, returned by the function construct_Mt.
  M = [Mt(time[i]/2, A, sizes) for i in range(2*max_iter-1)]

  # RK4 implementation
  for i in range(1, 2*(max_iter-1), 2):
    k_1 = M[i-1]@x
    k_2 = M[i]@(x+k_1*h/2)
    k_3 = M[i]@(x+k_2*h/2)
    k_4 = M[i+1]@(x+k_3*h)
    
    x += h/6 * (k_1 + 2*k_2 + 2*k_3 + k_4) 
    data = np.column_stack((data, x))

  return data


def make_symmetric(matrix):
  '''
  This function takes an upper (or lower) triangular matrix and makes it symmetric, i.e. takes the reflection about the diagonal entries.
  '''
  return (matrix + matrix.T)/2


def sigmoid(x, a=1, b=0):
  '''
  Sigmoid function used for time dependency in TED diffusion.
  '''
  return 1/(1+exp(-a*(x-b)))


def gaussian(x, a=1, b=0):
  '''
  Gaussian function used for time dependency in TED diffusion.
  '''
  return exp(-a*(x-b)**2)



def construct_Mt(A, C, sizes, sig=True, alpha=None):

  '''
  This function returns a lambda function that can be called and will output the TED diffusion operator M(t) for any t.

  ARGUMENTS:
    A : 2D Numpy array. Adjacency matrix.
    C : 2D Numpy array. Matrix of diffusion coefficients c_{ij} from node i to node j.
    sizes : List. Number of nodes per community.
    sig : Boolean (True by default). Time dependency function is a sigmoid function, gaussian otherwise.
    alpha : 1D Numpy array (None by default). Parameters for the time dependency functions (see argument 'a' in the 'sigmoid' and 'gaussian' functions).


  OUTPUT:
    M(t) : Lambda function
  '''

  K = len(sizes) # number of communities
  n = sum(sizes) # number of nodes
  d = int(K*(K+1)/2) # dimension of diffusion parameter y

  # The function used for the time dependency can be a sigmoid function or a Gaussian function.
  if sig:
    func = sigmoid
  else:
    func = gaussian

  # The argument alpha determines the shape of the time dependency function (sigmoid or Gaussian).
  # By default, we set it randomly (such that alpha is in the interval [1,9] for each dimension of y),
  # but the values can be changed simply by giving a Numpy array when calling the construct_Mt function.
  if alpha==None:
    alpha = [np.random.randint(1, high=9) for i in range(d)]

  # We return a callable function M(t, *args) such that M(t) = C(t)*A - D(C(t))
  # C(t) is constructed by doing the entrywise product of the regular matrix C (see ED diffusion case) and 
  # a similarly constructed matrix with the time dependencies h_{i,j}(t) instead of the diffusion coefficients c_{i,j}.
  # This gives us the entries for C_{i,j}(t) = c_{i,j} * h_{i,j}(t).
  # This formulation allows us to obtain D(C(t)) in a similar way.
    
  return lambda t, A, sizes: (C*construct_C(construct_cbar([func(t, a=alpha[i]) for i in range(d)]), sizes, n))*A - construct_D(A, C*construct_C(construct_cbar([func(t, a=alpha[i]) for i in range(d)]), sizes, n), n)






