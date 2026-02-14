# Imports
import networkx as nx
import numpy as np
from math import exp
from functools import cached_property


class DGraph():
    '''
    Graph object for handling different types of diffusion processes.

    Parameters
    ----------
    graph : nx.Graph
        The graph to be processed. Must be a valid NetworkX graph object.
    partition : list, optional
        List of lists, where each sublist represents nodes within the same community.
    diff_coefficients : list, optional
        A KxK symmetric matrix, where the entry (i,j) represents the diffusion coefficient between community i and community j.

    Methods
    -------
    diffusion(u_0, h, T, c, only_final)
        Performs "regular" diffusion on graph, given some initial state.
    
    ED_diffusion(u_0, h, T, only_final)
        Performs edge-dependent diffusion on graph, given some initial state.
    
    TED_diffusion(u_0, h, max_iter, time_dependency, **kwargs)
        Performs time-and-edge-dependent diffusion on graph, given some initial state, by implementing Runge-Kutta 4.

    Examples
    --------
    >>> G = nx.complete_graph(10)
    >>> graph = Graph(G)
    >>> initial_state = np.random.randint()
    >>> diffusion_data = graph.diffusion(initial_state)

    >>> # ED-diffusion on a stochastic block model with 10 nodes and 2 communities
    >>> partition = [[0, 1, 2, 3], [4, 5, 6, 7, 8, 9]]
    >>> sbm = nx.stochastic_block_model(sizes=[len(community) for community in partition], p=[[1, 0.2], [0.2, 1]])
    >>> graph = Graph(sbm, partition=partition)
    >>> initial_state = np.random.randint()
    >>> ed_diffusion_data = graph.ed_diffusion(initial_state)
    '''

    def __init__(self, graph, partition=None, diff_coefficients=None):
        self.graph = graph
        # Check if we have a NetworkX graph object
        if not isinstance(self.graph, nx.Graph):
            raise TypeError('The provided graph has to be a NetworkX graph object.')
        # If a partition is not specified, create one by splitting the nodes in two groups
        if partition is None:
            node_set = list(self.graph.nodes)
            self.partition = [node_set[:len(node_set)//2], node_set[len(node_set)//2:]]
        self.diff_coefficients = diff_coefficients
        # Initialize important values and matrices
        self._A = nx.adjacency_matrix(self.graph).toarray() # Adjacency matrix
        self._L = nx.laplacian_matrix(self.graph).toarray() # Laplacian matrix
        self._D = self._L + self._A # Degree matrix
        self._nb_nodes = len(self.graph.nodes)
        self._nb_edges = len(self.graph.edges)
        self._nb_communities = len(self.partition)
    

    @cached_property
    def _C(self):
        if self.diff_coefficients is None:
            raise ValueError('The diffusion coefficients were not provided.')
        if not np.allclose(np.array(self.diff_coefficients), np.array(self.diff_coefficients).T) or len(self.partition) != np.array(self.diff_coefficients).shape[0]:
            raise ValueError('The diffusion coefficients must be passed as a symmetric matrix with shape (K, K), where K is the number of communities.')
        K = np.array(self.diff_coefficients).shape[0]
        sizes = [len(community) for community in self.partition] # list of number of nodes per community
        C = [[list(self.diff_coefficients)[i][j]*np.ones((sizes[i],sizes[j])) for j in range(K)] for i in range(K)]

        return np.block(C)
    

    @cached_property
    def _DC(self):
        if self.diff_coefficients is None:
            raise ValueError('The diffusion coefficients were not provided.')
        if not np.allclose(np.array(self.diff_coefficients), np.array(self.diff_coefficients).T) or len(self.partition) != np.array(self.diff_coefficients).shape[0]:
            raise ValueError('The diffusion coefficients must be passed as a symmetric matrix with shape (K, K), where K is the number of communities.')
        d = [np.dot(self._C[i,:], self._A[i,:]) for i in range(self._nb_nodes)]
        return np.diag(d)



    def diffusion(self, u_0, h=0.1, T=10, c=1, only_final=False):
        """
        Short description of what the method does.

        Parameters
        ----------
        u_0 : numpy.ndarray or list
            Initial state of the graph at each node (at t=0).
        h : float, optional
            Step size
        T : float, optional
            Final time
        c : float, optional
            Diffusion constant
        only_final : bool, optional
            If False (by default), computes the diffusion at every time step. Otherwise, only computes u(T).

        Returns
        -------
        dict
            Dictionary containing useful metadata about the diffusion process, including array with the state of the graph at each time step.
        """
        # Check if L is symmetric, in which case, use the faster eigh function. Otherwise, use eig.
        if np.allclose(self._L, self._L.T):
            lmb, v = np.linalg.eigh(self._L)
        else:
            lmb, v = np.linalg.eig(self._L)
        matrix_of_states = [] #np.array(u_0)
        matrix_of_states.append(list(u_0))
        a_0 = np.linalg.inv(v)@np.array(u_0)
        time_steps = [i*h for i in range(1, int(T/h))]

        for t in time_steps:
            if only_final:
                E = np.exp(-c*lmb*T) # vector of eigenvalues exponentials
                a = E * a_0  # vector of 'a' coefficients
                u = np.sum(a[:, np.newaxis] * v.T, axis=0) # psi(t)
                matrix_of_states.append(list(u))
                break
            # M. E. J. Newman. Networks: An Introduction. Oxford; New York: Oxford University Press, 2010, pp. 149-151.
            E = np.exp(-c*lmb*t) # vector of eigenvalues exponentials
            a = E * a_0  # vector of 'a' coefficients
            u = np.sum(a[:, np.newaxis] * v.T, axis=0) # psi(t)
            matrix_of_states.append(list(u))
        
        time_steps.insert(0,0)
        
        metadata = {
            'data': np.array(matrix_of_states).T,
            'title': f'Regular diffusion on a graph with {self._nb_nodes} nodes and {self._nb_communities} communities',
            'time_step': h,
            'final_time': T,
            'diff_constant': c,
            'time_array': time_steps,
            'nodes': list(self.graph.nodes),
            'partition': self.partition,
            'only_final': only_final,
            'graph': self.graph
        }
        
        return metadata
    

    def ED_diffusion(self, u_0, h=0.1, T=10, only_final=False):
        """
        Short description of what the method does.

        Parameters
        ----------
        u_0 : numpy.ndarray or list
            Initial state of the graph at each node (at t=0).
        h : float, optional
            Step size
        T : float, optional
            Final time
        only_final : bool, optional
            If False (by default), computes the diffusion at every time step. Otherwise, only computes u(T).

        Returns
        -------
        dict
            Dictionary containing useful metadata about the diffusion process, including array with the state of the graph at each time step.
        """
        M = self._C * self._A - self._DC
        # Check if M is symmetric, in which case, use the faster eigh function. Otherwise, use eig.
        if np.allclose(M, M.T):
            lmb, v = np.linalg.eigh(M)
        else:
            lmb, v = np.linalg.eig(M)
        matrix_of_states = []
        matrix_of_states.append(list(u_0))
        a_0 = np.linalg.inv(v)@np.array(u_0)
        time_steps = [i*h for i in range(1, int(T/h))]

        for t in time_steps:
            if only_final:
                E = np.exp(lmb*T) # vector of eigenvalues exponentials
                a = E * a_0  # vector of 'a' coefficients
                u = np.sum(a[:, np.newaxis] * v.T, axis=0) # psi(t)
                matrix_of_states.append(list(u))
                break
            E = np.exp(lmb*t) # vector of eigenvalues exponentials
            a = E * a_0  # vector of 'a' coefficients
            u = np.sum(a[:, np.newaxis] * v.T, axis=0) # psi(t)
            matrix_of_states.append(list(u))
        
        time_steps.insert(0,0)
        
        metadata = {
            'data': np.array(matrix_of_states).T,
            'title': f'Edge-Dependent diffusion on a graph with {self._nb_nodes} nodes and {self._nb_communities} communities',
            'time_step': h,
            'final_time': T,
            'time_array': time_steps,
            'nodes': list(self.graph.nodes),
            'partition': self.partition,
            'only_final': only_final,
            'graph': self.graph
        }
        
        return metadata
    

    def TED_diffusion(self, u_0, h=0.001, max_iter=100, time_dependency='sigmoid', **kwargs):
        """
        Short description of what the method does.

        Parameters
        ----------
        u_0 : numpy.ndarray or list
            Initial state of the graph at each node (at t=0).
        h : float, optional
            Step size
        max_iter : int, optional
            Maximum number of iterations of Runge-Kutta 4 (100 by default).
        time_dependency : str, optional
            Function type influencing the time dynamics in the TED diffusion process. Can choose between ['sigmoid', 'gaussian', 'linear'].

        Returns
        -------
        dict
            Dictionary containing useful metadata about the diffusion process, including array with the state of the graph at each time step.
        """
        functions = {
            'sigmoid': lambda t,a=1,b=0: 1/(1+exp(-a*(t-b))),
            'gaussian': lambda t,a=1,b=0: exp(-a*(t-b)**2),
            'linear': lambda t,a=1: a*t
        }
        if time_dependency not in functions.keys():
            raise ValueError('The time dependency is not valid. Choose between ["sigmoid", "gaussian", "linear"].')
        f = functions[time_dependency]

        matrix_of_states = []
        matrix_of_states.append(list(u_0))
        u = np.array(u_0)
        k = 0
        time_steps = [k+i*h for i in range(max_iter+1)]

        # Hadamard product of C and A outside loop, for efficiency
        product = self._C*self._A

        while k < max_iter:
            # Different time iterations of M: M(t_k), M(t_k + h/2) and M(t_k + h)
            M1 = f(time_steps[k])*product - f(time_steps[k])*self._DC
            M2 = f(time_steps[k] + h/2)*product - f(time_steps[k] + h/2)*self._DC
            M3 = f(time_steps[k] + h)*product - f(time_steps[k] + h)*self._DC
            # Iterations
            k_1 = M1@u
            k_2 = M2@(u + k_1*h/2)
            k_3 = M2@(u + k_2*h/2)
            k_4 = M3@(u + k_3*h)
            u += h/6 * (k_1 + 2*k_2 + 2*k_3 + k_4)
            matrix_of_states.append(list(u))

            k += 1
        

        metadata = {
            'data': np.array(matrix_of_states).T,
            'title': f'Time-and-Edge-Dependent diffusion on a graph with {self._nb_nodes} nodes and {self._nb_communities} communities',
            'time_step': h,
            'time_array': time_steps,
            'time_dependency': time_dependency,
            'nodes': list(self.graph.nodes),
            'partition': self.partition,
            'graph': self.graph
        }

        return metadata



