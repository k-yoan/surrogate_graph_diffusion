# Imports
import equadratures as eq
import numpy as np
from tqdm import tqdm
from math import sqrt

from graph import DGraph
from src_sbm import solvers


class PolyApp():
    '''
    Docstring pour PolyApp

    Parameters
    ----------
    graph : graph.DGraph
        The graph to be used for polynomial approximation. Must be a graph.Graph instance.
    alpha : float, optional
        Scaling factor for the algorithm. Defaults to 0.05.
    mapping : dict, optional
        A dictionary mapping nodes to specific values.

    Attributes
    ----------
    attribute_name : type
        A description of the attribute's purpose and default state.
    metadata : dict
        A collection of key-value pairs for internal tracking.

    Methods
    -------
    method_name(arg1, arg2)
        Briefly describes what the method achieves.

    Notes
    -----
    Include any 'gotchas', performance warnings, or specific 
    NetworkX/NumPy version requirements here.

    Examples
    --------
    >>> processor = MyClassName(graph_object)
    >>> processor.run_analysis()
    '''
    # Class variables
    function_map = {
        'qcbp': solvers.qcbp,
        'weighted_qcbp': solvers.weighted_qcbp,
        'ls': solvers.ls
    }

    # Initialization
    def __init__(self, dgraph):
        self.dgraph = dgraph
        # Check if it is a DGraph object, and not a NetworkX object
    

    def callable(self, y, diffusion_type='ED', chosen_node=2):
        """
        Short description of what the method does.

        Parameters
        ----------
        m : int or list
            Number of sample points, or list containing a sequence of number of sample points.
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
        y = (y+1)/2 # transformation on y to take it from [-1,1] to [0,1]
        d = len(y)
        K = int(sqrt(2*d + 0.25) - 0.5)
        self.dgraph.diff_coefficients = np.zeros((K,K))
        # Construct c_bar (diffusion coefficients) matrix by filling the upper triangular entries from left to right, top to bottom
        idx = 0
        for i in range(K):
            for j in range(i,K):
                self.dgraph.diff_coefficients[i,j] = y[idx]
                idx += 1
        diagonal_entries = [self.dgraph.diff_coefficients[i,i] for i in range(K)]
        self.dgraph.diff_coefficients = self.dgraph.diff_coefficients + self.dgraph.diff_coefficients.T
        self.dgraph.diff_coefficients = self.dgraph.diff_coefficients - np.diag(diagonal_entries)
        # Perform diffusion
        u_0 = np.random.rand(self.dgraph._nb_nodes)*100  # random intial conditions
        if diffusion_type not in ['regular', 'ED', 'TED']:
            raise ValueError('The diffusion type is not valid. Choose between ["regular", "ED", "TED"].')
        elif diffusion_type == 'regular':
            diffusion_data = self.dgraph.diffusion(u_0, only_final=True)
        elif diffusion_type == 'ED':
            diffusion_data = self.dgraph.ED_diffusion(u_0, only_final=True)
        else:
            diffusion_data = self.dgraph.TED_diffusion(u_0)
        # Return chosen value (u_i(T) for chosen i)
        node = chosen_node
        if node not in self.dgraph.graph.nodes:
            raise ValueError('The chosen node is not valid.')

        return diffusion_data['data'][node, -1]


    def get_c_ref(self, basis, method, order, dimension, diffusion_type='ED'):
        """
        Short description of what the method does.

        Parameters
        ----------
        m : int or list
            Number of sample points, or list containing a sequence of number of sample points.
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
        if method not in ['qcbp', 'weighted_qcbp']:
            return None
        else:
            eq_parameters = [eq.Parameter(distribution='uniform', order=order, lower=-1.0, upper=1.0) for _ in range(dimension)]
            ref_basis = eq.Basis(basis, orders=[order for _ in range(dimension)])
            poly = eq.Poly(eq_parameters, ref_basis)
            index_set_size = ref_basis.get_cardinality()
            M = 10 * index_set_size
            err_grid = np.random.uniform(-1, 1, size=(M, dimension))
            A_err_grid = poly.get_poly(err_grid).T/sqrt(M)
            callable = lambda x: self.callable(x, diffusion_type=diffusion_type)
            b_err_grid = eq.evaluate_model(err_grid, callable)/sqrt(M)
            c_ref, _, _, _ = np.linalg.lstsq(A_err_grid, b_err_grid)
        
        return c_ref


    def get_rmse(self, m, basis, method, order, dimension, diffusion_type='ED', n_iterations=5):
        """
        Short description of what the method does.

        Parameters
        ----------
        m : int or list
            Number of sample points, or list containing a sequence of number of sample points.
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
        if method not in self.function_map.keys():
            raise ValueError(f'Options for method: {list(self.function_map.keys())}')
        
        errors = []
        eq_parameters = [eq.Parameter(distribution='uniform', order=order, lower=-1.0, upper=1.0) for _ in range(dimension)]
        eq_basis = eq.Basis(basis, orders=[order for _ in range(dimension)])
        callable = lambda x: self.callable(x, diffusion_type=diffusion_type)

        with tqdm(desc='Generating training data...', bar_format='{desc}: {elapsed}s') as pbar:
            X_train = np.random.uniform(-1, 1, size=(m, dimension))
            y_train = eq.evaluate_model(X_train, callable)
            pbar.set_description('Training data generation completed.')
        
        if method not in ['qcbp', 'weighted_qcbp']:
            eta_opt = None
        else:
            with tqdm(desc='Computing oracle parameter...', bar_format='{desc}: {elapsed}s') as pbar:
                ref_poly = eq.Poly(eq_parameters, eq_basis, method='least-squares', sampling_args={'mesh':'user-defined', 'sample-points':X_train, 'sample-outputs':y_train})
                S = ref_poly.get_poly(X_train).T
                c_ref = self.get_c_ref(basis=basis, method=method, order=order, dimension=dimension, diffusion_type=diffusion_type)
                eta_opt = np.linalg.norm(S@c_ref - y_train)
                pbar.set_description('Oracle parameter computation completed.')
        
        solver = self.function_map[method]

        # weights computation
        I = eq_basis.get_elements().T
        size_I = I.shape
        weights = np.prod(np.sqrt(2 * I + np.ones(size_I)), axis=0)
        use_weights = {
            'qcbp': None,
            'weighted_qcbp': weights,
            'ls': None
        }
        
        for i in tqdm(range(n_iterations), desc='Main RMSE computation loop.', colour='yellow'):
            with tqdm(desc='Computing coefficients...', bar_format='{desc}: {elapsed}s', colour='blue') as pbar:
                poly = eq.Poly(eq_parameters, eq_basis, method='custom-solver', 
                            sampling_args={'mesh':'user-defined', 'sample-points':X_train, 'sample-outputs':y_train},
                            solver_args={'solve':solver, 'eta':eta_opt, 'w':use_weights[method], 'verbose':False})
                poly.set_model()
                pbar.set_description('Coefficients computation completed.')

            with tqdm(desc='Computing RMSE...', bar_format='{desc}: {elapsed}s', colour='blue') as pbar:
                test_pts = np.random.uniform(-1, 1, size=(1000, dimension))
                test_evals = eq.evaluate_model(test_pts, callable)
                train_r2, test_r2 = poly.get_polyscore(X_test=test_pts, y_test=test_evals, metric='rmse')
                train_r2, test_r2
                pbar.set_description('RMSE computation completed.')

            errors.append(test_r2)
        
        return np.array(errors)
        
        





