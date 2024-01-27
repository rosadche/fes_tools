#==============================================================================
#                                 IMPORTS
#==============================================================================
import numpy as np
np.seterr(invalid='ignore')

import pandas as pd
import time
import math
"""
==============================================================================
                           Citations
==============================================================================
citation for dijkstra 
Dijkstra, E. W. A Note on Two Problems in Connexion with Graphs.Numer. Math. 1, 269–271

Citations for NEB:
(1) Mandelli, D.; Parrinello, M. arXiv Prepr. arXiv2106.06275 2021.
https://arxiv.org/pdf/2106.06275.pdf
(2) Henkelman, G.; Jónsson, H. J. Chem. Phys. 2000, 113, 9978–9985.
https://aip.scitation.org/doi/pdf/10.1063/1.1323224

Citation for loop finding (will be implemented later):
Chen, H., Ogden, D., Pant, S., Cai, W., Tajkhorshid, E., Moradi, M., Roux, B. and Chipot, C., 2022. 
A companion guide to the string method with swarms of trajectories: 
Characterization, performance, and pitfalls. 
Journal of Chemical Theory and Computation, 18(3), pp.1406-1422.
"""

#==============================================================================
#                           Timing Function
#==============================================================================
def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        print(f'Time taken: {elapsed:.6f} seconds')
        return result
    return wrapper

#==============================================================================
#                           Auxilary Function
#==============================================================================
def get_min_from_indices(mat, tuple_lst):
    """
    Given a matrix and a list of tuples, return the minimum value from the 
    matrix elements indicated by the index
    
    Args:
        mat (np.ndarray): a numpy array representing a matrix
        tuple_list (list of tuples): contains indices of mat
        
    Returns:
        min_index (tuple): the minimum value's index
        min_val (int or float, depends on mat's contents): the minimum value of mat from provided indices
    """
    
    min_val = np.inf
    min_index = (0,0)
    for i in range(len(tuple_lst)):
        index = tuple_lst[i]
        val = mat[index]
        
        if val <= min_val:
            min_val     = val
            min_index   = index
            
    return min_index, min_val


def get_max_from_indices(mat, tuple_lst):
    """
    Given a matrix and a list of tuples, return the maximum value from the 
    matrix elements indicated by the index
    
    Args:
        mat (np.ndarray): a numpy array representing a matrix
        tuple_list (list of tuples): contains indices of mat
        
    Returns:
        max_index (tuple): the maximum value's index
        max_val (int or float, depends on mat's contents): the maximum value of mat from provided indices
    """
    
    max_val = -np.inf
    max_index = (0,0)
    for i in range(len(tuple_lst)):
        index = tuple_lst[i]
        val = mat[index]
        
        if val >= max_val:
            max_val     = val
            max_index   = index
            
    return max_index, max_val


#==============================================================================
#                                FES CLASS
#==============================================================================
class fes:
    """
    A class for representing a Free Energy Surface (FES).
    
    This class is used for both single surfaces and analyzing surfaces 
    as they change over time.
    
    If reactant location is not provided, it will assume the reactant 
    is the lowest energy point of the left side of the first CV.
    If reactant location is not provided, it will assume the prioduct 
    is the lowest energy point of the right side of the first CV.
    
    Will be made generally applicable, currently only implemented for 1 or 2 dimensions 
    
    Args:
        df (pd.DataFrame or list of pd.DataFrame): Contains the data of the free energy surface(s) to be analyzed
        cv_cols (list or array of str): column name(s) of df with collective variables points of the free energy surface
        fe_col (str): column name of df that contains the free energy values
        fe_uncert (str, optional): column name of df that contains the free energy uncertainty
        react_loc (nested iterable [list of list of floats], optional): Points representing the upper and lower corners of the guess region for the reactant (dimension 2, len(cv_cols))
        prod_loc (nested iterable [list of list of floats], optional): Points representing the upper and lower corners of the guess region for the product (dimension 2, len(cv_cols))
        intermediate_locs ([list of list of floats], optional): Points representing the upper and lower corners of the guess region for an intermediate. (dimension 2, len(cv_cols))
        mfep_type (str, optional): May be 'neb' or 'dijkstra'. Only used for FES of dimesnions greater than 1. Default is dijkstra in case of no or incorrect argument. 
        warning (bool, optional):  Whether to print warnings. Default is True.
    
    Attributes:
        results (dict): hold key results as iterables for each free energy surface
            - ts_val (np.ndarray): transition state energies 
            - ts_loc (list of np.ndarray): transition state CV locations
            - minima_vals (np.ndarray): energy values associated with picked local minima
            - minima_locs (list): CV locations for picked minima
            - dG (np.ndarray): reactant to product overall free energy difference
            - fes (np.ndarray): the actual fes matrix for plotting
            - cv_vectors (list of np.ndarray): CV vector array for each time point, as they could change per dataframe
            - mfep (list of np.ndarray): mfep[0] is CV locations of minimum free energy path. mfep[1] is the associated energies.

    Raises:
        ValueError: If FES is more than 2 dimensional
        ValueError: If reactant_loc, product_loc, or intermediate_locs are of incorrect dimensions or type
        TypeError: If df is not type pd.DataFrame or a list of type pd.DataFrame
    
    """
    
    def __init__(self, df, cv_cols, fe_col, fe_uncert=None, reactant_loc=None, product_loc=None, intermediate_locs=None, mfep_type="dijkstra", warning=True):
        
        # make sure the df input is acceptable
        if isinstance(df, list):
            for i in range(len(df)):
                if not isinstance(df[i], pd.DataFrame):
                    raise TypeError('df must be a list of type pd.DataFrame')
            self.num_fes = len(df)
        elif isinstance(df, pd.DataFrame):
            self.num_fes = 1
            df = [df]
        else:
            raise TypeError('df must be type pd.DataFrame or a list of type pd.DataFrame')
        
        for dataframe in df:
            dataframe[fe_col] = dataframe[fe_col] - dataframe[fe_col].min()
        
        if not isinstance(cv_cols, list):
            cv_cols = [cv_cols]
        
        self.df         = df
        self.cv_cols    = cv_cols
        self.dims       = len(self.cv_cols)
        self.fe_col     = fe_col
        self.fe_uncert  = fe_uncert
        self.reactant_loc       = reactant_loc
        self.product_loc        = product_loc
        self.intermediate_locs  = intermediate_locs
        self.mfep_type          = mfep_type.rstrip("\n").lower().replace(" ", "")
        self.warning            = warning
        
        #------------------------------------------------------------------------------
        #                           quality checks
        #------------------------------------------------------------------------------
        # Check to make sure this is a FES we can handle in terms of dimensionality
        if not 0 < self.dims <= 2:
            raise ValueError('fes_tools can only handle 1 to 2 dimensional free energy surfaces at this time')
        
        # make sure the two "upper and lower corners" of the search area are of correct dimension
        # Reactant location (there may only be two points)
        # convert all the regions to np arrays for numba 
        if (self.reactant_loc != None):
            if (len(self.reactant_loc) != (2)):
                raise ValueError('reactant_loc must have outer length of 2 for upper and lower bounds of CV space')
            else:
                for i in range(len(self.reactant_loc)):
                    if len(self.reactant_loc[i]) != self.dims:
                        raise ValueError(f'inner list point {i} is not of correct length: must be len(cv_cols)')
            self.reactant_loc = np.array( self.reactant_loc )
        
        # Product location (there may only be two points)
        # convert all the regions to np arrays for numba 
        if (self.product_loc != None):
            if (len(self.product_loc) != (2)):
                raise ValueError('product_loc must have outer lenght of 2 for upper and lower bounds of CV space')
            else:
                for i in range(len(self.product_loc)):
                    if len(self.product_loc[i]) != self.dims:
                        raise ValueError(f'inner list point {i} is not of correct length: must be len(cv_cols)')
            self.product_loc = np.array( self.product_loc )
                        
        # Intermediate locations (there may be many sets of two points representing intermediate locations)
        # convert all the regions to np arrays for numba 
        if (self.intermediate_locs != None):
            # the outermost layer should group intermedediates thus has no restriction on length
            # the second layer groups upper "upper and lower corners" of the search area
            # the thrid layer holds the points 
            
            for i in range(len(self.intermediate_locs)):
                if self.intermediate_locs[i] != 2:
                    raise ValueError(f'intermediate_locs {i} must have outer length of 2 for upper and lower bounds of CV space')
                else:
                    for j in range(len(self.intermediate_locs)):
                        if self.intermediate_locs[j] != 2 * self.dims:
                            raise ValueError(f'inner list item {j} of intermediate {i} is not of correct length: must be 2 * len(cv_cols)')
                self.intermediate_locs = np.array( self.intermediate_locs )
        
        # Make sure the option for mfep_type is valid
        if self.mfep_type.lower() not in ["neb", "dijkstra"]:
            print(f"{self.mfep_type} is not a valid option for mfep_type: a simple Dijkstra's algorithm will be used instead")
            self.mfep_type = 'dijkstra'
        
        # Initialize arrays for the energy values and lists for the locations
        self.results   = dict()
        self.results["ts_val"] = np.zeros(self.num_fes)
        self.results["ts_loc"] = []
        self.results["minima_vals"]  = []
        self.results["minima_locs"]  = []
        self.results["dG"]     = np.zeros(self.num_fes)
        self.results["fes"]  = []
        self.results["cv_vectors"]  = []
    
        if self.dims > 1:
            self.results["mfep"]  = []
        
        for i in range(self.num_fes):
            self.analyze_fes(i)
    
    
    def analyze_fes(self, index):
        """
        Analyzes a single free energy surface dataframe
        
        Args:
            Index (int): which index of self.df to take
        
        """
        #------------------------------------------------------------------------------
        #       reshape the data frame to get the free energy surface as a matrix
        #------------------------------------------------------------------------------
        if self.dims == 1:
            fes_matrix         = self.df[index][self.fe_col].to_numpy()
            cv_vectors         = [self.df[index][self.cv_cols[0]].to_numpy()]
            if self.fe_uncert != None:
                fes_uncert_matrix  = self.df[index][self.fe_uncert].to_numpy()
                
        if self.dims == 2:
            pivot_table         = self.df[index].pivot_table(index=self.cv_cols[0], columns=self.cv_cols[1:], values=self.fe_col)
            cv_vectors     = [pivot_table.index.values, pivot_table.columns.values]
            fes_matrix     = pivot_table.T.values 
            
            if self.fe_uncert != None:
                pivot_table_uncert      = self.df[index].pivot_table(index=self.cv_cols[0], columns=self.cv_cols[1:], values=self.fe_uncert)
                fes_uncert_matrix  = pivot_table_uncert.T.values
        
        
        #------------------------------------------------------------------------------
        #           get value & index of reactant, intermediates, and product
        #------------------------------------------------------------------------------
        # if a reactant or product location isn't provided, guess the "defaults"
        # if 1d, lets guess the left for reactant, right for product
        # if greater than 1d, we do the same except only based on the first CV
        
        if not isinstance(self.reactant_loc, np.ndarray):
            if self.warning:
                print(f"Reactant location not provided: guessing left side of {self.cv_cols[0]} median value of {np.median(cv_vectors[0]):.2f}")
            self.reactant_loc = []
            self.reactant_loc.append( [cv_vectors[x].min() for x in range(len(cv_vectors))] )
            self.reactant_loc.append( [cv_vectors[x].max() if x != 0 else np.median(cv_vectors[x]) for x in range(len(cv_vectors))] )
            self.reactant_loc = np.asarray( self.reactant_loc )

        if not isinstance(self.product_loc, np.ndarray):
            if self.warning:
                print(f"Product location not provided: guessing right side of {self.cv_cols[0]} median value of {np.median(cv_vectors[0]):.2f}")
            self.product_loc = []
            self.product_loc.append( [cv_vectors[x].min() if x != 0 else np.median(cv_vectors[x]) for x in range(len(cv_vectors))] )
            self.product_loc.append( [cv_vectors[x].max()  for x in range(len(cv_vectors))] )
            self.product_loc = np.array( self.product_loc )
        
        # if intermediates aren't given, skip them after issuing a warning to user
        if (self.intermediate_locs == None):
            if self.warning and index == 0:
                print("Intermediate locations not provided: skipping since an intermediate may not be present")
        
        # Since location wass provided or now has been guessed, get the required information
        reactant_index, reactant_value    = self.get_extrema_from_region(fes_matrix, cv_vectors, region=self.reactant_loc, extrema="min")
        product_index, product_value      = self.get_extrema_from_region(fes_matrix,cv_vectors, region=self.product_loc, extrema="min")
        
        reactant_cv    = self.get_cv_loc_from_index(cv_vectors, reactant_index)
        product_cv     = self.get_cv_loc_from_index(cv_vectors, product_index)
        dG             = product_value - reactant_value

        if (self.intermediate_locs != None):
            intermediate_indices, intermediate_values  = get_intermediates(fes_matrix, cv_vectors, self.intermediate_locs)
            states_indices = [reactant_index, *intermediate_indices, product_index]
        else:
            states_indices = [reactant_index, product_index]
       
        states_cv      = [self.get_cv_loc_from_index(cv_vectors,x) for x in states_indices]
        states_vals    = [fes_matrix[states_indices[i]] for i in range(len(states_indices)) ] 
        
        #------------------------------------------------------------------------------
        #                   calculate the minimum free energy path
        #------------------------------------------------------------------------------
        mfep_cv     = []
        mfep_g      = []
        
        for step in range(len(states_indices)-1):
            if self.dims == 1:
                ts_index, ts_val = self.get_extrema_from_region(fes_matrix, cv_vectors, region=np.asarray([states_cv[step], states_cv[step+1]]), extrema="max")
                ts_cv = self.get_cv_loc_from_index(cv_vectors, ts_index)
                mfep_cv.append([states_cv[step], ts_cv, states_cv[step+1]])
                mfep_g.append([states_vals[step], ts_val, states_vals[step+1]])
                
            else:
                # want nan to be infinite for purpose of dijkstra
                mat_copy = np.nan_to_num(fes_matrix, nan=np.inf, copy=True)

                if self.dims == 2:
                    dist_mat, path_dict = dijkstra_2d(mat_copy, states_indices[step])
                
                mfep_indices = shortest_path(states_indices[step], states_indices[step+1], path_dict)

                if self.mfep_type == "neb":
                    # I want around 20 images if the fes shape is 100 > greater
                    # otherwise I will take 10 %  of the max shape
                    min_matrix_shape = min(fes_matrix.shape)
                    if min_matrix_shape >= 100:
                        desired_images = 20
                    else:
                        desired_images = 0.1 * min_matrix_shape
                    path_spacing = int(math.ceil(len(mfep_indices) / desired_images))
                    to_use = mfep_indices[::path_spacing]
                    to_use[-1] = mfep_indices[-1]
                    mfep_indices = neb_2d(fes_matrix, to_use)
                
                if step == 0:
                    # keep first point (left)
                    mfep_cv.append( [ self.get_cv_loc_from_index(cv_vectors, mfep_indices[i]) for i in range(len(mfep_indices)) ] )
                    mfep_g.append( [ fes_matrix[mfep_indices[i][0], mfep_indices[i][1]] for i in range(len(mfep_indices)) ] )
                else:
                     # remove the left point which will be duplicate
                    mfep_cv.append( [ self.get_cv_loc_from_index(cv_vectors, mfep_indices[i]) for i in range(1, len(mfep_indices)) ] )
                    mfep_g.append( [ fes_matrix[mfep_indices[i][0], mfep_indices[i][1]] for i in range(1, len(mfep_indices)) ] )
        
        
        # If number of steps is greater than 1 (reactant to product) then we need stitch together
        # all the sub_paths through the intermediate states
        # need to reverse mfep becuase it is [y,x]
        mfep_cv    = [x for l in mfep_cv for x in l]
        mfep_cv    = np.asarray( [x[::-1] for x in mfep_cv] )
        mfep_g     = np.asarray( [x for l in mfep_g for x in l] )
        
        # Get the highest point over all for the TS
        self.results["ts_val"][index] = np.max(mfep_g) - reactant_value
        self.results["ts_loc"].append( mfep_cv[np.argmax(mfep_g)] )
        self.results["minima_vals"].append(states_vals)
        self.results["minima_locs"].append(states_cv)
        self.results["dG"][index] = dG
        self.results["fes"].append(fes_matrix)
        self.results["cv_vectors"].append(cv_vectors)
        
        if self.dims > 1:
            self.results["mfep"].append([mfep_cv, mfep_g])
        
    #------------------------------------------------------------------------------
    #               MINIMUM FINDING FUNCTIONS OF FES CLASS
    #------------------------------------------------------------------------------
    def get_extrema_from_region(self, fes_matrix, cv_vectors, region, extrema="min"):
        """
        Gets the requested extrema type value from a region of CV space from the free energy surface
        
        Args:
            fes_matrix (np.ndarray): a matrix with the fes energy values
            cv_vectors (list of np.ndarray): contains the CV values for each axis of the fes
            region (np.ndarray): region (in CV space) to extract extrema from
            extrema (str): extrema, either "min" or "max"
            
        Returns:
            index (iterable): index of the extrema
            val (float): the value of the extrema
        """
        
        # Very fast due to numba as inner function, first time a bit slower due to compilation
        # also should be done this way to be general for 2 or 3 dimensions
        
        indices = []
        for i in range(len(cv_vectors)):
            indices.append( np.nonzero(np.logical_and(region[0,i] <= cv_vectors[i],  cv_vectors[i] <= region[1,i]))[0] )
        indices.reverse()
        
        grid    = np.meshgrid(*indices, indexing="ij")
        grid    = np.array([grid[i].flatten().tolist() for i in range(len(grid))] ).T
        points  = [ tuple(grid[i]) for i in range(len(grid)) ]
        
        if extrema.lower() in ["max", "maximum"]:
            index, val = get_max_from_indices(fes_matrix, points)
        else:
            index, val = get_min_from_indices(fes_matrix, points)
        
        return index, val
    
    
    def get_intermediates(fes_matrix, cv_vectors, regions_list):
        """
        Gets the intermediates as minimums of the fes matrix for each region of CV space 
        provided in regions_lst
        
        Args:
            fes_matrix (np.ndarray): a matrix with the fes energy values
            cv_vectors (list of np.ndarray): contains the CV values for each axis of the fes
            regions_list (list of np.ndarray): regions (in CV space) to extract extrema from
            
        Returns:
            min_details (tuple of (int, float)): contains index and value of intermediate minimums
        """
        min_details    = [get_minimum_from_region(fes_matrix, cv_vectors, regions_list[x]) for x in range(len(regions_list)) ] 
        return min_details
    
    
    # numba actually slower, even ignoring initial compilation time
    def get_cv_loc_from_index(self, cv_vectors, index):
        """
        Gets the CV values for each axis, given the index of the fes_matrix
        
        Args:
            cv_vectors (list of np.ndarray): contains the CV values for each axis of the fes
            index (iterable): index of the fes_matrix
            
        Returns:
            cv_point (list): the value of the CV(s) at that index
        """
        corrected_index = index[::-1]
        list = []
        cv_point = [list.append(cv_vectors[i][j])  for (i,j) in zip( range(len(cv_vectors)), corrected_index ) ]
        return list
        


#==============================================================================
#                           Dijkstra's Algorithm
#==============================================================================
def dijkstra_2d(mat, start):
    """
    Performs Dijkstra's graph search algorithm to find distance from a starting cell to 
    all other cells of a 2D matrix. 
    
    Args:
        mat (np.ndarray): a matrix with the fes energy values (2 dimensions)
        start (iterable): contains index of the starting cell of the matrix
        
    Returns:
        m (np.ndarray): matrix containing distance values of graph search from start
        P (dict): contains the connections between matrix element points
    """
    
    # visit_mask to hold 1 if visited, 0 if unvisited
    visit_mask  = np.zeros_like(mat, dtype=int)
    #m to hold the best distance to any point as the algoritm proceeds
    m = np.ones_like(mat) * np.inf
    # connectivity here is a list of possible step options: orthogonal & diagonal
    connectivity = [(i,j) for i in [-1, 0, 1] for j in [-1, 0, 1] if (not (i == j == 0))]

    #initialize starting point
    cc = start # current_cell
    m[cc] = 0
    # P will be dictionary of predecessors 
    P = {}
    #need to iterate until all locations on the graph are visited
    for arbitrarycounter in range(mat.size):
        #only test points if they are in the mat matrix, but do allow include edges
        neighbors = [tuple(e) for e in np.asarray(cc) - connectivity 
                    if e[0] >= 0 and e[1] >= 0 and e[0] < mat.shape[0] and e[1] < mat.shape[1] ]
        neighbors = [ e for e in neighbors if not visit_mask[e] ]
        tentative_distance = [abs(mat[e] - mat[cc]) for e in neighbors]
        tentative_distance = np.asarray([15.0 if x==0.0 else x for x in tentative_distance])
        for i,e in enumerate(neighbors):
            d = tentative_distance[i] + m[cc]
            if d < m[e]:
                m[e] = d
                P[e] = cc
        
        visit_mask[cc] = 1
        m_masked = np.ma.masked_array(m, visit_mask)
        cc = np.unravel_index(m_masked.argmin(), m.shape)
    return m, P


# very fast already, doesn't need numba speed up
def shortest_path(start, end, P):
    """
    Returns the shortest path from the starting cell to the ending cell
    based on the results of Dijkstra's graph search algorithm
    
    Args:
        start (iterable): contains index of the starting cell of the matrix
        end (iterable): contains index of the ending cell of the matrix
    Returns:
        path (list of iterables): shortest path connecting start and end
    """
    
    path = []
    step = end
    while 1:
        path.append(step)
        if step == start: break
        try:
            step = P[step]
        except KeyError:
            #print("The error occurred")
            break
    path.reverse()
    return path

#==============================================================================
#                           Simple Vector Functions
#==============================================================================
def paralell_vec(vector):
    """ 
    Returns a normalized vector parralell the original
    
    Args:
        vector (np.array): a vector
    
    Returns:
        par_vec (np.array): a vector parralell to input vector
    """
    
    par_vec = vector / np.linalg.norm(vector)
    return par_vec


def perpendicular_vec_2d(vector):
    """ 
    Returns a normalized vector perpendicular the original in 2 dimensions
    
    Args:
        vector (np.array): a vector
    
    Returns:
        per_vec (np.array): a vector perpendicular to input vector
    """
    
    par_vec = paralell_vec(vector)
    rot_mat = np.asarray([[0, 1], [-1, 0]])
    perp_vec = np.matmul(par_vec, rot_mat)
    return perp_vec
    

#==============================================================================
#                          Nudged Elastic Band Algorithm
#==============================================================================
def neb_2d(mat, neb_path, maxiter=50, convergence=1.0, spring_const=1.2, dt=2.0, freeze=30, climb=10):
    """ 
    Performs the Nuged Elastic Band algorithm
    
    This algorithm has optional climbing images near the tranition state and will
    run until converged as determined by a threshold, or until maxiter 
    iterations are compelted. Returns the minimum energy path in matrix indices.
    
    Args:
        mat (np.array): a 2D matrix filled with energy values
        initial_path (list of tuples): A guessed path in matrix indices. (dimension: 2 by num_images)
        maxiter (int): Maximum iterations before finsihing
        convergence (float): Required fraction of similarity for convergence
        spring_const (float): 
        dt (float): A measure of step size for each force propogation update
        freeze (int): Which iteration to freeze images not near the tranition state
        climb (int): Which iteration to begin clibing iamges near the trandition state
        
    Returns:
        neb_path (np.array): The minimum energy path in matrix indices 
    """

    # setup necessary constants for the calcuation
    iteration = 0
    converged = False
    
    #Gradient
    grad_y, grad_x = np.gradient(mat, edge_order=2)
    neb_history = []
    
    # loop over iterations (max num or until converged)
    while not converged and iteration <= maxiter:
    
        # determine vector parrallell to each image by determining rise and run of
        # each image in regards to both next and last
        z_path = [mat[y][x] for y,x in neb_path]
        z_max_ind = z_path.index(max(z_path))
        
        change_vec_lst = []
        neb_path_arr = [np.asarray(neb_path[i]) for i in range(len(neb_path)) ]
        
        for index in range(len(neb_path)):
            if index == 0:
                # first point will use forward difference
                change_vec = neb_path_arr[index + 1] - neb_path_arr[index]
            
            elif index == (len(neb_path) - 1):
                # last point will use backward difference
                change_vec = neb_path_arr[index] - neb_path_arr[index - 1]
            
            else:
                if z_path[index-1] < z_path[index] < z_path[index+1]:
                    # forward in this case
                    change_vec = neb_path_arr[index + 1] - neb_path_arr[index]
                
                elif z_path[index-1] > z_path[index] > z_path[index+1]:
                    #backward in this case
                    change_vec = neb_path_arr[index] - neb_path_arr[index-1]
                
                else:
                    #one weighted average in this case
                    dV_for = abs(z_path[index+1] - z_path[index])
                    dV_back = abs(z_path[index-1] - z_path[index])
                    dVmax = max(dV_for, dV_back)
                    dVmin = min(dV_for, dV_back)
                    for_vec =  neb_path_arr[index + 1] - neb_path_arr[index]
                    back_vec = neb_path_arr[index] - neb_path_arr[index-1]
                    
                    if z_path[index+1] > z_path[index-1]:
                        change_vec = dVmax*for_vec + dVmin*back_vec
                    else:
                        change_vec = dVmin*for_vec + dVmax*back_vec
            
            change_vec_lst.append(change_vec)
        
        # Normalize the paralell vectors at each image
        # array is [[y1,x1], [y2,x2]] 
        par_vec_image_lst =  [ paralell_vec(vec) for vec in change_vec_lst ]
        
        # F_perpendicular = total force on image minus parallel force on image
        # F_perp_i = −∇E(Ri) + parvec*(∇E(Ri)·parvec) 
        grad_at_image_lst = [ np.asarray( [grad_y[y][x], grad_x[y][x]] ) for (y, x) in neb_path ]
        F_perp_lst = [ -1.0 * grad_at_image + par_vec*np.dot(grad_at_image, par_vec) for grad_at_image, par_vec in zip(grad_at_image_lst, par_vec_image_lst) ]
        
        # determine elastic band spring vector forces on each image
        # F_spring_i = ki+1(R_i+1 − R_i) − ki(R_i − R_i−1)
        F_spring_lst = []
        for index in range(len(neb_path)):
            if index == 0 or index == (len(neb_path) - 1):
                f_spring_par = np.zeros(2)
            else:
                f_spring_mag = spring_const * (np.linalg.norm(neb_path_arr[index + 1] - neb_path_arr[index]) \
                    - np.linalg.norm(neb_path_arr[index] - neb_path_arr[index - 1]) )
                f_spring_par = f_spring_mag * par_vec_image_lst[index]
            F_spring_lst.append(f_spring_par)
        
        # get total forces on each image
        # There should be NO FORCES on the ENDPOINTS (these never move)
        # F_tot = F_perp_i + (F_spring_i)·parvec) parvec
        F_tot = [f_perp + f_spring for f_perp, f_spring in zip(F_perp_lst, F_spring_lst)]
        
        #Freeze images not near the TS region to avoid oscillation
        if iteration >= freeze and len(neb_path) > 10:
            for i in range(len(neb_path)):
                if not (z_max_ind - 3 <= i <= z_max_ind + 3):
                    F_tot[i] = np.asarray([0, 0])
        
        #make the current highest point climb
        F_grad_climber = -1.0 * grad_at_image_lst[z_max_ind] + \
            2 * par_vec_image_lst[z_max_ind] * np.dot(grad_at_image_lst[z_max_ind], par_vec_image_lst[z_max_ind])
        for i in range(len(F_grad_climber)):
            if F_grad_climber[i] > 0:
                F_grad_climber = np.ceil(F_grad_climber)
            else:
                F_grad_climber = np.floor(F_grad_climber)
        F_perp_lst[z_max_ind] = F_grad_climber
        F_spring_lst[z_max_ind] = np.zeros(2)
        
        #move the images according to their force
        # zip(*variable) will unzip a list = reverse of using zip on 2 lists
        #then need to decide how to update forces, need to specify whole matrix index steps
        to_move = [np.nan_to_num(dt * projvec, nan=0.0, posinf=5.0, neginf=-5.0) for projvec in F_tot]
        to_move = [np.where(a <= 2.0, a, 2.0) for a in to_move]
        to_move = [np.where(a >= -2.0, a, -2.0) for a in to_move]
        to_move = [np.around(a , decimals=0) for a in to_move]

        neb_path = [oldloc + movement for oldloc, movement in zip(np.asarray(neb_path), to_move)]
        neb_path = [list([int(y), int(x)]) for y,x in neb_path]
        
        #Keep the path inside the matrix
        for index in range(len(neb_path)):
            if neb_path[index][0] < 0:
                neb_path[index][0] = 0
            if neb_path[index][0] >= mat.shape[0]:
                neb_path[index][0] = mat.shape[0] - 1
            if neb_path[index][1] < 0:
                neb_path[index][1] = 0
            if neb_path[index][1] >= mat.shape[1]:
                neb_path[index][1] = mat.shape[1] - 1
    
        # calculate convergence measure(s)
        # allow the newest path to be same as last path OR to second to last path 
        # so continous loops can be avoided
        if iteration >= 5:
            test_convergence = np.zeros(len(neb_path))
            for index in range(len(neb_path)):
                if neb_path[index] == neb_history[-1][index] or neb_path[index] == neb_history[-2][index]:
                    test_convergence[index] = 1.0
            perc_same = np.sum(test_convergence) / float(len(neb_path))
            
            if perc_same >= convergence: 
                converged = True
        
        #append the current neb to the history list
        neb_history.append(neb_path)
        
        iteration += 1
    
    # when done, return the neb_path in matrix index space
    # need to convert to the location in CV space outside of function
    
    return neb_path
         

