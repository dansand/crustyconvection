import numpy as np
#import matplotlib.pyplot as plt
import collections
import itertools

# a small set of helper functions to
# call common array creation functions
# these are useful to ensure that
# all arrays are created as double-precision
# floats, no matter what data are provided
# as argument. For example array([1,3,4]) normally returns
# an array with data of type int, but arrayf([1,3,4])
# always creates an array of floats

kFloatType = np.float64

def arrayf( arg ):
    return np.array( arg, kFloatType )
def asarrayf( arg ):
    return np.asarray( arg, kFloatType )
def zerosf( arg ):
    return np.zeros( arg, kFloatType )
def identityf( arg ):
    return np.identity( arg, kFloatType )
def emptyf( arg ):
    return np.empty( arg, kFloatType )




# %load scripts/springs.py
#from numpy import *
#from myarray import *

## This is the spring constant
k = 1.

## This is our threshold for whether a number is approximately zero.
ZERO_THRESHOLD = 1e-8

def F_ij( p_i, p_j, r_ij ):
    '''
    Returns the force of the spring from 'p_i' to 'p_j' with rest length 'r_ij' acting on 'p_i'.
    Note that F_ij( p_i, p_j, r_ij ) equals -F_ij( p_j, p_i, r_ij ).
    '''

    p_ij = p_i - p_j
    len_p_ij = np.sqrt( sum( p_ij ** 2 ) )

    if abs( len_p_ij ) < ZERO_THRESHOLD:
        result = 0. * p_ij
    else:
        result = -k * ( len_p_ij - r_ij ) / len_p_ij * p_ij

    return result

def F( p, edges, edges_rest_lengths ):
    '''
    Returns a vector containing the force at every point.
    Note that the input 'p' is assumed to be a number-of-points by dimension array, where
    dimension is 2 for our example.  The result is flattened into a single vector
    of size number-of-points times dimension.
    '''

    dim = p.shape[1]

    Fp = zerosf( np.prod( p.shape ) )

    ## Loop over every edge and its corresponding rest length.
    ## (zip() simply combines two lists into one so we can loop over them together.)
    for (i,j), r_ij in zip( edges, edges_rest_lengths ):
        assert i != j

        Fij = F_ij( p[i], p[j], r_ij )

        Fp[ i*dim : (i+1) * dim ] += Fij
        ## 'edges' contains edges uniquely, so 'edges' will contain (i,j) but not (j,i).
        ## This means that we must add -Fij to row j as well as Fij to row i.
        Fp[ j*dim : (j+1) * dim ] += -Fij

    return Fp

def dF_ij_d_p_i( p_i, p_j, r_ij ):
    '''
    Returns the derivative with respect to 'p_i' of the force of the spring
    from 'p_i' to 'p_j' with rest length 'r_ij' acting on 'p_i'.
    Our dimension is 2, so this is a 2x2 quantity.
    Note that dF_ij_d_p_i( p_i, p_j, r_ij ) equals dF_ij_d_p_i( p_j, p_i, r_ij ).
    '''

    dim = p_i.shape[0]

    p_ij = p_i - p_j
    len_p_ij = np.sqrt( sum( p_ij ** 2 ) )
    if abs( len_p_ij ) < ZERO_THRESHOLD:
        result = -k * np.identity( dim )
    else:
        result = -k * np.identity( dim ) - k * r_ij / len_p_ij**3 * np.outer( p_ij, p_ij ) + k * r_ij / len_p_ij * np.identity( dim )
    return result

def J( p, edges, edges_rest_lengths ):
    '''
    Returns a matrix containing the derivative of the force at every point with respect to each point.
    Note that the input 'p' is assumed to be a number-of-points by dimension array, where
    dimension is 2 for our example.
    The result is flattened is a square matrix (of type numpy.array), of size NxN, where
    N = number-of-points times dimension.
    '''

    dim = p.shape[1]

    Jp = zerosf( ( np.prod( p.shape ), np.prod( p.shape ) ) )

    ## Loop over every edge and its corresponding rest length.
    ## (zip() simply combines two lists into one so we can loop over them together.)
    for (i,j), r_ij in zip( edges, edges_rest_lengths ):
        assert i != j

        dF = dF_ij_d_p_i( p[i], p[j], r_ij )
        assert ( ( Jp[ i*dim : (i+1) * dim, j*dim : (j+1) * dim ] - np.zeros( ( dim, dim ) ) ) ** 2 ).sum().sum() == 0.

        Jp[ i*dim : (i+1) * dim, j*dim : (j+1) * dim ] = -dF
        Jp[ i*dim : (i+1) * dim, i*dim : (i+1) * dim ] += dF
        ## 'edges' contains edges uniquely, so 'edges' will contain (i,j) but not (j,i).
        ## This means that we must add dF to the right places in column j as well.
        Jp[ j*dim : (j+1) * dim, i*dim : (i+1) * dim ] = -dF
        Jp[ j*dim : (j+1) * dim, j*dim : (j+1) * dim ] += dF

    return Jp

def constrain_system( A, rhs, rows ):
    '''
    This function modifies its input parameters, a system matrix 'A' and
    right-hand-side vector 'rhs', such that for every index i in 'rows',
    the row i of A is set to row i of the identity matrix and rhs[i] is set to zero.
    '''

    for i in rows:
        A[ i, : ] = np.zeros( A.shape[1] )
        ## We can also zero the column, which keeps the matrix symmetric, because
        ## we are zeroing the corresponding entries in the right-hand-side (x*0 = 0).
        A[ :, i ] = np.zeros( A.shape[0] )
        A[ i, i ] = 1
        rhs[i] = 0

    return A, rhs

def static_solution( p, edges, edges_rest_lengths, constraints, verbose = True ):
    '''
    Given a list of points 'p' as an n-by-2 array, a list of (i,j) pairs 'edges' denoting an edge
    between points p[i] and p[j], a list of rest lengths (one for each edge in 'edges'),
    and a list of position constraints (i, position) denoting p[i] = position,
    uses Newton's method to solve for the positions where the forces are all zero.

    NOTE: 'edges' must not have both (i,j) and (j,i)
    '''

    XSTEP_THRESHOLD = 1e-5
    F_THRESHOLD = 1e-8
    MAX_ITERATIONS = 100

    p_n = p.copy().flatten()
    dim = p.shape[1]

    constrain_rows = []
    for i, p_val in constraints:
        p_n[ i*dim : (i+1) * dim ] = p_val
        constrain_rows.extend( range( i*dim, (i+1) * dim ) )

    iteration = 0
    while True:
        if verbose: print '-- iteration', iteration, '--'
        iteration += 1

        Jp_n = J( p_n.reshape( p.shape ), edges, edges_rest_lengths )
        Fp_n = F( p_n.reshape( p.shape ), edges, edges_rest_lengths )
        mag2_Fp_n = sum( Fp_n ** 2 )
        if verbose: print '| F( p_n ) |^2:', mag2_Fp_n
        if mag2_Fp_n < F_THRESHOLD: break

        constrain_system( Jp_n, Fp_n, constrain_rows )

        # p_n_p_1 = p_n - dot( linalg.inv( Jp_n ), Fp_n )
        ## <=> p_n_p_1 - p_n = -linalg.inv( Jp_n ) * Fp_n
        ## <=> p_n - p_n_p_1 = np.linalg.inv( Jp_n ) * Fp_n
        ## <=> Jp_n * ( p_n - p_n_p_1 ) = Fp_n
        p_negative_delta = np.linalg.solve( Jp_n, Fp_n )
        ## p_n - ( p_n - p_n_p_1 ) = p_n_p_1
        p_n_p_1 = p_n - p_negative_delta

        diff2 = sum( ( p_n_p_1 - p_n ) ** 2 )
        if verbose: print '| p_n+1 - p_n |^2:', diff2
        p_n = p_n_p_1
        if diff2 < XSTEP_THRESHOLD: break

        if iteration >= MAX_ITERATIONS:
            print 'Diverged.'
            return p.copy()
            break

    return p_n.reshape( p.shape )

def compute_edge_lengths( p, edges ):
    '''
    Given a list of (i,j) pairs 'edges' denoting an edge between points p[i] and p[j],
    returns a list of rest lengths, one for each edge in 'edges'.

    NOTE: 'edges' must not have both (i,j) and (j,i)
    '''

    ## Check for duplicate edges, which are forbidden.
    edges = tuple( map( tuple, edges ) )
    from sets import ImmutableSet as Set
    assert len( Set( map( Set, edges ) ) ) == len( edges )

    result = []
    for i,j in edges:
        len_p_ij = np.sqrt( sum( (p[i] - p[j]) ** 2 ) )
        result.append( len_p_ij )

    return result


def node_neighbours(mesh):
    nodes = mesh.data_nodegId.reshape(mesh.elementRes[0] +1, mesh.elementRes[1] +1)
    testlist = []
    for index, value in np.ndenumerate(nodes):
        #print index, value
        #####Get x neighbour:
        if index[0] + 1 < nodes.shape[0]:
            testlist.append((nodes[index], nodes[(index[0] + 1),index[1] ])) #Add in the X direction
        #####Get Y neighbour:
        if index[1] + 1 < nodes.shape[1]:
            testlist.append((nodes[index], nodes[index[0], (index[1]+ 1) ])) #Add in the X direction
    return testlist

def mesh_element_keys(mesh):
    xkeys = np.linspace(mesh.minCoord[0], mesh.maxCoord[0], mesh.elementRes[0] + 1)
    ykeys = np.linspace(mesh.minCoord[1], mesh.maxCoord[1], mesh.elementRes[1] + 1)

    a = xkeys
    b = ykeys
    for i in range(mesh.elementRes[1]):
        a = np.row_stack((a, xkeys))
    for j in range(mesh.elementRes[0]):
        b = np.column_stack((b, ykeys))
    return a.flatten(), b.flatten()

def find_closest(A, target):
    #A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    return idx

def deform_1d(f, mesh,axis = 'x',norm = 'None', constraints = []):
    """This function deforms the mesh along the given axis,
    by solving an anlogous spring system. The deformation is described by:
    f: function, or vector, giving the equilibrium spring lengths.
        If a function is given, the equilibrium positions are treated a function of space
        and the function solves for the current node/spring positions
        then remaps the spatial function onto the updated position
        If a vector is give, the equilibrium spring lengths are considered a fucntion of
        the springs themselves and only one equilibrium solve is needed
        (though this is still a non-linear system)
    axis: the axis to deform along, x or y
    norm: You can choose to normalise the function / vector in a couple of different ways:
        "None", vector / function isn.t normalised, spings will be in tension / compression
        "Uniform": vector / function is normalised to axis width uniformally
        "Min": vector / function is normalised to axis width, while preserving the min separation
    constraints: extra constraints for the system (the mesh never changes size, but inernal nodel can be fixed)
    """
    xkeys, ykeys = mesh_element_keys(mesh)
    if axis == "y":
        thisaxis = 1
        usekeys = ykeys
    else:
        thisaxis = 0
        usekeys = xkeys
    print(norm)
    if not callable(f):
        #Sort out normalisation
        if norm == 'None':
            pass
        elif norm == 'Uniform':
            f = f/f.sum()
        elif norm == 'Min':
            beta = f.sum()
            N = mesh.elementRes[thisaxis]
            fm = min(f)
            alpha = ((mesh.maxCoord[thisaxis] - mesh.minCoord[thisaxis]) - beta)/(beta - N*fm)
            print(N, fm, beta, alpha)
            f = np.array(f) + ((np.array(f) - fm)*alpha)
        #p_undeformed  = mesh.data[:mesh.elementRes[ thisaxis] +1,:] #This need fixing
        p_undeformed  = np.array(zip(np.linspace(mesh.minCoord[thisaxis], mesh.maxCoord[thisaxis],
                                        mesh.elementRes[thisaxis] + 1), np.zeros(mesh.elementRes[thisaxis] + 1 )))


        mesh_nodes= np.arange(0, p_undeformed.shape[ 0] , 1)

        edges = [(int(i), int(i+1)) for i in mesh_nodes[:-1]]
        constraints = [ ( mesh_nodes[0], p_undeformed[ 0] ), ( mesh_nodes[-1], p_undeformed[-1] ) ] + constraints#Add egde constraints
        p_initial = p_undeformed.copy()
        print ('edges', len(edges))
        p_solution = static_solution( p_initial, edges, f, constraints )
        origxcoords = np.around(np.linspace(mesh.minCoord[thisaxis ], mesh.maxCoord[thisaxis ], mesh.elementRes[thisaxis ] + 1), 5)
        dictionary = dict(itertools.izip(origxcoords, p_solution[:,0]))
        #print dictionary
        with mesh.deform_mesh():
            for index, coord in enumerate(mesh.data):
                mesh.data[index][thisaxis] = dictionary[np.around(coord,5)[thisaxis]]
        #Print some stats
        print("Min, Max element width: ")
        print("%.5f" % min(compute_edge_lengths(p_solution, edges )))
        print("%.5f" % max(compute_edge_lengths(p_solution, edges )))
