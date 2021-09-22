################################################################################
#                                                                              #
#                            DataObject Class                                  #
#              Information storage of categorical data points                  #
#                                                                              #
################################################################################

import sys
from pathlib import Path
PATH_ROOT = Path(__file__).parent
sys.path.append(PATH_ROOT.as_posix())
sys.path.append((PATH_ROOT.parent / 'gmap').as_posix())

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist as cdist

from gmap import gmap

def similarity(a,b):
    """Computes the probability of 'a' being the same object as 'b' defined
    both by categorical values.

    Parameters
    ----------
    a       : 1D array
    b       : 1D array

    Output
    ------
    The probability of P(a=b|a,b). If a and b are of different sizes, it
    returns 0.
    """

    if a.shape != b.shape:
        return 0.
    return 1 - (a != b).sum() / a.shape[0]

class DataObject():
    """DataObject class stores information of categorical data points, 2D
    embedding and GMap regions.

    Variables
    ---------
    tt      : Pandas DataFrame where labels are integers.
    maps    : Dictionary that maps label names with values of 'tt'.
    maps_inv: As 'maps' but inverse.
    euclid2d: 2D array of positions of data points in the embedded space.
    """

    def __init__(self, table):
        """Create instance of dataObject.

        Parameters
        ----------
        table   : Pandas DataFrame with unique index and categorical variables
                  only. (n_samples, n_features)
        """

        tt = table.copy(deep = True)
        # Mapping classifications
        maps = {}
        maps_inv = {}
        for col in table:
            mapping = { oldName : newName for newName, oldName in enumerate(
                np.sort(table.loc[:,col].unique())
                ) }
            maps[col] = mapping
            maps_inv[col] = {v : k for k,v in mapping.items()}
            tt.loc[:,col] = tt[col].map(mapping)

        self.tt = tt
        self.maps = maps
        self.maps_inv = maps_inv


    def embedding(self, method):
        """Embed data points in a 2d euclidean space. It defines a n_sample euclidean space using a similarity measure [1].

        Parameters
        ----------
        method  : scikit-learn estimator (sklearn.manifold) to embed data.

        References
        ----------
        [1] Y. Qian, F. Li, J. Liang, B. Liu and C. Dang, "Space Structure and
        Clustering of Categorical Data," in IEEE Transactions on Neural Networks
        and Learning Systems, vol. 27, no. 10, pp. 2047-2059, Oct. 2016, doi:
        10.1109/TNNLS.2015.2451151.
        """

        # Find n_sample euclidean space
        euclid = cdist(self.tt, self.tt, metric = similarity)

        # Precompute distances
        dist = cdist(euclid, euclid, metric = 'euclidean')

        # Perform embedding method
        self.euclid2d = method.fit_transform(dist)
        # Garantees data points are not overlapped
        self.__randomPush__()
        self.euclid2d = __relaxGraph__(self.euclid2d)

    def gmap(self, k = 0):
        """Generate ouliers to define outer boundaries and generate gmap
        boundaries.

        Parameters
        ----------
        k       : (Optional) Integer. Which column of the original dataframe is used to
                  define labels used in gmap.

        Output
        ------
        patches_x   : List of arrays of x-coordinates of vectors for borders.
        patches_y   : List of arrays of y-coordinates of vectors for borders.
        lbl_names   : Label names for each zone.
        lbl_regions : Label id for each block.
        """

        # Parameters of size data cloud
        std = cdist(self.euclid2d, self.euclid2d, 'euclidean').std()
        maxx = cdist(self.euclid2d, self.euclid2d, 'euclidean').max() *0.5

        # Create outliers
        outliers = np.random.rand(100000,2)
        outliers = maxx * (outliers[:,0]*0.5+0.8)[:, np.newaxis] * np.array([np.cos(outliers[:,1]*2*np.pi), np.sin(outliers[:,1]*2*np.pi)]).T
        outliers += self.euclid2d.mean(0)

        # Filter out too-close-to-data
        outliers = outliers[(cdist(outliers, self.euclid2d, 'euclidean') > std*0.3).all(1),:]

        # Find borders
        patches_x, patches_y, lbl_regions, vertices_pos = gmap(self.euclid2d, outliers, self.tt.values[:, 0])
        lbl_names = [self.maps_inv[list(self.maps_inv.keys())[k]][lbl] for lbl in lbl_regions]

        return patches_x, patches_y, lbl_names, lbl_regions




    def __randomPush__(self):
        """Random slight shift to every data point in the embedded space.
        """
        # To work at any scale, find distances between data points
        dd = cdist(self.euclid2d, self.euclid2d, metric = 'euclidean')

        # Define range of distance as alpha-% of std of all distances
        delta = dd[np.triu_indices(dd.shape[0], k = 1)].flatten().std()*0.1

        # Shift in polar coordinates
        rdelta = np.random.rand(*self.euclid2d.shape)*np.array([[delta, 1.]])
        delta = rdelta[:,[0]] * np.array(
            [np.cos(rdelta[:,1]*2*np.pi),
            np.sin(rdelta[:,1]*2*np.pi)]
        ).T

        # Apply shift
        self.euclid2d += delta





import networkx as nx
def __relaxGraph__(points, dist = None, l = 4, ang_0 = np.pi/2.):
    """Shift positions to create uniformly densed data cloud. It uses Kamada Kawai algorithm to data points connected by proximity and angular distance.

    Parameters
    ----------
    points  : 2D array of positions
    dist    : (Optional) Distance matrix of 'points'
    l       : (Optional) Number of sectors to find neighbours around node.
    ang_0   : (Optional) Angle from where to break circle in sectors.

    Output
    ------
    newPos  : Newly defined positions
    """
    # Create edgeless graph
    g = nx.Graph()
    G = nx.Graph()
    nodes = []
    for i, point in enumerate(points):
        nodes.append(
        (i, {'x' : point[0], 'y' : point[1]})
        )
    G.add_nodes_from(nodes)

    # Build edges according to proximity and angular position
    ang = cdist(points, points, lambda p1, p2: np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
    ang[ang < 0] += 2*np.pi
    rad = cdist(points, points, metric = 'euclidean') if dist is None else dist # Distances
    secc = np.linspace(ang_0, ang_0 + 2*np.pi, l + 1) % (2*np.pi) # Angular sectors
    for node in G.nodes:
        secc_mask = np.array([False]*l)
        mask = [None]*l

        # Check neighbours in each sector
        nodeList = np.array(list(G.nodes))
        for i in range(l):
            # Angle range is [0, 2PI)
            if secc[i] > secc[i+1]:
                # Check no neighbour is in this sector
                neigh = list(dict(G.adj[node]).keys())
                # Check nodes in sector
                mask[i]  = ( ang[node] >= secc[i  ] ) * ( ang[node] < 0. )
                mask[i] += ( ang[node] <= secc[i+1] ) * ( ang[node] > 0. )
                # Exclude self point
                mask[i][node] = False
                secc_mask[i] = mask[i][neigh].any()

            else:
                # Check no neighbour is in this sector
                neigh = list(dict(G.adj[node]).keys())
                mask[i] = ( ang[node] >= secc[i] ) * ( ang[node] < secc[i+1] )
                # Exclude self point
                mask[i][node] = False
                secc_mask[i] = mask[i][neigh].any()

        # Find new neighbours in empty sectors
        for i in range(l):
            if secc_mask[i]:
                continue

            if secc[i] > secc[i+1]:
                # Add closest neighbour (in case there is one)
                try:
                    ii = np.argsort(rad[node, mask[i]])[0]
                except IndexError:
                    continue
                otherNode = node_list[mask[i]][ii]
                G.add_edges_from([(node, otherNode, {'weight' : rad[node, otherNode]})])

            else:
                # Add closest neighbour (in case there is one)
                try:
                    ii = np.argsort(rad[node, mask[i]])[0]
                except IndexError:
                    continue
                otherNode = nodeList[mask[i]][ii]
                G.add_edges_from([(node, otherNode, {'weight' : rad[node, otherNode]})])

    r = nx.kamada_kawai_layout(G, weight = None, pos = points, scale = 60)
    newPos = np.array([r[i] for i in G.nodes])
    return newPos
