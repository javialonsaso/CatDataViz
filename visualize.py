################################################################################
#                                                                              #
#         Visual representation of multidimensional categorical data           #
#                                                                              #
################################################################################

import sys
from pathlib import Path
PATH_ROOT = Path(__file__).parent
sys.path.append(PATH_ROOT.as_posix())

import numpy as np
import pandas as pd
import loader_template as ld
from dataObject import DataObject

from sklearn.manifold import MDS, TSNE, Isomap

# Load dataframe
print('Loading data...')
data = ld.dataLoader()

# Create dataObject object
print('Creating dataObject instance...')
do = DataObject(data)

# Different methods for embedding
mds = MDS(n_components = 2,
    metric = True,
    n_init = 4,
    verbose = 0,
    random_state = 50,
    dissimilarity = 'precomputed',
)
iso = Isomap(
    n_neighbors=50,
	n_components=2,
	eigen_solver='auto',
	tol=0,
	max_iter=None,
	path_method='auto',
	neighbors_algorithm='auto',
	n_jobs=None,
	metric='precomputed',
	p=2,
    metric_params=None)
tsne = TSNE(n_components=2,
	perplexity=5.0,
	early_exaggeration=6.0,
	learning_rate=5.0,
	n_iter=1000,
	n_iter_without_progress=300,
	min_grad_norm=1e-07,
	metric='precomputed',
	init='random',
	verbose=0,
	random_state=42,
	method='barnes_hut',
	angle=0.5,
	n_jobs=1,
    square_distances=True,
)

# Data embedding
print('2D embedding method...')
do.embedding(tsne)

# Border vertices of GMap
print('GMap algorithm...')
k = 0
patches_x, patches_y, labels, labels_num = do.gmap(k = k)


### GRAPH ###
print('Drawing...')
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
from bokeh.palettes import Spectral8, Spectral4, PuOr, magma


# Add parameters to show with hover tool
TOOLTIPS = [
    ("id", "@id"),
    ("lbl", "@lbl"),
]

# Create figure
p = figure(title = data.columns[k],
    tools="pan,lasso_select,box_select,wheel_zoom,hover", tooltips = TOOLTIPS)
p.sizing_mode = 'stretch_both'

# Objecto to save data positions
drawData = ColumnDataSource(dict(
    x = do.euclid2d[:,0],
    y = do.euclid2d[:,1],
    id = [id for id in data.index.tolist()],
    lbl = data.iloc[:,k],
))

# Object to save patch display information
palette = magma(np.unique(labels).size)
patches = ColumnDataSource(dict(
    xs = patches_x,
    ys = patches_y,
    color = [palette[i] for i in labels_num],
    id = [i for i in range(len(patches_x))],
    lbl = labels,
))

# Create glyphs on figure
p.circle(x = 'x', y = 'y', color = 'black', radius=.5, alpha=0.5, source = drawData)
p.patches('xs', 'ys', alpha=0.5, color = 'color', source = patches)

show(p)
