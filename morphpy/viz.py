import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import scipy.spatial.distance as sp
import seaborn as sns
import pandas as pd
import scipy.cluster.hierarchy as hc
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import manifold
import plotly.io as pio
pio.kaleido.scope.default_format = "pdf"


def plot_curve(X, titlestr=""):
    fig = go.Figure(data=go.Scatter(x=X[0,:], y=X[1,:], mode='lines'))
    fig = set_generic_fig_properties(fig, title_text=titlestr)
    fig.show()


def plot_pairwise_distance_matrix(Dvect):
    Dmat = sp.squareform(Dvect)
    fig = go.Figure(data=go.Heatmap(z=Dmat))
    fig = set_generic_fig_properties(fig, title_text="Geodesic Distance matrix")
    fig.show()
    fig.write_image("geodesic_distance_matrix.pdf")
    np.savetxt("Geodesic_Distances.csv", Dmat, delimiter=",")


def plot_shapes(Xarray1):
    N, n, T = np.shape(Xarray1)
    # n has to be equal to 2
    Xarray = np.copy(Xarray1)
    Xarray = center_and_rescale_curves(Xarray)

    fig = make_subplots(rows=int(np.ceil(N**0.5)), cols=int(np.ceil(N ** 0.5)))
    rows = int(np.ceil(N**0.5))
    cols = int(np.ceil(N**0.5))

    for ii in range(N):
        ix, iy = np.unravel_index(ii, (rows, cols))
        fig.add_trace(go.Scatter(x=Xarray[ii][0,], y=Xarray[ii][1,], mode='lines'), row=ix.tolist()+1, col=iy.tolist()+1)

    fig = set_generic_subplot_fig_properties(fig, title_text="All shapes")
    fig.show()
    fig.write_image("allshapes.pdf")


def set_generic_subplot_fig_properties(fig, height=600, width=600, title_text="", showticks=False, showlegend=False):
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)
    fig.update_layout(autosize=True, height=height, width=width, title_text=title_text,
                      showlegend=showlegend,
                      margin=dict(r=5, l=5, t=25, b=5))
    axes = [fig.layout[e] for e in fig.layout if e[1:5] == 'axis']
    for ii in range(1, len(fig.data)+1):
        fig['layout']['yaxis'+str(ii)].update(scaleanchor="x"+str(ii), scaleratio=1)
    fig.update_xaxes(showticklabels=showticks)
    fig.update_yaxes(showticklabels=showticks)

    return fig


def set_generic_fig_properties(fig, height=600, width=600, title_text="", showticks=False, showlegend=False):
    fig.update_layout(autosize=True, height=height, width=width, title_text=title_text,
                      yaxis=dict(scaleanchor="x"), showlegend=showlegend,
                      margin=dict(r=5, l=5, t=25, b=5), scene=dict(aspectmode="data"), yaxis2=dict(scaleanchor="x", scaleratio=1))
    fig.update_xaxes(showticklabels=showticks)
    fig.update_yaxes(showticklabels=showticks, autorange="reversed", scaleratio=1)
    fig.update_layout(xaxis_visible=True, xaxis_showticklabels=True, yaxis_visible=True, yaxis_showticklabels=True)

    return fig


def show_table(df):

    fig = go.Figure(data=[go.Table(
        header=None,
        cells=dict(values=df.transpose().values.tolist(),
                   fill_color='lavender',
                   align='left'))
    ])
    fig.update_layout(height=300, margin=dict(r=5, l=5, t=5, b=5))
    fig.show()


def plot_dendrogram(D, labelsdf):

    labelnames = labelsdf['name']
    df = pd.DataFrame(data=sp.squareform(D), index=labelnames, columns=labelnames)
    linkage = hc.linkage(D, method='average')
    sns.clustermap(df, row_linkage=linkage, col_linkage=linkage)
    plt.savefig('dendrogram.pdf')
    plt.show()



def plot_deformation_field(pmean, mag_def_field):

    data = go.Scatter(x=pmean[0,:], y=pmean[1,:], mode='markers', marker={'color': mag_def_field, 'colorscale': 'Rainbow', 'size': 10})
    fig = go.Figure(data=data)
    fig = set_generic_fig_properties(fig, title_text="Deformation field plotted along the mean shape")
    fig.show()
    fig.write_image("deformation_field.pdf")


def plot_tsne_embedding(Dvect, labelsdf):

    tsne = TSNE(n_components=2, random_state=0, metric='precomputed')
    projections = tsne.fit_transform(sp.squareform(Dvect))
    labelnames = labelsdf['label']
    fig = px.scatter(
        projections, x=0, y=1,
        color=labelnames, labels={'color': 'names'}
    )
    fig.show()


# TODO The MDS calculation could be separated from plotting
def plot_MDS(Dvect, labelsdf):

    seed = np.random.RandomState(seed=3)
    mds = manifold.MDS(n_components=3, max_iter=3000, eps=1e-9, random_state=seed,
                       dissimilarity="precomputed", n_jobs=1)
    pos = mds.fit(sp.squareform(Dvect)).embedding_

    # fig = go.Figure(data=go.Scatter(x=pos[0,:], y=pos[1,:]))
    labelsdf['X'] = pos[:, 0]
    labelsdf['Y'] = pos[:, 1]
    labelsdf['Z'] = pos[:, 2]

    # Rescale the scale column to lie between 0 and 1
    labelsdf['scale'] = labelsdf['scale']/max(labelsdf['scale'])

    fig = px.scatter(labelsdf, x="X", y="Y", color="label",
                     size='scale', hover_data=['name'], text="name")
    fig = set_generic_fig_properties(fig, title_text='Multidimensional Scaling', showlegend=True)
    fig.show()
    fig.write_image("mdsplot_X_Y.pdf")

    fig = px.scatter(labelsdf, x="X", y="Z", color="label",
                     size='scale', hover_data=['name'], text="name")
    fig = set_generic_fig_properties(fig, title_text='Multidimensional Scaling', showlegend=True)
    fig.show()
    fig.write_image("mdsplot_X_Z.pdf")

    fig = px.scatter(labelsdf, x="Y", y="Z", color="label",
                     size='scale', hover_data=['name'], text="name")
    fig = set_generic_fig_properties(fig, title_text='Multidimensional Scaling', showlegend=True)
    fig.show()
    fig.write_image("mdsplot_Y_Z.pdf")


# TODO The MDS calculation could be separated from plotting
def plot_shapes_on_MDS(Xarray, Dvect, labelsdf):

    k, n, T = Xarray.shape
    seed = np.random.RandomState(seed=3)
    mds = manifold.MDS(n_components=3, max_iter=3000, eps=1e-9, random_state=seed,
                       dissimilarity="precomputed", n_jobs=1)
    pos = mds.fit(sp.squareform(Dvect)).embedding_

    # fig = go.Figure(data=go.Scatter(x=pos[0,:], y=pos[1,:]))
    labelsdf['X'] = pos[:, 0]
    labelsdf['Y'] = pos[:, 1]
    labelsdf['Z'] = pos[:, 2]

    # Rescale the scale column to lie between 0 and 1
    labelsdf['scale'] = labelsdf['scale']/max(labelsdf['scale'])

    fig = px.scatter(labelsdf, x="X", y="Y", color="label",
                     size='scale', hover_data=['name'], text="name")
    newpos = np.vstack((labelsdf['X'], labelsdf['Y']))
    add_shapes_to_fig(fig, Xarray, newpos)
    fig = set_generic_fig_properties(fig, title_text='Multidimensional Scaling', showlegend=True)
    fig.show()
    fig.write_image("mdsplot_shape_X_Y.pdf")

    fig = px.scatter(labelsdf, x="X", y="Z", color="label",
                     size='scale', hover_data=['name'], text="name")
    newpos = np.vstack((labelsdf['X'], labelsdf['Z']))
    add_shapes_to_fig(fig, Xarray, newpos)
    fig = set_generic_fig_properties(fig, title_text='Multidimensional Scaling', showlegend=True)
    fig.show()
    fig.write_image("mdsplot_shape_X_Z.pdf")

    fig = px.scatter(labelsdf, x="Y", y="Z", color="label",
                     size='scale', hover_data=['name'], text="name")
    newpos = np.vstack((labelsdf['Y'], labelsdf['Z']))
    add_shapes_to_fig(fig, Xarray, newpos)
    fig = set_generic_fig_properties(fig, title_text='Multidimensional Scaling', showlegend=True)
    fig.show()
    fig.write_image("mdsplot_shape_Y_Z.pdf")


def center_and_rescale_curves(Xarray):
    k, n, T = Xarray.shape

    # Center and Scale
    for ii in range(k):
        Xarray[ii, :, :] -= np.mean(Xarray[ii, :, :], axis=1).reshape((n, 1))
        Xarray[ii, :, :] /= np.linalg.norm(Xarray[ii, :, :], 'fro')

    return Xarray


def shift_center(Xarray, newpos):

    # No error checking for dimensions of Xarray and pos
    Xarray = center_and_rescale_curves(Xarray)
    k, n, T = Xarray.shape
    for ii in range(k):
        Xarray[ii] += newpos[:, 0][:, np.newaxis]
    return Xarray


def shift_center_and_scale(Xarray, newpos, newscale):

    # No error checking for dimensions of Xarray and pos
    Xarray = center_and_rescale_curves(Xarray)
    Xarray *= newscale
    k, n, T = Xarray.shape
    for ii in range(k):
        Xarray[ii] += newpos[:, ii][:, np.newaxis]
    return Xarray


def get_bounding_box(coords):

    # Assume coords are always in 2D
    xmin = np.min(coords[0, :])
    xmax = np.max(coords[0, :])

    ymin = np.min(coords[1, :])
    ymax = np.max(coords[1, :])

    width = xmax - xmin
    height = ymax - ymin
    deltax = width/coords.shape[1]
    deltay = height/coords.shape[1]

    mean_x_spacing = np.mean(np.diff(np.sort(coords[0, :])))
    mean_y_spacing = np.mean(np.diff(np.sort(coords[1, :])))

    return {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax,
            'width': width, 'height': height, 'deltax': deltax, 'deltay': deltay,
            'mean_x_spacing': mean_x_spacing, 'mean_y_spacing': mean_y_spacing
            }


def add_shapes_to_fig(fig, Xarray, newpos):
    k, _, _ = Xarray.shape
    bbox = get_bounding_box(newpos)
    Xarraynew = shift_center_and_scale(Xarray, newpos, bbox['mean_x_spacing']*15)
    for ii in range(k):
        fig.add_trace(go.Scatter(x=Xarraynew[ii][0,], y=Xarraynew[ii][1,], mode='lines',
                                 line=dict(color='blue', width = 0.5)))


# TODO the below functionality could be merged with plot_MDS
def plot_tpca(covdata, labelsdf):

    # fig = go.Figure(data=go.Scatter(x=pos[0,:], y=pos[1,:]))
    labelsdf['X'] = covdata['Eigproj'][:, 0]
    labelsdf['Y'] = covdata['Eigproj'][:, 1]
    labelsdf['Z'] = covdata['Eigproj'][:, 2]

    # Rescale the scale column to lie between 0 and 1
    labelsdf['scale'] = labelsdf['scale']/max(labelsdf['scale'])

    fig = px.scatter(labelsdf, x="X", y="Y", color="label",
                     size='scale', hover_data=['name'], text="name")
    fig = set_generic_fig_properties(fig, title_text='Principal Component Analysis (PCA)', showlegend=True)
    fig.show()
    fig.write_image("pcaplot_X_Y.pdf")

    fig = px.scatter(labelsdf, x="X", y="Z", color="label",
                     size='scale', hover_data=['name'], text="name")
    fig = set_generic_fig_properties(fig, title_text='Principal Component Analysis (PCA)', showlegend=True)
    fig.show()
    fig.write_image("pcaplot_X_Z.pdf")

    fig = px.scatter(labelsdf, x="Y", y="Z", color="label",
                     size='scale', hover_data=['name'], text="name")
    fig = set_generic_fig_properties(fig, title_text='Principal Component Analysis (PCA)', showlegend=True)
    fig.show()
    fig.write_image("pcaplot_Y_Z.pdf")
    labelsdf.to_csv('PCAscores.csv', float_format="%g")

# TODO This functionality could be merged with above
def plot_shapes_on_tpca(Xarray, covdata, labelsdf):

    k, n, T = Xarray.shape

    # fig = go.Figure(data=go.Scatter(x=pos[0,:], y=pos[1,:]))
    labelsdf['X'] = covdata['Eigproj'][:, 0]
    labelsdf['Y'] = covdata['Eigproj'][:, 1]
    labelsdf['Z'] = covdata['Eigproj'][:, 2]

    # Rescale the scale column to lie between 0 and 1
    labelsdf['scale'] = labelsdf['scale']/max(labelsdf['scale'])

    fig = px.scatter(labelsdf, x="X", y="Y", color="label",
                     size='scale', hover_data=['name'], text="name")
    newpos = np.vstack((labelsdf['X'], labelsdf['Y']))
    add_shapes_to_fig(fig, Xarray, newpos)
    fig = set_generic_fig_properties(fig, title_text='Principal Component Analysis (PCA)', showlegend=True)
    fig.show()
    fig.write_image("pcaplot_shape_X_Y.pdf")

    fig = px.scatter(labelsdf, x="X", y="Z", color="label",
                     size='scale', hover_data=['name'], text="name")
    newpos = np.vstack((labelsdf['X'], labelsdf['Z']))
    add_shapes_to_fig(fig, Xarray, newpos)
    fig = set_generic_fig_properties(fig, title_text='Principal Component Analysis (PCA)', showlegend=True)
    fig.show()
    fig.write_image("pcaplot_shape_X_Z.pdf")

    fig = px.scatter(labelsdf, x="Y", y="Z", color="label",
                     size='scale', hover_data=['name'], text="name")
    newpos = np.vstack((labelsdf['Y'], labelsdf['Z']))
    add_shapes_to_fig(fig, Xarray, newpos)
    fig = set_generic_fig_properties(fig, title_text='Principal Component Analysis (PCA)', showlegend=True)
    fig.show()
    fig.write_image("pcaplot_shape_Y_Z.pdf")


def plot_pca_scree(covdata):

    # Plot variance
    PClabels = ["PC" + str(i) for i in range(1, 11)]
    df = {'Components': PClabels, '% Explained Variance': covdata['S'][range(10)] / np.sum(covdata['S']) * 100,
          '% Cumulative Explained Variance': np.cumsum(covdata['S'][range(10)]) / np.sum(covdata['S']) * 100}
    vardf = pd.DataFrame(data=df)
    # fig.add_trace(go.Bar(data = vardf, x='Components', y='% Explained Variance'))

    fig = go.Figure(go.Bar(x=vardf['Components'], y=vardf['% Explained Variance'], name="Variance"))
    fig.add_trace(go.Scatter(x=vardf['Components'], y=vardf['% Cumulative Explained Variance'], name="Cumul. Variance"))
    fig.update_layout(autosize=True, title_text='PCA Scree plot', showlegend=True)

    fig.show()
    fig.write_image("pca_screeplot.pdf")
