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


def plot_shapes(Xarray):
    N, n, T = np.shape(Xarray)
    # n has to be equal to 2

    fig = make_subplots(rows=int(np.ceil(N**0.5)), cols=int(np.ceil(N ** 0.5)))
    rows = int(np.ceil(N**0.5))
    cols = int(np.ceil(N**0.5))

    for ii in range(N):
        ix, iy = np.unravel_index(ii, (rows, cols))
        fig.add_trace(go.Scatter(x=Xarray[ii][0,], y=Xarray[ii][1,], mode='lines'), row=ix.tolist()+1, col=iy.tolist()+1)

    fig = set_generic_fig_properties(fig, title_text="All shapes")
    fig.show()
    fig.write_image("allshapes.pdf")


def set_generic_fig_properties(fig, height=600, width=600, title_text="", showticks=False, showlegend=False):
    fig.update_layout(autosize=False, height=height, width=width, title_text=title_text,
                      yaxis=dict(scaleanchor="x", scaleratio=1), showlegend=showlegend,
                      margin=dict(r=5, l=5, t=25, b=5), scene=dict(aspectmode="data"))
    fig.update_xaxes(showticklabels=showticks)
    fig.update_yaxes(showticklabels=showticks, autorange="reversed")
    return fig
    # layout = dict(plot_bgcolor='white', margin=dict(t=0, b=0, r=0, l=0, pad=0),
    #               xaxis=dict(showgrid=False, zeroline=False, mirror=True, linecolor='gray'),
    #               yaxis=dict(showgrid=False, zeroline=False, mirror=True, linecolor='gray'))



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


def plot_shapes_on_MDS(Dvect, labelsdf):

    seed = np.random.RandomState(seed=3)
    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
                       dissimilarity="precomputed", n_jobs=1)
    pos = mds.fit(sp.squareform(Dvect)).embedding_

    # fig = go.Figure(data=go.Scatter(x=pos[0,:], y=pos[1,:]))
    labelsdf['X'] = pos[:, 0]
    labelsdf['Y'] = pos[:, 1]
    # Rescale the scale column to lie between 0 and 1
    labelsdf['scale'] = labelsdf['scale']/max(labelsdf['scale'])

    fig = px.scatter(labelsdf, x="X", y="Y", color="label",
                     size='scale', hover_data=['name'], text="name")

    fig = set_generic_fig_properties(fig, title_text='Multidimensional Scaling', showlegend=True)
    fig.show()
    fig.write_image("mdsplot.pdf")


def center_and_rescale_curves(Xarray):
    k, n, T = Xarray.shape

    # Center and Scale
    for ii in range(k):
        Xarray[ii, :, :] - np.mean(Xarray[ii, :, :], axis=1).reshape((n, 1))
        Xarray[ii, :, :]/= np.linalg.norm(Xarray[ii, :, :], 'fro')

    return Xarray