from morphpy.curveio import readsvgcurvelist
import sys
from pandas import read_csv
from pysrvf.main_mean import get_data_mean, get_deformation_field_from_tangent_vectors
from pysrvf.generic_utils import batch_curve_to_q, q_to_curve
from pysrvf.compute_geodesic import geodesic_distance_all
from pysrvf.tpca import tpca_from_data
from .viz import plot_shapes, plot_curve, plot_pairwise_distance_matrix, show_table, \
    plot_deformation_field, plot_dendrogram, plot_MDS, plot_shapes_on_MDS, plot_tpca, plot_shapes_on_tpca


class Morphpy(object):
    def __init__(self, filelist="", csv=""):
        if filelist and csv:
            self.load(filelist=filelist, csv=csv)

    def load(self, filelist="", csv="", T=100):

        if filelist == "" or csv == "":
            sys.stdout.write("Please provide a valid filelist and a csv file\n")
            return
        try:
            self.Xarray = readsvgcurvelist(filelist, T=T)
            self.taxoncodes = read_csv(csv)
        except (FileNotFoundError, OSError, Exception) as excp:
            sys.stdout.write(str(excp) + '\n')
            sys.exit(1)

    def show(self):
        show_table(self.taxoncodes)
        plot_shapes(self.Xarray)

    def analyze(self):
        self.qarray, self.is_closed = batch_curve_to_q(self.Xarray)
        self.alpha_arr, self.alpha_t_arr, A_norm_iter_arr, E_geo_C_arr, self.gamma_arr, \
            self.geo_dist_arr = geodesic_distance_all(self.qarray, 'all', self.is_closed)

        self.covdata = tpca_from_data(self.Xarray)
        self.pmean = q_to_curve(self.covdata['qmean'])
        self.mag_def_field = get_deformation_field_from_tangent_vectors(self.covdata['alpha_t_array'])
        self.eigproj = self.covdata['Eigproj'][:, 3]

    def save(self):
        plot_curve(self.pmean, 'Mean Shape')
        plot_pairwise_distance_matrix(self.geo_dist_arr)
        plot_deformation_field(self.pmean, self.mag_def_field)
        plot_dendrogram(self.geo_dist_arr, self.taxoncodes)
        plot_MDS(self.geo_dist_arr, self.taxoncodes)
        plot_shapes_on_MDS(self.Xarray, self.geo_dist_arr, self.taxoncodes)
        plot_tpca(self.covdata, self.taxoncodes)
        plot_shapes_on_tpca(self.Xarray, self.covdata, self.taxoncodes)
