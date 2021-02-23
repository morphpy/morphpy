from pysrvf.tpca import tpca_from_data



def get_tpca(Xarray):
    covdata = tpca_from_data(Xarray)
    
    return covdata