from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import joblib
from tqdm import tqdm
import numpy as np
from utils import log_info
from wass_dpc import load_data
import os


datapath="./data/"
knn_path=datapath+"knn_tmp/"
wass_kinds=["in","out","out_out"]#["in",]
rebuild_knn=True
alg_name="WassDPC"
max_min_scaler = MinMaxScaler()
data_names=["R15","S2","Flame","banana-ball","seeds","banknote","segmentation","phoneme","MFCCs","mnist_784"]
requirement=1000
# data_names = []
# for tmp_name in os.listdir("./data/"):
#     if tmp_name[-4:]==".csv":
#         data_names.append(tmp_name)

for data_name in tqdm(data_names,bar_format='{l_bar}{bar:10}{r_bar}'):
    x_mat, label_true, n_clusters=load_data(datapath,data_name)
    x_mat=MinMaxScaler().fit_transform(x_mat)
    max_neigh=int((x_mat.shape[0]/n_clusters)*(50/100))
    nbrs = NearestNeighbors(n_neighbors=max_neigh,algorithm="kd_tree").fit(x_mat)
    joblib.dump(nbrs,knn_path+data_name+".joblib")
    # knn_dists,knn_indices=nbrs.kneighbors(x_mat)
    # np.savez(f"./data/knn_tmp/knn_dists_indices_({data_name}_{max_neigh}).npz",knn_dists=knn_dists,knn_indices=knn_indices)



