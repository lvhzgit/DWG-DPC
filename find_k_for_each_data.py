from wass_dpc import WassDPC,load_data
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
import utils,time
from tqdm import tqdm


datapath="./data/"
knn_path=datapath+"knn_tmp/"
log_path_base="./data/others/"
data_names=["R15","S2","Flame","banana-ball","seeds","banknote","segmentation","phoneme","MFCCs","mnist_784"]

def result_analysis(label_true,pred_labels,alg_name,data_name,shape,k_neighbors,n_clusters,time_cost,log_path):
    nmi=NMI(label_true,pred_labels)
    ari=ARI(label_true,pred_labels)
    ami=AMI(label_true,pred_labels)
    utils.log_info(
        f"{data_name} datasets process: alg_name={alg_name}, shape={shape}, k_neighbors={k_neighbors}, n_clusters={n_clusters}, nmi={nmi:.4f}, ari={ari:.4f}, ami={ami:.4f}, time_cost={time_cost:.4f}s",
        log_path=log_path
    )
    return nmi

# ########################## for WassDPC (k) #####################################################
# wass_kinds=["in","out","out_out"]#["in",]
# alg_name="WassDPC"
# for data_name in ["R15","S2","Flame","banana-ball","seeds","banknote","segmentation","phoneme","MFCCs"]:
#     x_mat, label_true, n_clusters=load_data(datapath,data_name)
#     x_mat=MinMaxScaler().fit_transform(x_mat)
#     max_neigh=int((x_mat.shape[0]/n_clusters)*(50/100))
#     knn_info={
#         "search_index":knn_path+data_name+".joblib",
#         "dists_indices":knn_path+f"knn_dists_indices_({data_name}_{max_neigh}).npz",
#         "max_neigh":max_neigh
#         }
#     end_flag=int((x_mat.shape[0]//n_clusters)*1.5)
#     for wass_kind in wass_kinds:
#         # k=32
#         for k in tqdm(range(2,end_flag),bar_format='{l_bar}{bar:10}{r_bar}'):
#             t0=time.perf_counter()
#             try:
#                 wass_dpc=WassDPC(k,n_clusters,knn_info=knn_info,wass_kind=wass_kind)
#                 pred_labels,centers=wass_dpc.fit(x_mat)
#             except Exception as e:
#                 continue
#             t1=time.perf_counter()
#             result_analysis(
#                 label_true=label_true,
#                 pred_labels=pred_labels,
#                 alg_name=alg_name+f"({wass_kind})",
#                 data_name=data_name,
#                 shape=x_mat.shape,
#                 k_neighbors=k,
#                 n_clusters=n_clusters,
#                 time_cost=t1-t0,
#                 log_path=log_path_base+alg_name+"_log.txt"
#             )


# ######################## for FastDEC (k) ####################################################
# from FastDEC import FastDEC
# alg_name="FastDEC"
# for data_name in ["R15","S2","Flame","banana-ball","seeds","banknote","segmentation","phoneme","MFCCs"]:
#     x_mat, label_true, n_clusters=load_data(datapath,data_name)
#     x_mat=MinMaxScaler().fit_transform(x_mat)
#     max_neigh=int((x_mat.shape[0]/n_clusters)*(50/100))
#     knn_info={
#         "search_index":knn_path+data_name+".joblib",
#         "dists_indices":knn_path+f"knn_dists_indices_({data_name}_{max_neigh}).npz",
#         "max_neigh":max_neigh
#         }
#     end_flag=int((x_mat.shape[0]//n_clusters)*1.5)
#     for k in tqdm(range(2,end_flag),bar_format='{l_bar}{bar:10}{r_bar}'):
#         t0=time.perf_counter()
#         try:
#             fastdec=FastDEC(k,n_clusters,knn_info=knn_info)
#             pred_labels=fastdec.fit(x_mat)
#         except Exception as e:
#             continue
#         t1=time.perf_counter()
#         result_analysis(
#                 label_true=label_true,
#                 pred_labels=pred_labels,
#                 alg_name=alg_name,
#                 data_name=data_name,
#                 shape=x_mat.shape,
#                 k_neighbors=k,
#                 n_clusters=n_clusters,
#                 time_cost=t1-t0,
#                 log_path=log_path_base+alg_name+f"_log.txt"
#             )


# ################### QuickDSC (k)????? ####################################################################
# from QuickDSC import QuickDSC

# for data_name in ["phoneme","MFCCs","mnist_784"]:
#     alg_name="QuickDSC"
#     x_mat, label_true, n_clusters=load_data(datapath,data_name)
#     x_mat=MinMaxScaler().fit_transform(x_mat)
#     end_flag=int((x_mat.shape[0]//n_clusters)*1.5)

#     for k in tqdm(range(2,end_flag),desc=data_name,bar_format='{l_bar}{bar:10}{r_bar}'):  # 2-end_flag
#         t0=time.perf_counter()
#         try:
#             quickdsc=QuickDSC(k,n_clusters,beta=0.9)
#             quickdsc.fit(x_mat)
#             t1=time.perf_counter()
#         except Exception as e:
#             t1=time.perf_counter()
#             continue
#         pred_labels=quickdsc.labels_
#         result_analysis(
#                 label_true=label_true,
#                 pred_labels=pred_labels,
#                 alg_name=alg_name,
#                 data_name=data_name,
#                 shape=x_mat.shape,
#                 k_neighbors=k,
#                 n_clusters=n_clusters,
#                 time_cost=t1-t0,
#                 log_path=log_path_base+alg_name+"_log.txt"
#             )


# #################### QuickshiftPP (k) ##################################################################
# from QuickshiftPP import QuickshiftPP

# for data_name in data_names:
#     alg_name="QuickshiftPP"
#     x_mat, label_true, n_clusters=load_data(datapath,data_name)
#     x_mat=MinMaxScaler().fit_transform(x_mat)
#     end_flag=int((x_mat.shape[0]//n_clusters)*1.5)
#     for k in tqdm(range(2,end_flag),bar_format='{l_bar}{bar:10}{r_bar}'):
#         t0=time.perf_counter()
#         try:
#             quickshiftpp = QuickshiftPP(k=k, beta=.3, epsilon=0, ann="kdtree")
#             quickshiftpp.fit(x_mat)
#         except Exception as e:
#             continue
#         pred_labels = quickshiftpp.memberships.astype(int)
#         t1=time.perf_counter()
#         result_analysis(
#                 label_true=label_true,
#                 pred_labels=pred_labels,
#                 alg_name=alg_name,
#                 data_name=data_name,
#                 shape=x_mat.shape,
#                 k_neighbors=k,
#                 n_clusters=n_clusters,
#                 time_cost=t1-t0,
#                 log_path=log_path_base+alg_name+"_log.txt"
#             )


# ######################### SNNDPC (k) ######################################################################
# from SNNDPC import SNNDPC

# for data_name in data_names:
#     alg_name="SNNDPC"
#     x_mat, label_true, n_clusters=load_data(datapath,data_name)
#     x_mat=MinMaxScaler().fit_transform(x_mat)
#     end_flag=int((x_mat.shape[0]//n_clusters)*1.5)
#     for k in tqdm(range(2,end_flag),bar_format='{l_bar}{bar:10}{r_bar}'):
#         t0=time.perf_counter()
#         try:
#             centroid, pred_labels=SNNDPC(k=k,nc=n_clusters,data=x_mat)
#         except Exception as e:
#             continue
#         t1=time.perf_counter()
#         result_analysis(
#                 label_true=label_true,
#                 pred_labels=pred_labels,
#                 alg_name=alg_name,
#                 data_name=data_name,
#                 shape=x_mat.shape,
#                 k_neighbors=k,
#                 n_clusters=n_clusters,
#                 time_cost=t1-t0,
#                 log_path=log_path_base+alg_name+"_log.txt"
#             )



# ########################## FINCH (k) #############################################################
# from finch import FINCH

# for data_name in data_names:
#     alg_name="FINCH"
#     x_mat, label_true, n_clusters=load_data(datapath,data_name)
#     x_mat=MinMaxScaler().fit_transform(x_mat)
#     end_flag=int((x_mat.shape[0]//n_clusters)*1.5)
#     for k in tqdm(range(2,end_flag),bar_format='{l_bar}{bar:10}{r_bar}'):
#         t0=time.perf_counter()
#         try:
#             c, num_clust, req_c = FINCH(x_mat, req_clust=n_clusters,use_ann_above_samples=k,verbose=False,distance='euclidean')
#         except Exception as e:
#             continue
#         pred_labels=req_c
#         t1=time.perf_counter()
#         result_analysis(
#                 label_true=label_true,
#                 pred_labels=pred_labels,
#                 alg_name=alg_name,
#                 data_name=data_name,
#                 shape=x_mat.shape,
#                 k_neighbors=k,
#                 n_clusters=n_clusters,
#                 time_cost=t1-t0,
#                 log_path=log_path_base+alg_name+"_log.txt"
#             )


# ############################### QuickShift ??? ###################################################
# from quickshift_fromNick_Ol import QuickShift

# for data_name in data_names:
#     alg_name="QuickShift"
#     x_mat, label_true, n_clusters=load_data(datapath,data_name)
#     x_mat=MinMaxScaler().fit_transform(x_mat)
#     end_flag=int((x_mat.shape[0]//n_clusters)*1.5)
#     t0=time.perf_counter()

#     for bandwidth_i in tqdm(range(1,20),bar_format='{l_bar}{bar:10}{r_bar}'):
#         bandwidth=bandwidth_i/100
#         for tau_i in range(5,200,5):
#             tau=tau_i/100
#             try:
#                 quickshift=QuickShift(tau=tau,bandwidth=bandwidth)
#                 quickshift.fit(x_mat)
#             except Exception as e:
#                 continue
#             pred_labels=quickshift.labels_.astype(int)
#             if NMI(label_true, pred_labels)<0.01:
#                 continue
#             t1=time.perf_counter()
#             result_analysis(
#                     label_true=label_true,
#                     pred_labels=pred_labels,
#                     alg_name=alg_name,
#                     data_name=data_name,
#                     shape=x_mat.shape,
#                     k_neighbors=None,
#                     n_clusters=n_clusters,
#                     time_cost=t1-t0,
#                     log_path=log_path_base+alg_name+"_log.txt"
#                 )

# ##################### DBSCAN (eps,min_samples)#####################################################
# from sklearn.cluster import DBSCAN

# for data_name in data_names:
#     alg_name="DBSCAN"
#     x_mat, label_true, n_clusters=load_data(datapath,data_name)
#     x_mat=MinMaxScaler().fit_transform(x_mat)
#     for eps_i in tqdm(range(1,51),bar_format='{l_bar}{bar:10}{r_bar}'):
#         eps=eps_i/10
#         for min_samples in range(2,10):
#             t0=time.perf_counter()
#             try:
#                 dbscan=DBSCAN(eps=eps,min_samples=min_samples).fit(x_mat)
#             except Exception as e:
#                 continue
#             pred_labels=dbscan.labels_
#             t1=time.perf_counter()
#             if NMI(label_true, pred_labels)<0.01:
#                 continue
#             result_analysis(
#                     label_true=label_true,
#                     pred_labels=pred_labels,
#                     alg_name=alg_name,
#                     data_name=data_name,
#                     shape=x_mat.shape,
#                     k_neighbors=f"(eps:{eps},min_samples:{min_samples})",
#                     n_clusters=n_clusters,
#                     time_cost=t1-t0,
#                     log_path=log_path_base+alg_name+"_log.txt"
#                 )


# ############################### MeanShift #####################################################
# from sklearn.cluster import MeanShift

# for data_name in data_names:
#     alg_name="MeanShift"
#     x_mat, label_true, n_clusters=load_data(datapath,data_name)
#     x_mat=MinMaxScaler().fit_transform(x_mat)
#     t0=time.perf_counter()
#     for bandwidth_i in tqdm(range(1,200),bar_format='{l_bar}{bar:10}{r_bar}'):
#         bandwidth=bandwidth_i/10
#         try:
#             meanshift=MeanShift(bandwidth=bandwidth).fit(x_mat)
#         except Exception as e:
#             continue
#         pred_labels=meanshift.labels_
#         t1=time.perf_counter()
#         if NMI(label_true, pred_labels)<0.01:
#             continue
#         result_analysis(
#                 label_true=label_true,
#                 pred_labels=pred_labels,
#                 alg_name=alg_name,
#                 data_name=data_name,
#                 shape=x_mat.shape,
#                 k_neighbors=f"(bandwidth={bandwidth})",
#                 n_clusters=n_clusters,
#                 time_cost=t1-t0,
#                 log_path=log_path_base+alg_name+"_log.txt"
#             )

# ############################### for WassDPC percentage k (k) ###########################################
# data_names=[
#     ["banknote","MFCCs"],
#     ["phoneme","banana-ball","seeds",],
#     ["segmentation","S2","Flame"],
#     ["mnist_784"]
# ]
# wass_kinds=["in","out","out_out"]#["in",]
# alg_name="WassDPC"
# flag=2 #[0,1,2,3]

# for data_name in tqdm(data_names[flag],bar_format='{l_bar}{bar:10}{r_bar}'):
#     x_mat, label_true, n_clusters=load_data(datapath,data_name)
#     x_mat=MinMaxScaler().fit_transform(x_mat)
#     max_neigh=int((x_mat.shape[0]/n_clusters)*(50/100))
#     knn_info={
#         "search_index":knn_path+data_name+".joblib",
#         "dists_indices":knn_path+f"knn_dists_indices_({data_name}_{max_neigh}).npz",
#         "max_neigh":max_neigh
#         }
#     for wass_kind in wass_kinds:
#         history_nmi=list()
#         k_old=0
#         for p in range(1,51):#1 133,190
#             k=int((x_mat.shape[0]/n_clusters)*(p/100))
#             if k < 2 or k-k_old<1:
#                 continue
#             k_old=k
#             t0=time.perf_counter()
#             wass_dpc=WassDPC(k,n_clusters,knn_info=knn_info,wass_kind=wass_kind)
#             try:
#                 pred_labels,centers=wass_dpc.fit(x_mat,knn_path,data_name)
#             except Exception as e:
#                 continue
#             t1=time.perf_counter()
#             nmi=result_analysis(
#                 label_true=label_true,
#                 pred_labels=pred_labels,
#                 alg_name=alg_name+f"({wass_kind})",
#                 data_name=data_name,
#                 shape=x_mat.shape,
#                 k_neighbors=k,
#                 n_clusters=n_clusters,
#                 time_cost=t1-t0,
#                 log_path=log_path_base+alg_name+f"_log_{flag}.txt"
#             )
#             if len(history_nmi)<20:
#                 history_nmi.append(nmi)
#             else:
#                 history_nmi.pop(0)
#                 history_nmi.append(nmi)
#             max_nmi,min_nmi=max(history_nmi),min(history_nmi)
#             if len(history_nmi)>=20 and max_nmi-min_nmi<1e-6:
#                 break
#             if nmi>=1:
#                 break