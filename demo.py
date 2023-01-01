from wass_dpc import WassDPC,load_data
from FastDEC import FastDEC
from QuickDSC import QuickDSC
from QuickshiftPP import QuickshiftPP
from SNNDPC import SNNDPC
from finch import FINCH
from quickshift_fromNick_Ol import QuickShift
from sklearn.cluster import DBSCAN,MeanShift
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
import utils,time

datapath="./data/"
knn_path=datapath+"knn_tmp/"
wass_kinds=["in","out","out_out"]#["out_out"]
alg_names=["WassDPC","FastDEC","QuickDSC","QuickshiftPP","SNNDPC","FINCH","QuickShift","DBSCAN","MeanShift"]
data_names=["R15","S2","Flame","banana-ball","seeds","banknote","segmentation","phoneme","MFCCs","mnist_784"]
log_path_base="./data/others/"

def result_analysis(label_true,pred_labels,alg_name,data_name,shape,k_neighbors,n_clusters,time_cost,log_path):
    nmi=NMI(label_true,pred_labels)
    ari=ARI(label_true,pred_labels)
    ami=AMI(label_true,pred_labels)
    utils.log_info(
        f"{data_name} datasets process: alg_name={alg_name}, shape={shape}, k_neighbors={k_neighbors}, n_clusters={n_clusters} nmi={nmi:.4f}, ari={ari:.4f}, ami={ami:.4f}, time_cost={time_cost:.4f}s",
        log_path=log_path
    )

for data_name in ["mnist_784"]:# data_names
    x_mat, label_true, n_clusters=load_data(datapath,data_name)
    x_mat=MinMaxScaler().fit_transform(x_mat)
    max_neigh=int((x_mat.shape[0]/n_clusters)*(50/100))
    knn_info={
        "search_index":knn_path+data_name+".joblib",
        "dists_indices":knn_path+f"knn_dists_indices_({data_name}_{max_neigh}).npz",
        "max_neigh":max_neigh
        }
    for alg_name in ["WassDPC"]:#["WassDPC"]alg_names
        k=int((x_mat.shape[0]/n_clusters)*0.4)
        if alg_name=="WassDPC":
            for wass_kind in wass_kinds:
                t0=time.perf_counter()
                wass_dpc=WassDPC(k,n_clusters,knn_info=knn_info,wass_kind=wass_kind)
                pred_labels,centers=wass_dpc.fit(x_mat)
                t1=time.perf_counter()
                result_analysis(
                    label_true=label_true,
                    pred_labels=pred_labels,
                    alg_name=alg_name+f"({wass_kind})",
                    data_name=data_name,
                    shape=x_mat.shape,
                    k_neighbors=k,
                    n_clusters=n_clusters,
                    time_cost=t1-t0,
                    log_path=log_path_base+alg_name+"_log.txt"
                )
        elif alg_name=="FastDEC":
            t0=time.perf_counter()
            fastdec=FastDEC(k,n_clusters,knn_info=knn_info)
            pred_labels=fastdec.fit(x_mat)
            t1=time.perf_counter()
            result_analysis(
                    label_true=label_true,
                    pred_labels=pred_labels,
                    alg_name=alg_name,
                    data_name=data_name,
                    shape=x_mat.shape,
                    k_neighbors=k,
                    n_clusters=n_clusters,
                    time_cost=t1-t0,
                    log_path=log_path_base+alg_name+"_log.txt"
                )
        elif alg_name=="DBSCAN":
            eps=5
            min_samples=3
            t0=time.perf_counter()
            dbscan=DBSCAN(eps=eps,min_samples=min_samples).fit(x_mat)
            pred_labels=dbscan.labels_
            t1=time.perf_counter()
            result_analysis(
                    label_true=label_true,
                    pred_labels=pred_labels,
                    alg_name=alg_name,
                    data_name=data_name,
                    shape=x_mat.shape,
                    k_neighbors=k,
                    n_clusters=n_clusters,
                    time_cost=t1-t0,
                    log_path=log_path_base+alg_name+"_log.txt"
                )
        elif alg_name=="MeanShift":
            t0=time.perf_counter()
            meanshift=MeanShift().fit(x_mat)
            pred_labels=meanshift.labels_
            t1=time.perf_counter()
            result_analysis(
                    label_true=label_true,
                    pred_labels=pred_labels,
                    alg_name=alg_name,
                    data_name=data_name,
                    shape=x_mat.shape,
                    k_neighbors=k,
                    n_clusters=n_clusters,
                    time_cost=t1-t0,
                    log_path=log_path_base+alg_name+"_log.txt"
                )
        elif alg_name=="QuickDSC":
            t0=time.perf_counter()
            quickdsc=QuickDSC(k,n_clusters,beta=0.9)
            quickdsc.fit(x_mat)
            pred_labels=quickdsc.labels_
            t1=time.perf_counter()
            result_analysis(
                    label_true=label_true,
                    pred_labels=pred_labels,
                    alg_name=alg_name,
                    data_name=data_name,
                    shape=x_mat.shape,
                    k_neighbors=k,
                    n_clusters=n_clusters,
                    time_cost=t1-t0,
                    log_path=log_path_base+alg_name+"_log.txt"
                )
        elif alg_name=="QuickShift":
            t0=time.perf_counter()
            quickshift=QuickShift()
            quickshift.fit(x_mat)
            pred_labels=quickshift.labels_.astype(int)
            t1=time.perf_counter()
            result_analysis(
                    label_true=label_true,
                    pred_labels=pred_labels,
                    alg_name=alg_name,
                    data_name=data_name,
                    shape=x_mat.shape,
                    k_neighbors=k,
                    n_clusters=n_clusters,
                    time_cost=t1-t0,
                    log_path=log_path_base+alg_name+"_log.txt"
                )
        elif alg_name=="QuickshiftPP":
            t0=time.perf_counter()
            quickshiftpp = QuickshiftPP(k=k, beta=.3, epsilon=0, ann="kdtree")
            quickshiftpp.fit(x_mat)
            pred_labels = quickshiftpp.memberships.astype(int)
            t1=time.perf_counter()
            result_analysis(
                    label_true=label_true,
                    pred_labels=pred_labels,
                    alg_name=alg_name,
                    data_name=data_name,
                    shape=x_mat.shape,
                    k_neighbors=k,
                    n_clusters=n_clusters,
                    time_cost=t1-t0,
                    log_path=log_path_base+alg_name+"_log.txt"
                )
        elif alg_name=="SNNDPC":
            t0=time.perf_counter()
            centroid, pred_labels=SNNDPC(k=k,nc=n_clusters,data=x_mat)
            t1=time.perf_counter()
            result_analysis(
                    label_true=label_true,
                    pred_labels=pred_labels,
                    alg_name=alg_name,
                    data_name=data_name,
                    shape=x_mat.shape,
                    k_neighbors=k,
                    n_clusters=n_clusters,
                    time_cost=t1-t0,
                    log_path=log_path_base+alg_name+"_log.txt"
                )
        elif alg_name=="FINCH":
            t0=time.perf_counter()
            c, num_clust, req_c = FINCH(x_mat, req_clust=n_clusters,use_ann_above_samples=k,verbose=False,distance='euclidean')
            pred_labels=req_c
            t1=time.perf_counter()
            result_analysis(
                    label_true=label_true,
                    pred_labels=pred_labels,
                    alg_name=alg_name,
                    data_name=data_name,
                    shape=x_mat.shape,
                    k_neighbors=k,
                    n_clusters=n_clusters,
                    time_cost=t1-t0,
                    log_path=log_path_base+alg_name+"_log.txt"
                )
