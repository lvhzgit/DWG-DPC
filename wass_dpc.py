import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import normalized_mutual_info_score
import joblib,time
import numba

def load_data(datapath,data_name):
    """
        load source datasets without minmax scale
    """
    data = np.loadtxt(datapath+data_name+".csv", delimiter=",")
    x_mat = data[:,:-1]
    label_true = data[:,-1].flatten().astype(np.int8)
    n_clusters = len(set(label_true))
    return x_mat, label_true, n_clusters

class WassDPC:
    """
    Parameters
    ----------
    k: The number of neighbors (i.e. the k in k-NN density)\n
    n_clusters: number of clustering result\n
    ann: {'ball_tree', 'kd_tree'}, Approximate Nearest Neighbor Algorithms\n
    denstiy_estimator: {"gaussion","kth","flat"}, Kernel Density Estimation Algorithms\n
    knn_info: knn_info dict, stored knn path, default None, None is to rebuild knn\n
    space index and not None is directly using knn_info,
        knn_info dict definition:
        knn_info={
            "search_index":knn_path+data_name+".joblib",
            "dists_indices":knn_path+f"knn_dists_indices_({data_name}_{max_neigh}).npz",
            "max_neigh":max_neigh
            }
        \n
    wass_kind: {"in", "out", "out_out"}, approaches to calculate wass_dist
    """
    def __init__(self, k,n_clusters,denstiy_estimator="gaussion",ann="kd_tree",knn_info=None,wass_kind="out_out"):
        self.k = k
        self.n_clusters = n_clusters
        self.density_estimator = denstiy_estimator
        self.ann = ann
        self.knn_info=knn_info
        self.wass_kind=wass_kind
        self.max_neigh=0

    def knn_relationship(self, x_mat):
        if self.knn_info is None:
            self.max_neigh=int((x_mat.shape[0]/n_clusters)*(50/100))
            nbrs = NearestNeighbors(n_neighbors=self.max_neigh,algorithm=self.ann).fit(x_mat)            
            knn_dists,knn_indices=nbrs.kneighbors(x_mat)
        else:
            self.max_neigh=self.knn_info["max_neigh"]
            nbrs=joblib.load(self.knn_info["search_index"])
            knn_dists_indices=np.load(self.knn_info["dists_indices"])
            knn_dists,knn_indices=knn_dists_indices["knn_dists"],knn_dists_indices["knn_indices"]
        return nbrs,knn_dists[:,1:],knn_indices[:,1:]
    
    @staticmethod
    # @numba.njit()
    def density_estimation(density_estimator,knn_dists,k):
        density = np.full(knn_dists.shape[0], 0, np.float64)
        if density_estimator == 'gaussion':
            mu = np.mean(knn_dists)
            density = np.sum(np.exp(-knn_dists / 2 / mu), axis=1)
        if density_estimator == 'kth':
            density = 1/knn_dists[:, k - 1]
        if density_estimator == 'flat':
            mu = np.mean(knn_dists)
            for i in range(knn_dists.shape[0]):
                idx=  np.where(knn_dists[i] < mu)[0]
                density[i] = len(idx)
        return density

    def find_DCs(self,x_mat,knn_dists,knn_indices):
        """
            return values: core_idx,DC_member_count,all_DC_ids. Purpose for DC_member_count\
                 is to speedup DC_Graph construct 
        """
        n_neighbors, new_knn_indices, new_knn_dists=self.k, knn_indices[:,:self.k], knn_dists[:,:self.k]
        density = self.density_estimation(self.density_estimator,new_knn_dists,self.k)
        subordinate = np.full(x_mat.shape[0], -1, dtype=np.int32)
        for i in range(x_mat.shape[0]):
            neighbor_i=new_knn_indices[i]
            for j in range(neighbor_i.shape[0]):
                if density[i] < density[neighbor_i[j]]:
                    subordinate[i] = neighbor_i[j]
                    break
        core_idx = np.where(subordinate == -1)[0]
        if self.n_clusters > len(core_idx):
            while self.n_clusters > len(core_idx):
                n_neighbors = int(n_neighbors*0.7)
                new_knn_indices = new_knn_indices[:,0:n_neighbors]
                new_knn_dists = new_knn_dists[:,0:n_neighbors]
                density = self.density_estimation(self.density_estimator,new_knn_dists,n_neighbors)
                subordinate = np.full(x_mat.shape[0], -1, dtype=np.int32)
                for i in range(x_mat.shape[0]):
                    neighbor_i=new_knn_indices[i]
                    for j in range(neighbor_i.shape[0]):
                        if density[i] < density[neighbor_i[j]]:
                            subordinate[i] = neighbor_i[j]
                            break
                core_idx = np.where(subordinate == -1)[0]
        for i in range(subordinate.shape[0]):
            if i not in core_idx:
                tmp_arr=np.array([],dtype=np.int32)
                tmp_arr=np.append(tmp_arr,i)
                while subordinate[tmp_arr[-1]] not in core_idx:
                    tmp_arr=np.append(tmp_arr,subordinate[tmp_arr[-1]])
                DC_core=subordinate[tmp_arr[-1]]
                subordinate[tmp_arr]=DC_core
        all_DC_ids=subordinate.copy()
        all_DC_ids[core_idx]=core_idx
        DC_member_count=np.zeros(shape=core_idx.shape[0],dtype=np.int32)
        for new_id,old_id in enumerate(core_idx):
            replace_index=np.where(all_DC_ids==old_id)[0]
            all_DC_ids[replace_index]=new_id
            DC_member_count[new_id]+=replace_index.shape[0]
        return core_idx,DC_member_count,all_DC_ids,density

    @staticmethod
    # @numba.njit()
    def DC_graph_edgecount_affinity(n_cores,anchors_DC_flags,new_knn_indices):
        """
            new_knn_indices: new_x_mat knn relationshiip, not x_mat
        """
        edgecount_affinity=np.zeros(shape=(n_cores,n_cores),dtype=np.int32)
        for DC_i in range(n_cores):
            menbership=np.where(anchors_DC_flags==DC_i)[0]
            for new_i in menbership:
                neighbor_i=new_knn_indices[new_i]
                neighbor_subor=anchors_DC_flags[neighbor_i]
                subor_count=np.bincount(neighbor_subor)
                for DC_j in range(subor_count.shape[0]):
                    if DC_i!=DC_j:
                        edgecount_affinity[DC_i,DC_j]+=subor_count[DC_j]
        return edgecount_affinity

    @staticmethod
    # @numba.njit()
    def wass_sim_in(DC_i,DC_j,new_knn_dists,new_knn_indices,DC_i_in,anchors_DC_flags):
        """
            p: the power of denominator. p>=2. To ensure 3**p/3 < 1**p/1, not 3/3==1/1.
        """
        DC_B=np.where(anchors_DC_flags==DC_j)[0]
        d_ba_arr=np.ones(shape=1)*-1
        for j in DC_B:
            neighbors_j=new_knn_indices[j]
            neigh_subor=anchors_DC_flags[neighbors_j]
            A_idx=np.where(neigh_subor==DC_i)[0]
            if len(A_idx)!=0:
                for i in A_idx:
                    if d_ba_arr.shape[0]==1 and d_ba_arr[0]==-1:
                        d_ba_arr[0]=new_knn_dists[j][i]
                    else:
                        d_ba_arr=np.append(d_ba_arr,new_knn_dists[j][i])
            else:
                continue
        mean_value=np.mean(d_ba_arr)
        num_count=d_ba_arr.shape[0]
        if num_count==0:
            return 0
        else:
            wass_sim=(num_count/DC_i_in)*np.sum(np.exp(-d_ba_arr / 2 / mean_value))
            return wass_sim/num_count

    @staticmethod
    # @numba.njit()
    def wass_sim_out(DC_i,DC_j,new_knn_dists,new_knn_indices,DC_i_out,anchors_DC_flags):
        """
            p: the power of denominator. p>=2. To ensure 3**p/3 < 1**p/1, not 3/3==1/1.
        """
        DC_A=np.where(anchors_DC_flags==DC_i)[0]
        d_ab_arr=np.ones(shape=1)*-1
        for i in DC_A:
            neighbors_i=new_knn_indices[i]
            neigh_subor=anchors_DC_flags[neighbors_i]
            B_idx=np.where(neigh_subor==DC_j)[0]
            if len(B_idx)!=0:
                for j in B_idx:
                    if d_ab_arr.shape[0]==1 and d_ab_arr[0]==-1:
                        d_ab_arr[0]=new_knn_dists[j][i]
                    else:
                        d_ab_arr=np.append(d_ab_arr,new_knn_dists[i][j])
            else:
                continue
        mean_value=np.mean(d_ab_arr)
        num_count=d_ab_arr.shape[0]
        if num_count==0:
            return 0
        else:
            wass_sim=(num_count/DC_i_out)*np.sum(np.exp(-d_ab_arr / 2 / mean_value))
            return wass_sim/num_count

    @staticmethod
    # @numba.njit()
    def wass_sim_out_out(DC_i,DC_j,new_knn_dists,new_knn_indices,DC_i_out,DC_j_out,anchors_DC_flags):
        """
            p: the power of denominator. p>=2. To ensure 3**p/3 < 1**p/1, not 3/3==1/1.
        """
        d_ab_arr,d_ba_arr=np.ones(shape=1)*-1,np.ones(shape=1)*-1
        DC_A,DC_B=np.where(anchors_DC_flags==DC_i)[0],np.where(anchors_DC_flags==DC_j)[0]
        for i in DC_A:
            neighbors_i=new_knn_indices[i]
            neigh_subor=anchors_DC_flags[neighbors_i]
            B_idx=np.where(neigh_subor==DC_j)[0]
            if len(B_idx)!=0:
                for j in B_idx:
                    if d_ab_arr.shape[0]==1 and d_ab_arr[0]==-1:
                        d_ab_arr[0]=new_knn_dists[j][i]
                    else:
                        d_ab_arr=np.append(d_ab_arr,new_knn_dists[i][j])
            else:
                continue
        mean_value_ab=np.mean(d_ab_arr)
        num_count_ab=d_ab_arr.shape[0]
        if num_count_ab==0:
            wass_sim_ab=0
        else:
            wass_sim_ab=(num_count_ab/DC_i_out)*np.sum(np.exp(-d_ab_arr / 2 / mean_value_ab))
        for j in DC_B:
            neighbors_j=new_knn_indices[j]
            neigh_subor=anchors_DC_flags[neighbors_j]
            A_idx=np.where(neigh_subor==DC_i)[0]
            if len(A_idx)!=0:
                for i in A_idx:
                    if d_ba_arr.shape[0]==1 and d_ba_arr[0]==-1:
                        d_ba_arr[0]=new_knn_dists[j][i]
                    else:
                        d_ba_arr=np.append(d_ba_arr,new_knn_dists[j][i])
            else:
                continue
        mean_value_ba=np.mean(d_ba_arr)
        num_count_ba=d_ba_arr.shape[0]
        if mean_value_ba==0:
            wass_sim_ba=0
        else:
            wass_sim_ba=(num_count_ba/DC_j_out)*np.sum(np.exp(-d_ba_arr / 2 / mean_value_ba))
        if num_count_ab+num_count_ba==0:
            return 0
        else:
            return (wass_sim_ab/num_count_ab)+(wass_sim_ba/num_count_ba)

    def wasserstein_similarities(self,DC_i,DC_j,knn_dists,knn_indices,edgecount_affinity,anchors_DC_flags):
        # data, indices, indptr=knn_dists.flatten(),knn_indices.flatten(),np.arange(0,(k-1)*x_mat.shape[0]+1,k-1)
        # knn_graph=csr_matrix((data, indices, indptr), shape=(knn_indices.shape[0],knn_indices.shape[0]))       
        if self.wass_kind=="in":
            DC_i_in=edgecount_affinity[:,DC_i].sum()
            wass_sim=self.wass_sim_in(DC_i,DC_j,knn_dists,knn_indices,DC_i_in,anchors_DC_flags)
        elif self.wass_kind=="out":
            DC_i_out=edgecount_affinity[DC_i].sum()
            wass_sim=self.wass_sim_out(DC_i,DC_j,knn_dists,knn_indices,DC_i_out,anchors_DC_flags)
        elif self.wass_kind=="out_out":
            DC_i_out=edgecount_affinity[DC_i].sum()
            DC_j_out=edgecount_affinity[DC_j].sum()
            wass_sim=self.wass_sim_out_out(DC_i,DC_j,knn_dists,knn_indices,DC_i_out,DC_j_out,anchors_DC_flags)
        return wass_sim

    @staticmethod
    @numba.njit()
    def nearest_density_higher_DC(rho,n_cores,affinity):
        sorted_idx=np.argsort(-rho)
        DC_subordinate,delta=np.full(n_cores,-1,np.int32),np.zeros(shape=n_cores)
        for i in range(1,n_cores):
            i_idx, j_idices = sorted_idx[i],sorted_idx[0:i]
            nearest_big_idx=np.argmin(affinity[i_idx][j_idices])
            j_nearest=j_idices[nearest_big_idx]
            delta[i_idx]=affinity[i_idx][j_nearest]
            DC_subordinate[i_idx]=j_nearest
        delta[sorted_idx[0]]=np.max(delta)+1
            # for j in range(0,i):
            #     i_idx, j_idx = sorted_idx[i],sorted_idx[j]
            #     if delta[i_idx]>affinity[i_idx,j_idx]:
            #         delta[i_idx] = affinity[i_idx,j_idx]
            #         DC_subordinate[i_idx] = j_idx
        return delta,DC_subordinate

    @staticmethod
    @numba.njit()
    def kmpp_next_anchor(x_mat,v0,other_vs):
        if other_vs.shape[0]==1:
            u=other_vs[0]
        else:
            tmp_dist=np.empty(shape=other_vs.shape[0])
            for i in range(other_vs.shape[0]):
                vi=other_vs[i]
                tmp_dist[i]=np.sqrt(np.sum((x_mat[v0]-x_mat[vi])**2))
            u_idx=np.argmax(tmp_dist)
            u=other_vs[u_idx]
        return u

    @staticmethod
    # @numba.njit()
    def find_DCs_anchors(x_mat,core_idx,DC_member_count,densities,all_DC_ids,anchors_mnum=30,alpha=0.2,d=2):
        """
            let Out to be outside num, T_num to be Total num. Out=T_num-(T_num**(1/d)-2)**d, T_num>2**d.\
            DC_anchors==-1 represent not anchor.
        """
        tmp_res=all_DC_ids.copy()
        Out_nums=DC_member_count-(DC_member_count**(1/d)-2)**d
        for DC_i in range(core_idx.shape[0]):
            # Out_num=int(DC_member_count[DC_i]-((DC_member_count[DC_i])**(1/d)-2)**d)
            Out_num=int(Out_nums[DC_i])
            source_idx = np.where(all_DC_ids==DC_i)[0]
            tmp_res[source_idx]=-1
            tmp_res[core_idx[DC_i]]=DC_i
            rho_DC_i=densities[source_idx]
            tail_num=int(anchors_mnum*alpha)
            front_num=anchors_mnum-tail_num
            if Out_num>anchors_mnum:
                sorted_idx = np.argsort(rho_DC_i)
                max_rho_idx,min_rho_idx=sorted_idx[-tail_num:],sorted_idx[:Out_num]
                tmp_res[source_idx[max_rho_idx]]=DC_i
                candidate_vs=source_idx[min_rho_idx]
                tmp_vs=np.array([candidate_vs[0]])
                other_vs=np.delete(candidate_vs,0)
                while tmp_vs.shape[0]<front_num: # anchor choose via kmean++ init way.
                    mean_vec = np.sum(x_mat[tmp_vs],0)/tmp_vs.shape[0]
                    if other_vs.shape[0]==1:
                        u_idx=0
                    else:
                        tmp_dist=np.sqrt(np.sum((mean_vec-x_mat[other_vs])**2,1))
                        u_idx=np.argmax(tmp_dist)
                    tmp_vs=np.append(tmp_vs,other_vs[u_idx])
                    other_vs=np.delete(other_vs,u_idx)
                tmp_res[tmp_vs]=DC_i
            elif Out_num<=anchors_mnum and anchors_mnum<DC_member_count[DC_i]:
                min_rhos = np.argsort(rho_DC_i)[:anchors_mnum]
                candidate_vs=source_idx[min_rhos]
                tmp_res[candidate_vs]=DC_i
            else:
                tmp_res[source_idx]=DC_i
        anchors_source_idx=np.where(tmp_res!=-1)[0]
        anchors_DC_flags=tmp_res[anchors_source_idx]
        return anchors_source_idx,anchors_DC_flags

    def DC_graph(self,all_DC_ids,DC_member_count,core_idx,x_mat,densities):
        n_cores = core_idx.shape[0]
        anchors_source_idx,anchors_DC_flags=self.find_DCs_anchors(x_mat,core_idx,DC_member_count,densities,all_DC_ids)
        new_x_mat=x_mat[anchors_source_idx]
        n_instances=new_x_mat.shape[0]
        nbrs = NearestNeighbors(n_neighbors=n_instances,algorithm="kd_tree").fit(new_x_mat)
        query_res=nbrs.kneighbors(new_x_mat)
        knn_dists,knn_indices=query_res[0][:,1:],query_res[1][:,1:]
        edgecount_affinity=self.DC_graph_edgecount_affinity(n_cores,anchors_DC_flags,knn_indices)
        wass_affinity = np.zeros(shape=(n_cores,n_cores))
        for DC_i in range(n_cores):
            for DC_j in range(n_cores):
                if DC_i!=DC_j:
                    wass_sim_ij=self.wasserstein_similarities(DC_i,DC_j,knn_dists,knn_indices,edgecount_affinity,anchors_DC_flags)
                    if wass_sim_ij!=0:
                        wass_affinity[DC_i,DC_j]=1/wass_sim_ij
                    else:
                        raise Exception("There are on edges between DC_i and DC_j")
        wass_affinity=wass_affinity+wass_affinity.T
        rho=self.density_estimation(self.density_estimator,wass_affinity,n_cores)
        delta,DC_subordinate=self.nearest_density_higher_DC(rho,n_cores,wass_affinity)
        return rho,delta,DC_subordinate


# def DC_graph(self,all_DC_ids,DC_member_count,core_idx,nbrs,x_mat,densities, knn_dists,knn_indices):
#     n_cores = core_idx.shape[0]
#     n_neighbors,expand=self.k, 0
#     full_num,curr_num=n_cores*n_cores-n_cores,0
#     new_knn_dists,new_knn_indices=np.empty(shape=knn_dists.shape),np.empty(shape=knn_indices.shape)
#     DC_anchors=self.DC_mult_anchors(x_mat,core_idx,DC_member_count,densities,all_DC_ids)
#     n_instances=DC_anchors.shape[0]-np.where(DC_anchors==-1)[0].shape[0]
#     while curr_num<full_num and n_neighbors<=n_instances-1:
#         n_neighbors =n_neighbors+expand
#         if n_neighbors<self.max_neigh:
#             new_knn_dists,new_knn_indices=knn_dists[:,:n_neighbors],knn_indices[:,:n_neighbors]
#         else:
#             if n_neighbors>n_instances-1:
#                 raise Exception("existing no knn relationship in some of DCs!")
#             query_res = nbrs.kneighbors(x_mat,n_neighbors=n_neighbors+1)
#             new_knn_dists,new_knn_indices=query_res[0][:,1:],query_res[1][:,1:]
#         edgecount_affinity=self.DC_graph_edgecount_affinity(n_cores,all_DC_ids,new_knn_indices)
#         curr_num=np.count_nonzero(edgecount_affinity)
#         expand_tmp=n_instances//self.k-n_neighbors//self.k
#         expand=max(expand_tmp,1)
#         affinity = np.zeros(shape=(n_cores,n_cores))
#     if n_neighbors>n_instances-1:
#         raise Exception("existing no knn relationship in some of DCs!")
#     for DC_i in range(n_cores):
#         for DC_j in range(n_cores):
#             if DC_i!=DC_j:
#                 affinity[DC_i,DC_j]=self.wasserstein_similarities(DC_i,DC_j,new_knn_dists,new_knn_indices,edgecount_affinity,all_DC_ids)
#     rho=self.density_estimation(self.density_estimator,affinity,n_cores)
#     delta,DC_subordinate=self.nearest_density_higher_DC(rho,n_cores,affinity)
#     return rho,delta,DC_subordinate

    def final_DC_cluster(self,instances_num,rho,delta,DC_subordinate,core_idx,all_DC_ids):
        """
            Note
            ----------
            shareed same DC index: rho,delta,DC_gamma,sorted_idx,topK_DC,DC_labels,core_idx
            all_DC_ids: project source index into DC index
        """
        DC_gamma=rho*delta
        sorted_idx=np.argsort(-DC_gamma)
        topK_DC=sorted_idx[:self.n_clusters]
        DC_labels=np.full(DC_gamma.shape[0],-1,np.int32)
        for label_count,i in enumerate(topK_DC):
            DC_labels[i]=label_count        
        for i in range(DC_labels.shape[0]):
            if DC_labels[i]==-1:
                tmp_arr=np.array([],dtype=np.int32)
                tmp_arr=np.append(tmp_arr,i)
                while DC_subordinate[tmp_arr[-1]] not in topK_DC:
                    tmp_arr=np.append(tmp_arr,DC_subordinate[tmp_arr[-1]])
                DC_top_i=DC_subordinate[tmp_arr[-1]]
                DC_labels[tmp_arr]=DC_labels[DC_top_i]
                DC_subordinate[tmp_arr]=DC_top_i
        pred_labels=np.full(instances_num,-1,np.int32)
        for i in range(core_idx.shape[0]):
            locate_original=np.where(all_DC_ids==i)[0]
            pred_labels[locate_original]=DC_labels[i]
        return pred_labels, core_idx[topK_DC]

    def fit(self,x_mat):
        nbrs,knn_dists,knn_indices=self.knn_relationship(x_mat)
        core_idx,DC_member_count,all_DC_ids,densities=self.find_DCs(x_mat,knn_dists,knn_indices)
        if self.n_clusters==len(core_idx):
            pred_labels,centers=all_DC_ids,core_idx
            return pred_labels,centers
        else:
            rho,delta,DC_subordinate=self.DC_graph(all_DC_ids,DC_member_count,core_idx,x_mat,densities)
            pred_labels,centers=self.final_DC_cluster(x_mat.shape[0],rho,delta,DC_subordinate,core_idx,all_DC_ids)
            return pred_labels,centers

if __name__=="__main__":
    datapath="./data/"
    knn_path=datapath+"knn_tmp/"
    wass_kind="out_out"
    # data_names=["banana-ball","Flame","R15","S2","seeds","banknote","segmentation","phoneme","MFCCs","mnist_784"]
    data_names=["banknote"]
    for data_name in data_names:
        x_mat, label_true, n_clusters=load_data(datapath,data_name)
        max_neigh=int((x_mat.shape[0]/n_clusters)*(50/100))
        knn_info={
        "search_index":knn_path+data_name+".joblib",
        "dists_indices":knn_path+f"knn_dists_indices_({data_name}_{max_neigh}).npz",
        "max_neigh":max_neigh
        }
        # k=int((x_mat.shape[0]/n_clusters)*0.4)
        k=20
        t0=time.perf_counter()
        wass_dpc=WassDPC(k,n_clusters,knn_info=knn_info,wass_kind=wass_kind)
        pred_labels,centers=wass_dpc.fit(x_mat)
        t1=time.perf_counter()
        nmi=normalized_mutual_info_score(label_true,pred_labels)
        print(f"wass_dpc result: data_name={data_name},k={k}, n_clusters={n_clusters} nmi={nmi}, time_cost={t1-t0}s")
    





