import gzip
import time
from tqdm import tqdm
from bidict import bidict
import numpy as np
from scipy.sparse import dok_matrix
from scipy.sparse.linalg import eigen
from sklearn.neighbors import NearestNeighbors


def log_info(s:str,log_path="./data/others/log.txt"):
    print(s)
    with open(log_path,"a") as fp:
        t = time.strftime(r"%Y-%m-%d %H:%M:%S", time.localtime())
        fp.write("\n"+t+": "+s+"\n")


def lines_count(file_name, isgzfile):
    if isgzfile:
        with gzip.open(file_name, 'rb') as f:
            count = 0
            buf_size = 1024 * 1024 * 3
            buf = f.read(buf_size)
            while buf:
                count += buf.count(b'\n')
                buf = f.read(buf_size)
            return count
    else:
        with open(file_name, 'rb') as f:
            count = 0
            buf_size = 1024 * 1024 * 3
            buf = f.read(buf_size)
            while buf:
                count += buf.count(b'\n')
                buf = f.read(buf_size)
            return count


def iter_csr_M(csr_M):
    # tmp_M = csr_M.tocoo(copy=True)
    # for u,v,t in zip(tmp_M.row,tmp_M.col,tmp_M.data):
    #     yield u,v,t
    for i in range(len(csr_M.indptr)-1):
        columns_i = csr_M.indices[csr_M.indptr[i]:csr_M.indptr[i+1]]
        datas_i = csr_M.data[csr_M.indptr[i]:csr_M.indptr[i+1]]
        for j,di in zip(columns_i,datas_i):
            yield (i,j,di)


def comput_connected_components(graph_mat,hp_edge2id=None,hypergraph=False):
    """
        hypergraph: bool; if True, compute connected component for hypergraph, otherwise for traditional graph
    """
    tmp_queue,res = list(),list()
    curr_component,visited_nodes = set(),set()
    soft_graph_mat = graph_mat+graph_mat.T
    node_size = soft_graph_mat.shape[0]
    graph_type = "hypergraph" if hypergraph else "graph"
    for i in tqdm(range(node_size),bar_format="{l_bar}{bar:10}{r_bar}",desc="comput_{}_components".format(graph_type)):
        if i not in visited_nodes:
            tmp_queue.append(i)
            visited_nodes.add(i)
            while len(tmp_queue)>0:
                curr_v = tmp_queue.pop(0)
                curr_component.add(curr_v)
                for j in soft_graph_mat[curr_v].indices:
                    if hypergraph and hp_edge2id!=None:
                        hp_edge = hp_edge2id.inv[j]
                        for k in hp_edge:
                            if k not in visited_nodes:
                                tmp_queue.append(k)
                                visited_nodes.add(k)
                    elif (not hypergraph) and hp_edge2id == None:
                        if j not in visited_nodes:
                            tmp_queue.append(j)
                            visited_nodes.add(j)
                    else:
                        raise Exception("arguments input error")
            res.append(curr_component.copy())
            curr_component.clear()
    return res

def svd_edge_inference(node_u:int,mG,g,top_k=3,datapath="./data/"):
    """
        complete missing path \n
        return [(node_u,vi_original,etyp_choose),]
    """
    u_type_id = g.phi[node_u]
    u_type = mG.node_typ2id.inv[u_type_id]
    neigh_info = [(i[0],i[1]) for i in mG.get_neighbors_info(u_type)]
    res = []
    for neig_i in neigh_info:
        tmp_nodes,tmp_node2id = set(),bidict()
        vtyp_choose,etyp_choose = mG.node_typ2id[neig_i[0]],mG.edge_typ2id[neig_i[1]]
        for nodetype_i in [u_type,neig_i[0]]:
            with open(datapath+"实体/"+nodetype_i+".txt","r",encoding="utf-8-sig") as fp:
                for line in fp:
                    node_i = line.strip()
                    if len(node_i)==0:
                        continue
                    else:
                        tmp_nodes.add(g.node2id_bdict[node_i])
        tmp_nodes = list(tmp_nodes)
        tmp_nodes.sort()
        for i,ni in enumerate(tmp_nodes):
            tmp_node2id[ni]=i
        tmp_nodes.clear()
        tmp_mat = dok_matrix((len(tmp_node2id),len(tmp_node2id)))
        for edge_i in iter_csr_M(g.adj_mat):
            ui,vi,ei = edge_i
            if {g.phi[ui],g.phi[vi]}=={u_type_id,vtyp_choose} and ei==etyp_choose:
                encode_ui,encode_vi = tmp_node2id[ui],tmp_node2id[vi]
                tmp_mat[encode_ui,encode_vi]=g.weight_mat[ui,vi]
        tmp_mat = tmp_mat.tocsr()
        # u_M,sigma_M,v_MT = svd(tmp_mat.toarray(),full_matrices=False,hermitian=True)
        # u_M,sigma_M,v_MT = svds(tmp_mat,k=max(len(tmp_node2id)-2,4))
        AAT = tmp_mat * tmp_mat.T
        s,u_M= eigen.eigsh(AAT)
        nbrs = NearestNeighbors(n_neighbors=top_k*2)
        nbrs.fit(u_M)
        queried_u = tmp_node2id[node_u]
        infer_res = nbrs.kneighbors([u_M[queried_u]],return_distance=False)
        for i in infer_res[0,1:]:
            vi_original = tmp_node2id.inv[i]
            if not g.adj_mat[node_u,vi_original] and g.phi[vi_original]==vtyp_choose:
                res_i = (node_u,vi_original,etyp_choose)
                res.append(res_i)
            if len(res)>top_k:
                break
    return res


if __name__ == "__main__":
    pass

