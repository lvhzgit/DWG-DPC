# DWG-DPC
一种利用k近邻密度支配域代表团的密度峰值聚类方法
# 大致思想
利用knn图构建密度支配域，基于密度支配域，应用kmean++思想采样代表团。再用高斯核函数和代表团近邻指向关系建立加权高斯域间相似度，最后使用DPC完成聚类。由于英文名Delegations based Weighted-Gaussian DPC，因此命名DWG-DPC。
