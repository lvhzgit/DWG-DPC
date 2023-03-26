# DWG-DPC
一种利用k近邻密度支配域代表团的密度峰值聚类方法
# 大致思想
利用knn图构建密度支配域，基于密度支配域，应用kmean++思想采样代表团。再用高斯核函数和代表团近邻指向关系建立加权高斯域间相似度，最后使用DPC完成聚类。由于英文名Delegations based Weighted-Gaussian DPC，因此命名DWG-DPC。
# 注意
项目原来的名称是WassDPC，只是因为其相似度计算与wasserstein distance类似，所以才将其命名。其实这样实在不妥，因此才更改名称。项目中的wass_dpc.py是主要算法实现。
