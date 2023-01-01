# !/usr/bin/python
# -*- coding:utf-8 -*-

# author "chen"

import os
import math
import csv
import numpy as np
import configparser as ConfigParser
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn.metrics import adjusted_rand_score as ARI
from mpl_toolkits.mplot3d import Axes3D

def read_config():
    config = ConfigParser.RawConfigParser()
    config.read('config.cfg', encoding = 'utf-8')
    config_dict = {}
    config_dict['DATA_File'] = config.get('section', 'DATA_File')
    config_dict['k'] = config.getint('section', 'k')
    config_dict['beta'] = config.getfloat('section', 'beta')
    config_dict['ann'] = config.get('section', 'ann')
    config_dict['metric'] = config.get('section', 'metric')
    config_dict['n_clusters'] = config.getint('section', 'n_clusters')
    return config_dict
    

def get_data(filename):
    data = []
    label = []

    with open(filename, 'r') as file_obj:
        csv_reader = csv.reader(file_obj)
        csv_reader = list(csv_reader)
        for row in csv_reader:
            row = list(row)
            point = []
            for d in row[:-1]:
                point.append(float(d))
            data.append(point)
            label.append(int(float(row[-1])))
    X = np.array(data)
    min_max_scaler = preprocessing.MinMaxScaler()
    X_minMax = min_max_scaler.fit_transform(X)
    # X_minMax =X
    return X_minMax, np.array(label, np.int8)

    # return np.array(data, np.float32), np.array(label, np.int8)


def show_density(data, density):
    n = data.shape[0]
    label = np.full(n, 1, np.int32)

    for i in range(1,10):
        a = np.percentile(density, i*10)
        label[density > a] = i+1
    plt.figure(figsize=[6.40,5.60])
    X = []
    Y = []
    for point in data:
        X.append(point[0])
        Y.append(point[1])
    density = density -4
    plt.scatter(x=X, y=Y, c=density, s=150, cmap=plt.cm.Reds)
    plt.show()

def show_data(data, label):
    plt.figure(figsize=[6.40,5.60])
    X = []
    Y = []
    for point in data:
        X.append(point[0])
        Y.append(point[1])
    plt.scatter(x=X, y=Y, c=label, s=8)
    plt.show()
def show_result(data, label, topK_id):
    plt.figure(figsize=[6.40, 5.60])
    n = data.shape[0]
    X = []
    Y = []
    for i in range(n):
        X.append(data[i][0])
        Y.append(data[i][1])
    plt.scatter(X, Y, c=label,marker='.', s=100)
    center_X = []
    center_Y = []
    for i in topK_id:
        center_X.append(data[i][0])
        center_Y.append(data[i][1])
    plt.scatter(x=center_X, y=center_Y, marker='*', c='red', s=300)
    plt.show()

def plot_sig(density, weight, topK_idx):
    center_X = []
    center_Y = []
    non_center_X = []
    non_center_Y = []
    n_cores = len(density)
    for i in range(n_cores):
        if i in topK_idx:
            center_X.append(density[i])
            center_Y.append(weight[i])
        else:
            non_center_X.append(density[i])
            non_center_Y.append(weight[i])

    plt.figure(figsize=[6.40,5.60])
    plt.scatter(non_center_X, non_center_Y, marker='.', s=200, c='blue')
    plt.scatter(center_X, center_Y, marker='*', s=200,  c='red')
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1), fontsize=12,
              frameon=False)
    plt.show()

def plot_sig1(density, weight, k_th, topK_idx_core):
    num = len(density)
    center_X = []
    center_Y = []
    center_Z = []
    non_center_X = []
    non_center_Y = []
    non_center_Z = []
    n_cores = len(density)
    for i in range(n_cores):
        if i in topK_idx_core:
            center_X.append(density[i])
            center_Y.append(weight[i])
            center_Z.append(k_th[i])
        else:
            non_center_X.append(density[i])
            non_center_Y.append(weight[i])
            non_center_Z.append(k_th[i])

    fig = plt.figure(figsize=[6.40,5.60])
    # plt.scatter(non_center_X, non_center_Y, marker='.', s=200, c='blue')
    # plt.scatter(center_X, center_Y, marker='*', s=200,  c='red')
    ax = Axes3D(fig)
    ax.set_xlabel('density')
    ax.set_ylabel('geometric distance')
    ax.set_zlabel('rank')
    for i in range(num):
        if i in topK_idx_core:
            ax.scatter(density[i], weight[i], k_th[i], c = 'r', marker='p',s =200)
        else:
            ax.scatter(density[i], weight[i], k_th[i], c='b', marker='.',s=200)
    # ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1), fontsize=12,
              frameon=False)
    plt.show()


def normalize_by_minmax(data_sequence):

    min_v = np.min(data_sequence)
    max_v = np.max(data_sequence)
    range_v = max_v - min_v

    data_sequence = (data_sequence - min_v)/range_v

    return data_sequence

def show_DC(data, core_idx,cdh_ids,density):
    start_X =[]
    start_Y = []
    end_X = []
    end_Y = []
    n = data.shape[0]
    label = np.full(n, -1, np.int32)
    count= 0
    sorted_density = np.argsort(-density)
    for i in core_idx:
        label[i] = count
        count += 1
    for i in sorted_density:
        if label[i] == -1:
            label[i] = label[cdh_ids[i]]
    for i in range(n):
        if cdh_ids[i] > -1:
            start_X.append(data[i][0])
            start_Y.append([data[i][1]])
            end_X.append(data[cdh_ids[i]][0])
            end_Y.append(data[cdh_ids[i]][1])
    plt.figure(figsize=[6.40, 5.60])
    DC_X = []
    DC_Y = []
    center_X = []
    center_Y = []
    n = data.shape[0]
    for i in range(n):
        if i in core_idx:
            center_X.append(data[i][0])
            center_Y.append(data[i][1])
            DC_X.append(data[i][0])
            DC_Y.append(data[i][1])
        else:
            DC_X.append(data[i][0])
            DC_Y.append(data[i][1])
    # plt.scatter(point_X, point_Y, marker='.', c='black', s=20, label='non-mode points')
    plt.scatter(DC_X, DC_Y, marker='.', s=100, c=label)
    plt.scatter(center_X, center_Y, marker='.', s=700, c='blue')
    end_X = np.array(end_X)
    end_Y = np.array(end_Y)
    start_X = np.array(start_X)
    start_Y = np.array(start_Y)
    for i in range(n-len(core_idx)):
        plt.quiver(start_X[i], start_Y[i], end_X[i] - start_X[i],end_Y[i] - start_Y[i], angles="xy",scale_units="xy", scale=1,width = 0.008 )
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    # reverse the order
    ax.legend(handles[::-1], labels[::-1], ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1), fontsize=12,
              frameon=False)
    plt.show()

def show_DC_less(data, core_idx, cdh_ids, density, true_cores):
    New_cores = []
    for i in core_idx:
        if i in true_cores:
            continue
        else:
            New_cores.append(i)
    New_cores = np.array(New_cores)
    start_X = []
    start_Y = []
    end_X = []
    end_Y = []
    n = data.shape[0]
    label = np.full(n, -1, np.int32)
    count = 0
    sorted_density = np.argsort(-density)
    for i in core_idx:
        label[i] = count
        count += 1
    for i in sorted_density:
        if label[i] == -1:
            label[i] = label[cdh_ids[i]]
    for i in range(n):
        if cdh_ids[i] > -1:
            start_X.append(data[i][0])
            start_Y.append([data[i][1]])
            end_X.append(data[cdh_ids[i]][0])
            end_Y.append(data[cdh_ids[i]][1])
    plt.figure(figsize=[6.40, 5.60])
    DC_X = []
    DC_Y = []
    center_X = []
    center_Y = []
    n = data.shape[0]
    for i in range(n):
        if i in core_idx:
            center_X.append(data[i][0])
            center_Y.append(data[i][1])
            DC_X.append(data[i][0])
            DC_Y.append(data[i][1])
        else:
            DC_X.append(data[i][0])
            DC_Y.append(data[i][1])
    # plt.scatter(point_X, point_Y, marker='.', c='black', s=20, label='non-mode points')
    plt.scatter(DC_X, DC_Y, marker='.', s=100, c=label)
    plt.scatter(center_X, center_Y, marker='.', s=700, c='blue')
    end_X = np.array(end_X)
    end_Y = np.array(end_Y)
    start_X = np.array(start_X)
    start_Y = np.array(start_Y)
    for i in range(n - len(core_idx)):
        plt.quiver(start_X[i], start_Y[i], end_X[i] - start_X[i], end_Y[i] - start_Y[i], angles="xy",
                   scale_units="xy", scale=1, width=0.008)
    ax = plt.gca()
    for i in New_cores:
        plt.quiver(data[i][0], data[i][1], data[cdh_ids[i]][0] - data[i][0] , data[cdh_ids[i]][1] - data[i][1], angles="xy",
                   scale_units="xy", scale=1, width=0.008)
    handles, labels = ax.get_legend_handles_labels()
    # reverse the order
    ax.legend(handles[::-1], labels[::-1], ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1), fontsize=12,
              frameon=False)
    plt.show()

def show_DC1(data, cdh_ids, label, mat):
    start_X =[]
    start_Y = []
    end_X = []
    end_Y = []
    n = data.shape[0]
    for i in range(n):
            start_X.append(data[i][0])
            start_Y.append(data[i][1])
            end_X.append(data[cdh_ids[i]][0])
            end_Y.append(data[cdh_ids[i]][1])
    plt.figure(figsize=[6.40, 5.60])
    DC_X = []
    DC_Y = []
    center_X = []
    center_Y = []
    n = data.shape[0]
    for i in range(n):
        DC_X.append(data[i][0])
        DC_Y.append(data[i][1])
    center_1 = []
    center_2 = []
    for i in mat:
        center_1.append(i[0])
        center_2.append(i[1])
    plt.scatter(DC_X, DC_Y, marker='.', s=200, c='red')
    # plt.scatter(point_X, point_Y, marker='.', c='black', s=20, label='non-mode points')
    plt.scatter(DC_X, DC_Y, marker='.', s=100, c=label)
    # plt.scatter(center_X, center_Y, marker='.', s=700, c='blue')
    end_X = np.array(end_X)
    end_Y = np.array(end_Y)
    start_X = np.array(start_X)
    start_Y = np.array(start_Y)
    for i in range(n):
        # plt.quiver(start_X[i], start_Y[i], end_X[i] - start_X[i],end_Y[i] - start_Y[i], angles="xy",scale_units="xy", scale=1,width = 0.0008 )
        plt.plot([start_X[i], end_X[i]], [start_Y[i], end_Y[i]], color='black')
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    # reverse the order
    ax.legend(handles[::-1], labels[::-1], ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1), fontsize=12,
              frameon=False)

    plt.show()

