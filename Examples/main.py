# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import os
from Tools import Graph, Utility
from Learner import GenerateFeatures
import xgboost as xgb
modelpath = '../Learner/ML4C_learner.model'
clf = xgb.XGBClassifier()
clf.load_model(modelpath)
THRESHOLD = 0.1

def orient_skeleton(datapath, skeletonpath, savedir):
    '''
    `datapath': str, path to the data file, should endswith .npy.
        data should be a numpy array in shape (samplesize, num_of_variables).
        currently we only support [DISCRETE] variables.
    `skeletonpath': str, path to the skeleton adjacent matrix, should endswith .txt.
        adjacent matrix should be a numpy array in shape (num_of_variables, num_of_variables).
        if is i--j in the skeleton, then a[i, j] = a[j, i] = 1, otherwise a[i, j] = a[j, i] = 0.
    '''
    os.makedirs(savedir, exist_ok=True)
    data_basename = os.path.basename(datapath).split('.')[0]
    def _conflict_va_wins(va, vb, sa, sb):
        # va, vb are 3-item tuples, sa, sb are there respective scores
        # return False means that va will be removed, and otherwise saved.
        # it returns False, iff
        #   va conflicts with vb, and score sa <= sb.
        # note: the higher score, the better.
        #    in that case nowpval/prepval, too much 0. values)
        if va[0] == vb[1] or va[0] == vb[2]:
            if va[1] == vb[0] or va[2] == vb[0]:
                return sa > sb
        return True

    skeleton = Graph.MixedGraph(graphtxtpath=skeletonpath)
    indexed_features = GenerateFeatures.GetAllTForkFeatures(datapath, skeletonpath)
    tfork_indexs = indexed_features[:, :3].astype(int)
    tfork_features = indexed_features[:, 3:]
    pred_scores = clf.predict_proba(tfork_features)[:, 1]

    is_vstrucs = pred_scores >= THRESHOLD
    isv_TXYs = list(map(tuple, tfork_indexs[is_vstrucs]))
    isv_scores = pred_scores[is_vstrucs]
    vstrucs_candidates = list(zip(isv_TXYs, isv_scores))

    rmlists = set()
    for (va, sa) in vstrucs_candidates:
        flag = False
        for (vb, sb) in vstrucs_candidates:
            if not _conflict_va_wins(va, vb, sa, sb):
                flag = True
                break
        if flag:
            rmlists.add(va)

    pdag = Graph.MixedGraph(nodeIDs=skeleton.NodeIDs)
    for ((j, i, k), _) in vstrucs_candidates:
        if (j, i, k) not in rmlists:
            pdag.add_di_edge(i, j)
            pdag.add_di_edge(k, j)
    for (fromnode, tonode) in skeleton.UndirectedEdges:
        if not (pdag.has_di_edge(fromnode, tonode) or pdag.has_di_edge(tonode, fromnode)):
            pdag.add_undi_edge(fromnode, tonode)
    pdag.apply_meek_rules()
    cpdag_adjmat = pdag.getAdjacencyMatrix()
    resultpath = os.path.join(savedir, f"{data_basename}_result.txt")
    np.savetxt(resultpath, cpdag_adjmat, fmt='%i')
    return cpdag_adjmat, resultpath


if __name__ == '__main__':
    datapath = './benchmarks/data/alarm.npy' # samplesize is 10000
    skeletonpath = './benchmarks/skeletons/alarm.txt'
    truedagpath = './benchmarks/truth_dags/alarm.txt'
    savedir = './benchmarks/results'

    cpdag_prediced_by_ml4c, resultpath = orient_skeleton(datapath, skeletonpath, savedir)
    print(Utility.cal_score(
        truth_G=Graph.DiGraph(graphtxtpath=truedagpath),
        result_G=Graph.MixedGraph(graphtxtpath=resultpath)
    ))
