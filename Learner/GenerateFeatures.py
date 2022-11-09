# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#!/usr/bin/env python
#-*- coding: utf-8 -*-
import numpy as np
import os, copy, time
from Tools import Graph, Utility, Dataset, CITester
from itertools import product

maxConSizeForBackSearch = 2

def GetAllTForkFeatures(datapath, skeletonpath):
    svdir = os.path.join(os.path.dirname(datapath), 'features')
    os.makedirs(svdir, exist_ok=True)
    data_basename = os.path.basename(datapath).split('.')[0]
    outfile = os.path.join(svdir, f"{data_basename}_features.npy")
    if os.path.exists(outfile): return np.load(outfile)

    skeleton = Graph.MixedGraph(graphtxtpath=skeletonpath)
    tforks = skeleton.tforks
    PC_of_nodes = {node: skeleton.getPC(node) for node in skeleton.NodeIDs}

    dataset = Dataset.Dataset(datapath)
    cit = CITester.CITester(dataset.IndexedDataT, maxCountOfSepsets=50)

    all_features = []
    tic = time.time()
    for ind, tf in enumerate(tforks):
        if ind % 100 == 0:
            tac = time.time()
            print(f"============ processing {ind}/{len(tforks)} TForks, takes {tac - tic : .2f} sec ============")
            tic = tac
        T, X, Y = tf
        PCT, PCX, PCY = PC_of_nodes[T], PC_of_nodes[X], PC_of_nodes[Y]
        PCX_minus_T = PCX - {T}
        PCY_minus_T = PCY - {T}
        PCX_PCY_pairs = [_ for _ in {tuple(sorted(tp)) for tp in product(PCX_minus_T, PCY_minus_T)} # sort and set to remove repeat
                         if _[0] != _[1] and not skeleton.adjacent_in_mixed_graph(_[0], _[1])] # add this restrict: no adjacent pair
        Xpcypairs = [_ for _ in product({X}, PCY_minus_T) if not skeleton.adjacent_in_mixed_graph(_[0], _[1])]
        Ypcxpairs = [_ for _ in product({Y}, PCX_minus_T) if not skeleton.adjacent_in_mixed_graph(_[0], _[1])]
        feature = cit.ExtractTForkFeatureBasedOnPC(T, X, Y, PCT, PCX, PCY, Xpcypairs, Ypcxpairs, PCX_PCY_pairs)
        all_features.append([[T, X, Y], feature])

    ke = Utility.Kernel_Embedding()
    processed_all_features = []
    for tf, raw_feature in all_features:
        T, X, Y = tf
        scalings_overlaps = raw_feature[:12] # 12 = 5 scalings + 7 overlaps
        est_conds = raw_feature[12:]
        unitary_XY_T = list(est_conds[0][0][0])
        processed_fts = copy.copy(scalings_overlaps + unitary_XY_T)

        unitary_flag = True
        for condlist in est_conds:
            for estlist in condlist:
                if unitary_flag: # jump the first unitary, no need to percentile
                    unitary_flag = False; continue
                pvals = [_[0] for _ in estlist]
                svrts = [_[1] for _ in estlist]
                meanstdmaxmin_values = [len(pvals)] + Utility.meanstdmaxmin(pvals) + Utility.meanstdmaxmin(svrts) # length=1+4+4=9
                pval_embd = ke.get_empirical_embedding(pvals) # length=15
                svrt_embd = ke.get_empirical_embedding(svrts) # length=15
                processed_fts.extend(meanstdmaxmin_values + pval_embd + svrt_embd)

        processed_all_features.append([T, X, Y] + processed_fts)
        # for each line of record, feature length = 755 = 5 scalings + 7 overlaps + 2 unitarys + (1 size + \
        #       2 * 4 meanstdmaxmin + 2 * 15 embedding-dim) * 19 est-cond-pairs left  (note: *2 means pval and svrt)
        # and thus outfile is in shape (*, 758), with first 3 dims [T, X, Y]

    processed_all_features = np.array(processed_all_features)
    np.save(outfile, processed_all_features)
    return processed_all_features

def GetAllTForkLabelsForTraining(featurepath, truedagpath):
    features = np.load(featurepath)
    tforks = features[:, :3]
    truth = Graph.DiGraph(graphtxtpath=truedagpath)
    vstrucs = truth.vstrucs
    labels = [tuple(tf) in vstrucs for tf in tforks]
    np.save(featurepath.replace('features.npy', 'labels.npy'), np.array(labels))

if __name__ == '__main__':
    ########### for training synthetic data ###########
    samplesize = 10000
    graph_path = './synthetics/graph'
    data_path = './synthetics/data'
    feature_path = './synthetics/data/features'
    for graph_txt_name in [_ for _ in os.listdir('./synthetics/graph') if _.endswith('_graph.txt')]:
        data_npy_name = graph_txt_name.replace('_graph.txt', f'_{samplesize}.npy')
        GetAllTForkFeatures(os.path.join(data_path, data_npy_name), os.path.join(graph_path, graph_txt_name))
        # here we use dag for mixed graph; it's the same as using skeleton.
        GetAllTForkLabelsForTraining(os.path.join(feature_path, data_npy_name.replace('.npy', '_features.npy')),
                                     os.path.join(graph_path, graph_txt_name))
    # ########### for training synthetic data ###########