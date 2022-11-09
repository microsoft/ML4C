# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#!/usr/bin/env python
#-*- coding: utf-8 -*-
import numpy as np
class Dataset(object):
    def __init__(self, dataPath, downsample=None):
        '''
        note: we calculate unique at this data prep step,
                so that each variable's observations are inversed indexs 0, 1, ...
                and so no need to unique again and again during CITest.
              in fact, the benchmarks .txt/.npy is already saved as inversed indexes.
        :param dataPath: string, endswith .npy or .npz or .txt, where it's a frame
            of (sampleSize, varCount) shape, and categorical int32 datatype
        :param downsample: downsample from all samples. default by None.
            if int: randomly pick int samples from all
            if float in range (0, 1]: randomly pick percent from all
            if list: pick samples accoding to list of indexes
        '''
        rawData = np.loadtxt(dataPath) if dataPath.endswith('.txt') else np.load(dataPath) # now (sampleSize, varCount)
        assert np.equal(rawData, rawData.astype(int)).all(), \
            "Currently we only support [DISCRETE] variables. If your variables are indeed discrete, please convert them to integers."
        rawData = rawData.astype(np.int32)
        rawSampleSize = rawData.shape[0]
        if downsample != None:
            if isinstance(downsample, list):
                rawData = rawData[downsample]
            else:
                downsample = downsample if isinstance(downsample, int) else rawSampleSize * downsample
                assert downsample <= rawSampleSize
                rawData = rawData[np.random.choice(range(rawSampleSize), downsample, replace=False)]
        rawData = rawData.T # now (varCount, sampleSize)
        def _unique(row):
            return np.unique(row, return_inverse=True)[1]
        self.IndexedDataT = np.apply_along_axis(_unique, 1, rawData).astype(np.int32)
        self.SampleSize = self.IndexedDataT.shape[1]
        self.VarCount = self.IndexedDataT.shape[0]


