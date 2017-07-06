#-------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
#-------------------------------------------------------------

__all__ = [ 'getNumCols', 'convertToMatrixBlock', 'convert_caffemodel', 'convert_lmdb_to_jpeg', 'convertToNumPyArr', 'convertToPandasDF', 'SUPPORTED_TYPES' , 'convertToLabeledDF', 'convertImageToNumPyArr']

import numpy as np
import pandas as pd
import os
import math
from pyspark.context import SparkContext
from scipy.sparse import coo_matrix, spmatrix, csr_matrix
from .classloader import *

SUPPORTED_TYPES = (np.ndarray, pd.DataFrame, spmatrix)

def getNumCols(numPyArr):
    if numPyArr.ndim == 1:
        return 1
    else:
        return numPyArr.shape[1]

def get_pretty_str(key, value):
    return '\t"' + key + '": ' + str(value) + ',\n'
        
def save_tensor_csv(tensor, file_path, shouldTranspose):
    w = w.reshape(w.shape[0], -1)
    if shouldTranspose:
        w = w.T
    np.savetxt(file_path, w, delimiter=',')
    with open(file_path + '.mtd', 'w') as file:
        file.write('{\n\t"data_type": "matrix",\n\t"value_type": "double",\n')
        file.write(get_pretty_str('rows', w.shape[0]))
        file.write(get_pretty_str('cols', w.shape[1]))
        file.write(get_pretty_str('nnz', np.count_nonzero(w)))
        file.write('\t"format": "csv",\n\t"description": {\n\t\t"author": "SystemML"\n\t}\n}\n')
    
def convert_caffemodel(sc, deploy_file, caffemodel_file, output_dir, format="binary", is_caffe_installed=False):
    """
    Saves the weights and bias in the caffemodel file to output_dir in the specified format. 
    This method does not requires caffe to be installed.
    
    Parameters
    ----------
    sc: SparkContext
        SparkContext
    
    deploy_file: string
        Path to the input network file
        
    caffemodel_file: string
        Path to the input caffemodel file
    
    output_dir: string
        Path to the output directory
    
    format: string
        Format of the weights and bias (can be binary, csv or text)
    
    is_caffe_installed: bool
        True if caffe is installed
    """
    if is_caffe_installed:
        if format != 'csv':
            raise ValueError('The format ' + str(format) + ' is not supported when caffe is installed. Hint: Please specify format=csv')
        import caffe
        net = caffe.Net(deploy_file, caffemodel_file, caffe.TEST)
        for layerName in net.params.keys():
            num_parameters = len(net.params[layerName])
            if num_parameters == 0:
                continue
            elif num_parameters == 2:
                # Weights and Biases
                layerType = net.layers[list(net._layer_names).index(layerName)].type
                shouldTranspose = True if layerType == 'InnerProduct' else False
                save_tensor_csv(net.params[layerName][0].data, os.path.join(output_dir, layerName + '_weight.mtx'), shouldTranspose)
                save_tensor_csv(net.params[layerName][1].data, os.path.join(output_dir, layerName + '_bias.mtx'), shouldTranspose)
            elif num_parameters == 1:
                # Only Weight
                layerType = net.layers[list(net._layer_names).index(layerName)].type
                shouldTranspose = True if layerType == 'InnerProduct' else False
                save_tensor_csv(net.params[layerName][0].data, os.path.join(output_dir, layerName + '_weight.mtx'), shouldTranspose)
            else:
                raise ValueError('Unsupported number of parameters:' + str(num_parameters))
    else:
        createJavaObject(sc, 'dummy')
        utilObj = sc._jvm.org.apache.sysml.api.dl.Utils()
        utilObj.saveCaffeModelFile(sc._jsc, deploy_file, caffemodel_file, output_dir, format)

    
def convert_lmdb_to_jpeg(lmdb_img_file, output_dir):
    """
    Saves the images in the lmdb file as jpeg in the output_dir. This method requires caffe to be installed along with lmdb and cv2 package.
    To install cv2 package, do `pip install opencv-python`.
    
    Parameters
    ----------
    lmdb_img_file: string
        Path to the input lmdb file
    
    output_dir: string
        Output directory for images (local filesystem)
    """
    import lmdb, caffe, cv2
    lmdb_cursor = lmdb.open(lmdb_file, readonly=True).begin().cursor()
    datum = caffe.proto.caffe_pb2.Datum()
    i = 1
    for _, value in lmdb_cursor:
        datum.ParseFromString(value)
        data = caffe.io.datum_to_array(datum)
        output_file_path = os.path.join(output_dir, 'file_' + str(i) + '.jpg')
        image = np.transpose(data, (1,2,0)) # CxHxW to HxWxC in cv2
        cv2.imwrite(output_file_path, image)
        i = i + 1


def convertToLabeledDF(sparkSession, X, y=None):
    from pyspark.ml.feature import VectorAssembler
    if y is not None:
        pd1 = pd.DataFrame(X)
        pd2 = pd.DataFrame(y, columns=['label'])
        pdf = pd.concat([pd1, pd2], axis=1)
        inputColumns = ['C' + str(i) for i in pd1.columns]
        outputColumns = inputColumns + ['label']
    else:
        pdf = pd.DataFrame(X)
        inputColumns = ['C' + str(i) for i in pdf.columns]
        outputColumns = inputColumns
    assembler = VectorAssembler(inputCols=inputColumns, outputCol='features')
    out = assembler.transform(sparkSession.createDataFrame(pdf, outputColumns))
    if y is not None:
        return out.select('features', 'label')
    else:
        return out.select('features')

def _convertSPMatrixToMB(sc, src):
    src = coo_matrix(src,  dtype=np.float64)
    numRows = src.shape[0]
    numCols = src.shape[1]
    data = src.data
    row = src.row.astype(np.int32)
    col = src.col.astype(np.int32)
    nnz = len(src.col)
    buf1 = bytearray(data.tostring())
    buf2 = bytearray(row.tostring())
    buf3 = bytearray(col.tostring())
    createJavaObject(sc, 'dummy')
    return sc._jvm.org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtilsExt.convertSciPyCOOToMB(buf1, buf2, buf3, numRows, numCols, nnz)
            
def _convertDenseMatrixToMB(sc, src):
    numCols = getNumCols(src)
    numRows = src.shape[0]
    arr = src.ravel().astype(np.float64)
    buf = bytearray(arr.tostring())
    createJavaObject(sc, 'dummy')
    return sc._jvm.org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtilsExt.convertPy4JArrayToMB(buf, numRows, numCols)

def _copyRowBlock(i, sc, ret, src, numRowsPerBlock,  rlen, clen):
    rowIndex = int(i / numRowsPerBlock)
    tmp = src[i:min(i+numRowsPerBlock, rlen),]
    mb = _convertSPMatrixToMB(sc, tmp) if isinstance(src, spmatrix) else _convertDenseMatrixToMB(sc, tmp)
    sc._jvm.org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtilsExt.copyRowBlocks(mb, rowIndex, ret, numRowsPerBlock, rlen, clen)
    return i
    
def convertToMatrixBlock(sc, src, maxSizeBlockInMB=8):
    if not isinstance(sc, SparkContext):
        raise TypeError('sc needs to be of type SparkContext')
    isSparse = True if isinstance(src, spmatrix) else False
    src = np.asarray(src, dtype=np.float64) if not isSparse else src
    if len(src.shape) != 2:
        src_type = str(type(src).__name__)
        raise TypeError('Expected 2-dimensional ' + src_type + ', instead passed ' + str(len(src.shape)) + '-dimensional ' + src_type)
    # Ignoring sparsity for computing numRowsPerBlock for now
    numRowsPerBlock = int(math.ceil((maxSizeBlockInMB*1000000) / (src.shape[1]*8)))
    multiBlockTransfer = False if numRowsPerBlock >= src.shape[0] else True
    if not multiBlockTransfer:
        return _convertSPMatrixToMB(sc, src) if isSparse else _convertDenseMatrixToMB(sc, src)
    else:
        # Since coo_matrix does not have range indexing
        src = csr_matrix(src) if isSparse else src
        rlen = int(src.shape[0])
        clen = int(src.shape[1])
        ret = sc._jvm.org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtilsExt.allocateDenseOrSparse(rlen, clen, isSparse)
        [ _copyRowBlock(i, sc, ret, src, numRowsPerBlock,  rlen, clen) for i in range(0, src.shape[0], numRowsPerBlock) ]
        sc._jvm.org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtilsExt.postProcessAfterCopying(ret)
        return ret

def convertToNumPyArr(sc, mb):
    if isinstance(sc, SparkContext):
        numRows = mb.getNumRows()
        numCols = mb.getNumColumns()
        createJavaObject(sc, 'dummy')
        buf = sc._jvm.org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtilsExt.convertMBtoPy4JDenseArr(mb)
        return np.frombuffer(buf, count=numRows*numCols, dtype=np.float64).reshape((numRows, numCols))
    else:
        raise TypeError('sc needs to be of type SparkContext') # TODO: We can generalize this by creating py4j gateway ourselves

# Example usage: convertImageToNumPyArr(im, img_shape=(3, 224, 224), add_rotated_images=True, add_mirrored_images=True)
# The above call returns a numpy array of shape (6, 50176) in NCHW format
def convertImageToNumPyArr(im, img_shape=None, add_rotated_images=False, add_mirrored_images=False):
    from PIL import Image
    if img_shape is not None:
        num_channels = img_shape[0]
        size = (img_shape[1], img_shape[2])
    else:
        num_channels = 1 if im.mode == 'L' else 3
        size = None
    if num_channels != 1 and num_channels != 3:
        raise ValueError('Expected the number of channels to be either 1 or 3')
    if size is not None:
        im = im.resize(size, Image.LANCZOS)
    expected_mode = 'L' if num_channels == 1 else 'RGB'
    if expected_mode is not im.mode:
        im = im.convert(expected_mode)
    def _im2NumPy(im):
        if expected_mode == 'L':
            return np.asarray(im.getdata()).reshape((1, -1))
        else:
            # (H,W,C) --> (C,H,W) --> (1, C*H*W)
            return np.asarray(im).transpose(2, 0, 1).reshape((1, -1))
    ret = _im2NumPy(im)
    if add_rotated_images:
        ret = np.vstack((ret, _im2NumPy(im.rotate(90)), _im2NumPy(im.rotate(180)), _im2NumPy(im.rotate(270)) ))
    if add_mirrored_images:
        ret = np.vstack((ret, _im2NumPy(im.transpose(Image.FLIP_LEFT_RIGHT)), _im2NumPy(im.transpose(Image.FLIP_TOP_BOTTOM))))
    return ret

def convertToPandasDF(X):
    if not isinstance(X, pd.DataFrame):
        return pd.DataFrame(X, columns=['C' + str(i) for i in range(getNumCols(X))])
    return X
