# -------------------------------------------------------------
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
# -------------------------------------------------------------

from typing import Dict

from systemds.operator import OperationNode
from systemds.script_building.dag import OutputType
from systemds.utils.consts import VALID_INPUT_TYPES

__all__ = ['l2svm', 'lm', 'kmeans', 'pca', 'multiLogReg', 'multiLogRegPredict']


def l2svm(x: OperationNode, y: OperationNode, **kwargs: Dict[str, VALID_INPUT_TYPES]) -> OperationNode:
    """
    Perform L2SVM on matrix with labels given.

    :param x: Input dataset
    :param y: Input labels in shape of one column
    :param kwargs: Dictionary of extra arguments 
    :return: `OperationNode` containing the model fit.
    """
    x._check_matrix_op()
    params_dict = {'X': x, 'Y': y}
    params_dict.update(kwargs)
    return OperationNode(x.sds_context, 'l2svm', named_input_nodes=params_dict)


def lm(x: OperationNode, y: OperationNode, **kwargs: Dict[str, VALID_INPUT_TYPES]) -> OperationNode:
    """
    Performs LM on matrix with labels given.

    :param x: Input dataset
    :param y: Input labels in shape of one column
    :param kwargs: Dictionary of extra arguments 
    :return: `OperationNode` containing the model fit.
    """

    if x.shape[0] == 0:
        raise ValueError("Found array with 0 feature(s) (shape={s}) while a minimum of 1 is required."
                         .format(s=x.shape))
    if y.shape[0] == 0:
        raise ValueError("Found array with 0 feature(s) (shape={s}) while a minimum of 1 is required."
                         .format(s=y.shape))

    params_dict = {'X': x, 'y': y}
    params_dict.update(kwargs)
    return OperationNode(x.sds_context, 'lm', named_input_nodes=params_dict)


def kmeans(x: OperationNode, **kwargs: Dict[str, VALID_INPUT_TYPES]) -> OperationNode:
    """
    Performs KMeans on matrix input.

    :param x: Input dataset to perform K-Means on.
    :param k: The number of centroids to use for the algorithm.
    :param runs: The number of concurrent instances of K-Means to run (with different initial centroids).
    :param max_iter: The maximum number of iterations to run the K-Means algorithm for.
    :param eps: Tolerance for the algorithm to declare convergence using WCSS change ratio.
    :param is_verbose: Boolean flag if the algorithm should be run in a verbose manner.
    :param avg_sample_size_per_centroid: The average number of records per centroid in the data samples.
    :return: `OperationNode` List containing two outputs 1. the clusters, 2 the cluster ID associated with each row in x.
    """

    x._check_matrix_op()
    if x.shape[0] == 0:
        raise ValueError("Found array with 0 feature(s) (shape={s}) while a minimum of 1 is required."
                         .format(s=x.shape))

    if 'k' in kwargs.keys() and kwargs.get('k') < 1:
        raise ValueError(
            "Invalid number of clusters in K-Means, number must be integer above 0")

    params_dict = {'X': x}
    params_dict.update(kwargs)
    return OperationNode(x.sds_context, 'kmeans', named_input_nodes=params_dict, output_type=OutputType.LIST, number_of_outputs=2)


def pca(x: OperationNode, **kwargs: Dict[str, VALID_INPUT_TYPES]) -> OperationNode:
    """
    Performs PCA on the matrix input

    :param x: Input dataset to perform Principal Componenet Analysis (PCA) on.
    :param K: The number of reduced dimensions.
    :param center: Boolean specifying if the input values should be centered.
    :param scale: Boolean specifying if the input values should be scaled.
     :return: `OperationNode` List containing two outputs 1. The dimensionality reduced X input, 2. A matrix to reduce dimensionality similarly on unseen data.
    """

    x._check_matrix_op()
    if x.shape[0] == 0:
        raise ValueError("Found array with 0 feature(s) (shape={s}) while a minimum of 1 is required."
                         .format(s=x.shape))

    if 'K' in kwargs.keys() and kwargs.get('K') < 1:
        raise ValueError(
            "Invalid number of dimensions in PCA, number must be integer above 0")

    if 'scale' in kwargs.keys():
        if kwargs.get('scale') == True:
            kwargs.set('scale', "TRUE")
        elif kwargs.get('scale' == False):
            kwargs.set('scale', "FALSE")

    if 'center' in kwargs.keys():
        if kwargs.get('center') == True:
            kwargs.set('center', "TRUE")
        elif kwargs.get('center' == False):
            kwargs.set('center', "FALSE")

    params_dict = {'X': x}
    params_dict.update(kwargs)
    return OperationNode(x.sds_context, 'pca', named_input_nodes=params_dict,  output_type=OutputType.LIST, number_of_outputs=2)


def multiLogReg(x: OperationNode, y: OperationNode, **kwargs: Dict[str, VALID_INPUT_TYPES]) -> OperationNode:
    """
    Performs Multiclass Logistic Regression on the matrix input
    using Trust Region method.

    See: Trust Region Newton Method for Logistic Regression, Lin, Weng and Keerthi, JMLR 9 (2008) 627-650)

    :param x: Input dataset to perform logstic regression on
    :param y: Labels rowaligned with the input dataset
    :param icpt: Intercept, default 2, Intercept presence, shifting and rescaling X columns:
        0 = no intercept, no shifting, no rescaling;
        1 = add intercept, but neither shift nor rescale X;
        2 = add intercept, shift & rescale X columns to mean = 0, variance = 1
    :param tol: float tolerance for the algorithm.
    :param reg: Regularization parameter (lambda = 1/C); intercept settings are not regularized.
    :param maxi: Maximum outer iterations of the algorithm
    :param maxii: Maximum inner iterations of the algorithm
     :return: `OperationNode` of a matrix containing the regression parameters trained.
    """
    
    if x.shape[0] == 0:
        raise ValueError("Found array with 0 feature(s) (shape={s}) while a minimum of 1 is required."
                         .format(s=x.shape))
    if y.shape[0] == 0:
        raise ValueError("Found array with 0 feature(s) (shape={s}) while a minimum of 1 is required."
                         .format(s=y.shape))
    if -1 in x.shape:
        output_shape = (-1,)
    else:
        output_shape = (x.shape[1],)
        
    params_dict = {'X': x, 'Y': y}
    params_dict.update(kwargs)
    return OperationNode(x.sds_context, 'multiLogReg', named_input_nodes=params_dict, shape = output_shape)


def multiLogRegPredict(x: OperationNode, b: OperationNode, y: OperationNode, **kwargs: Dict[str, VALID_INPUT_TYPES]) -> OperationNode:
    """
    Performs prediction on input data x using the model trained, b.

    :param x: The data to perform classification on.
    :param b: The regression parameters trained from multiLogReg.
    :param y: The Labels expected to be contained in the X dataset, to calculate accuracy.
    :param verbose: Boolean specifying if the prediction should be verbose.
    :return: `OperationNode` List containing three outputs. 
        1. The predicted means / probabilities
        2. The predicted response vector
        3. The scalar value of accuracy
    """

    if x.shape[0] == 0:
        raise ValueError("Found array with 0 feature(s) (shape={s}) while a minimum of 1 is required."
                         .format(s=x.shape))
    if b.shape[0] == 0:
        raise ValueError("Found array with 0 feature(s) (shape={s}) while a minimum of 1 is required."
                         .format(s=y.shape))
    if y.shape[0] == 0:
        raise ValueError("Found array with 0 feature(s) (shape={s}) while a minimum of 1 is required."
                         .format(s=y.shape))

    params_dict = {'X': x, 'B': b, 'Y': y}
    params_dict.update(kwargs)
    return OperationNode(x.sds_context, 'multiLogRegPredict', named_input_nodes=params_dict,  output_type=OutputType.LIST, number_of_outputs=3, output_types=[OutputType.MATRIX,OutputType.MATRIX,OutputType.DOUBLE])
