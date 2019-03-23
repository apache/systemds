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

# Script to generate caffe proto and .caffemodel files from Keras models


import numpy as np
import os
import math
from itertools import chain
try:
    from itertools import imap
except ImportError:
    # Support Python 3x
    imap = map
from ..converters import *
from ..classloader import *
import keras
from keras import backend as K
from keras.layers import Activation

try:
    import py4j.java_gateway
    from py4j.java_gateway import JavaObject
    from pyspark import SparkContext
except ImportError:
    raise ImportError(
        'Unable to import `pyspark`. Hint: Make sure you are running with PySpark.')

# --------------------------------------------------------------------------------------
# Design Document:
# We support Keras model by first converting it to Caffe models and then using Caffe2DML to read them
#
# Part 1: Keras network to Caffe network conversion:
# - Core logic: model.layers.flatMap(layer => _parseJSONObject(_parseKerasLayer(layer)))
# That is, for each layer, we first convert it into JSON format and then convert the JSON object into String
# - This is true for all the layers except the "specialLayers" (given in below hashmap). These are redirected to their custom parse function in _parseKerasLayer.
# - To add an activation, simply add the keras type to caffe type in supportedCaffeActivations.
# - To add a layer, add the corresponding caffe layer type in supportedLayers. If the layer accepts parameters then update layerParamMapping too.
# - The above logic is implemented in the function converKerasToCaffeNetwork
#
#
# Example guide to add a new layer that does not have a weight and bias (eg: UpSampling2D or ZeroPadding2D):
# - Add mapping of Keras class to Caffe layer in the supportedLayers map below
# - Define a helper method that returns Caffe's layer parameter in JSON-like data structure. See getConvParam, getUpSamplingParam, getPaddingParam, etc.
# - Add mapping of Keras class to Caffe layer parameter in the layerParamMapping map below
# --------------------------------------------------------------------------------------

supportedCaffeActivations = {
    'relu': 'ReLU',
    'softmax': 'Softmax',
    'sigmoid': 'Sigmoid'
}
supportedLayers = {
    keras.layers.InputLayer: 'Data',
    keras.layers.Dense: 'InnerProduct',
    keras.layers.Dropout: 'Dropout',
    keras.layers.Add: 'Eltwise',
    keras.layers.Concatenate: 'Concat',
    keras.layers.Conv2DTranspose: 'Deconvolution',
    keras.layers.Conv2D: 'Convolution',
    keras.layers.UpSampling2D: 'Upsample',
    keras.layers.MaxPooling2D: 'Pooling',
    keras.layers.AveragePooling2D: 'Pooling',
    keras.layers.SimpleRNN: 'RNN',
    keras.layers.LSTM: 'LSTM',
    keras.layers.Flatten: 'Flatten',
    keras.layers.BatchNormalization: 'None',
    keras.layers.Activation: 'None',
    keras.layers.ZeroPadding2D: 'Padding'
}


def _getInboundLayers(layer):
    in_names = []
    # get inbound nodes to current layer (support newer as well as older APIs)
    inbound_nodes = layer.inbound_nodes if hasattr(
        layer, 'inbound_nodes') else layer._inbound_nodes
    for node in inbound_nodes:
        node_list = node.inbound_layers  # get layers pointing to this node
        in_names = in_names + node_list
    return list(in_names)
    # For Caffe2DML to reroute any use of Flatten layers
    #return list(chain.from_iterable([_getInboundLayers(l) if isinstance(
    #    l, keras.layers.Flatten) else [l] for l in in_names]))


def _getCompensatedAxis(layer):
    compensated_axis = layer.axis
    # Cover all cases for anything accessing the 0th index or the last index
    if layer.axis > 0 and layer.axis < layer.input[0].shape.ndims - 1:
        compensated_axis = layer.axis + 1
    elif layer.axis < -1 and layer.axis > -(layer.input[0].shape.ndims):
        compensated_axis = layer.axis + 1
    elif layer.axis == -1 or layer.axis == layer.input[0].shape.ndims - 1:
        compensated_axis = 1
    return compensated_axis


str_keys = ['name', 'type', 'top', 'bottom']


def toKV(key, value):
    return str(key) + ': "' + str(value) + \
           '"' if key in str_keys else str(key) + ': ' + str(value)


def _parseJSONObject(obj):
    rootName = list(obj.keys())[0]
    ret = ['\n', rootName, ' {']
    for key in obj[rootName]:
        if isinstance(obj[rootName][key], dict):
            ret = ret + ['\n\t', key, ' {']
            for key1 in obj[rootName][key]:
                ret = ret + ['\n\t\t', toKV(key1, obj[rootName][key][key1])]
            ret = ret + ['\n\t', '}']
        elif isinstance(obj[rootName][key], list):
            for v in obj[rootName][key]:
                ret = ret + ['\n\t', toKV(key, v)]
        else:
            ret = ret + ['\n\t', toKV(key, obj[rootName][key])]
    return ret + ['\n}']


def _getBottomLayers(layer):
    return [bottomLayer.name for bottomLayer in _getInboundLayers(layer)]


def _parseActivation(layer, customLayerName=None):
    kerasActivation = keras.activations.serialize(layer.activation)
    if kerasActivation not in supportedCaffeActivations:
        raise TypeError(
            'Unsupported activation ' +
            kerasActivation +
            ' for the layer:' +
            layer.name)
    if customLayerName is not None:
        return {'layer': {'name': customLayerName,
                          'type': supportedCaffeActivations[kerasActivation], 'top': layer.name, 'bottom': layer.name}}
    else:
        return {'layer': {'name': layer.name,
                          'type': supportedCaffeActivations[kerasActivation], 'top': layer.name,
                          'bottom': _getBottomLayers(layer)}}


def _shouldParseActivation(layer):
    ignore_activation = [keras.layers.SimpleRNN, keras.layers.LSTM]
    return hasattr(layer, 'activation') and (type(
        layer) not in ignore_activation) and keras.activations.serialize(layer.activation) != 'linear'


def _parseKerasLayer(layer):
    layerType = type(layer)
    if layerType in specialLayers:
        return specialLayers[layerType](layer)
    elif layerType == keras.layers.Activation:
        return [_parseActivation(layer)]
    param = layerParamMapping[layerType](layer)
    layerArgs = {}
    layerArgs['name'] = layer.name
    if layerType == keras.layers.InputLayer:
        layerArgs['type'] = 'Data'
        layerArgs['top'] = 'label' # layer.name: TODO
    else:
        layerArgs['type'] = supportedLayers[layerType]
        layerArgs['bottom'] = _getBottomLayers(layer)
        layerArgs['top'] = layer.name
    if len(param) > 0:
        paramName = list(param.keys())[0]
        layerArgs[paramName] = param[paramName]
    ret = { 'layer': layerArgs }
    return [ret, _parseActivation(
        layer, layer.name + '_activation')] if _shouldParseActivation(layer) else [ret]


def _parseBatchNorm(layer):
    # TODO: Ignoring axis
    bnName = layer.name + '_1'
    config = layer.get_config()
    bias_term = 'true' if config['center'] else 'false'
    return [{'layer': {'name': bnName, 'type': 'BatchNorm', 'bottom': _getBottomLayers(layer), 'top': bnName,
                       'batch_norm_param': {'moving_average_fraction': layer.momentum, 'eps': layer.epsilon}}}, {
                'layer': {'name': layer.name, 'type': 'Scale', 'bottom': bnName, 'top': layer.name,
                          'scale_param': {'bias_term': bias_term}}}]


# The special are redirected to their custom parse function in _parseKerasLayer
specialLayers = {
    keras.layers.BatchNormalization: _parseBatchNorm
}

# Used by convolution and maxpooling to return the padding value as integer based on type 'same' and 'valid'
def getPadding(kernel_size, padding):
    if padding.lower() == 'same':
        return int(kernel_size/2)
    elif padding.lower() == 'valid':
        return 0
    else:
        raise ValueError('Unsupported padding:' + str(padding))

# Used by padding to extract different types of possible padding:
# int: the same symmetric padding is applied to height and width.
# tuple of 2 ints: interpreted as two different symmetric padding values for height and width: (symmetric_height_pad, symmetric_width_pad)
# tuple of 2 tuples of 2 ints: interpreted as  ((top_pad, bottom_pad), (left_pad, right_pad))
def get2Tuple(val):
    return [val, val] if isinstance(val, int) else [val[0], val[1]]

# Helper method to return Caffe's ConvolutionParameter in JSON-like data structure
def getConvParam(layer):
    # TODO: dilation_rate, kernel_constraint and bias_constraint are not supported
    stride = (1, 1) if layer.strides is None else get2Tuple(layer.strides)
    kernel_size = get2Tuple(layer.kernel_size)
    config = layer.get_config()
    if not layer.use_bias:
        raise Exception('use_bias=False is not supported for the Conv2D layer. Consider setting use_bias to true.')
    return {'num_output': layer.filters, 'bias_term': str(config['use_bias']).lower(
    ), 'kernel_h': kernel_size[0], 'kernel_w': kernel_size[1], 'stride_h': stride[0], 'stride_w': stride[1],
            'pad_h': getPadding(kernel_size[0], layer.padding), 'pad_w': getPadding(kernel_size[1], layer.padding)}


# Helper method to return newly added UpsampleParameter
# (search for UpsampleParameter in the file src/main/proto/caffe/caffe.proto) in JSON-like data structure
def getUpSamplingParam(layer):
    # TODO: Skipping interpolation type
    size = get2Tuple(layer.size)
    return {'size_h': size[0], 'size_w': size[1]}

# Helper method to return newly added PaddingParameter
# (search for UpsampleParameter in the file src/main/proto/caffe/caffe.proto) in JSON-like data structure
def getPaddingParam(layer):
    if isinstance(layer.padding, int):
        padding = get2Tuple(layer.padding) + get2Tuple(layer.padding)
    elif hasattr(layer.padding, '__len__') and len(layer.padding) == 2:
        padding = get2Tuple(layer.padding[0]) + get2Tuple(layer.padding[1])
    else:
        raise ValueError('padding should be either an int, a tuple of 2 ints or or a tuple of 2 tuples of 2 ints. Found: ' + str(layer.padding))
    return {'top_pad': padding[0], 'bottom_pad': padding[1], 'left_pad': padding[2], 'right_pad': padding[3], 'pad_value':0}

# Helper method to return Caffe's PoolingParameter in JSON-like data structure
def getPoolingParam(layer, pool='MAX'):
    stride = (1, 1) if layer.strides is None else get2Tuple(layer.strides)
    pool_size = get2Tuple(layer.pool_size)
    return {'pool': pool, 'kernel_h': pool_size[0], 'kernel_w': pool_size[1],
            'stride_h': stride[0], 'stride_w': stride[1], 'pad_h': getPadding(pool_size[0], layer.padding),
            'pad_w': getPadding(pool_size[1], layer.padding)}

# Helper method to return Caffe's RecurrentParameter in JSON-like data structure
def getRecurrentParam(layer):
    if (not layer.use_bias):
        raise Exception('Only use_bias=True supported for recurrent layers')
    if (keras.activations.serialize(layer.activation) != 'tanh'):
        raise Exception('Only tanh activation supported for recurrent layers')
    if (layer.dropout != 0 or layer.recurrent_dropout != 0):
        raise Exception('Only dropout not supported for recurrent layers')
    return {'num_output': layer.units, 'return_sequences': str(
        layer.return_sequences).lower()}

# Helper method to return Caffe's InnerProductParameter in JSON-like data structure
def getInnerProductParam(layer):
    if len(layer.output_shape) != 2:
        raise Exception('Only 2-D input is supported for the Dense layer in the current implementation, but found '
                        + str(layer.input_shape) + '. Consider adding a Flatten before ' + str(layer.name))
    if not layer.use_bias:
        raise Exception('use_bias=False is not supported for the Dense layer. Consider setting use_bias to true.')
    return {'num_output': layer.units}

# Helper method to return Caffe's DropoutParameter in JSON-like data structure
def getDropoutParam(layer):
    if layer.noise_shape is not None:
        supported = True
        if len(layer.input_shape) != len(layer.noise_shape):
            supported = False
        else:
            for i in range(len(layer.noise_shape)-1):
                # Ignore the first dimension
                if layer.input_shape[i+1] != layer.noise_shape[i+1]:
                    supported = False
        if not supported:
            raise Exception('noise_shape=' + str(layer.noise_shape) + ' is not supported for Dropout layer with input_shape='
                            + str(layer.input_shape))
    return {'dropout_ratio': layer.rate}

layerParamMapping = {
    keras.layers.InputLayer: lambda l:
    {'data_param': {'batch_size': l.batch_size}},
    keras.layers.Dense: lambda l:
    {'inner_product_param': getInnerProductParam(l)},
    keras.layers.Dropout: lambda l:
    {'dropout_param': getDropoutParam(l)},
    keras.layers.Add: lambda l:
    {'eltwise_param': {'operation': 'SUM'}},
    keras.layers.Concatenate: lambda l:
    {'concat_param': {'axis': _getCompensatedAxis(l)}},
    keras.layers.Conv2DTranspose: lambda l:
    {'convolution_param': getConvParam(l)}, # will skip output_padding
    keras.layers.UpSampling2D: lambda l:
    {'upsample_param': getUpSamplingParam(l)},
    keras.layers.ZeroPadding2D: lambda l:
    {'padding_param': getPaddingParam(l)},
    keras.layers.Conv2D: lambda l:
    {'convolution_param': getConvParam(l)},
    keras.layers.MaxPooling2D: lambda l:
    {'pooling_param': getPoolingParam(l, 'MAX')},
    keras.layers.AveragePooling2D: lambda l:
    {'pooling_param': getPoolingParam(l, 'AVE')},
    keras.layers.SimpleRNN: lambda l:
    {'recurrent_param': getRecurrentParam(l)},
    keras.layers.LSTM: lambda l:
    {'recurrent_param': getRecurrentParam(l)},
    keras.layers.Flatten: lambda l: {},
}


def _checkIfValid(myList, fn, errorMessage):
    bool_vals = np.array([fn(elem) for elem in myList])
    unsupported_elems = np.where(bool_vals)[0]
    if len(unsupported_elems) != 0:
        raise ValueError(errorMessage +
                         str(np.array(myList)[unsupported_elems]))


def _transformLayer(layer, batch_size):
    if isinstance(layer, keras.layers.InputLayer):
        layer.batch_size = batch_size
    return [layer]


def _appendKerasLayers(fileHandle, kerasLayers, batch_size):
    if len(kerasLayers) >= 1:
        transformedLayers = list(
            chain.from_iterable(
                imap(
                    lambda layer: _transformLayer(
                        layer,
                        batch_size),
                    kerasLayers)))
        jsonLayers = list(
            chain.from_iterable(
                imap(
                    lambda layer: _parseKerasLayer(layer),
                    transformedLayers)))
        parsedLayers = list(
            chain.from_iterable(
                imap(
                    lambda layer: _parseJSONObject(layer),
                    jsonLayers)))
        fileHandle.write(''.join(parsedLayers))
        fileHandle.write('\n')


def lossLayerStr(layerType, bottomLayer):
    return 'layer {\n  name: "loss"\n  type: "' + layerType + \
           '"\n  bottom: "' + bottomLayer + '"\n  bottom: "label"\n  top: "loss"\n}\n'


def _appendKerasLayerWithoutActivation(fileHandle, layer, batch_size):
    if not isinstance(layer, keras.layers.Activation):
        lastLayerActivation = layer.activation
        layer.activation = keras.activations.linear
        _appendKerasLayers(fileHandle, [layer], batch_size)
        layer.activation = lastLayerActivation


def _getExactlyOneBottomLayer(layer):
    bottomLayers = _getBottomLayers(layer)
    if len(bottomLayers) != 1:
        raise Exception('Expected only one bottom layer for ' +
                        str(layer.name) + ', but found ' + str(bottomLayers))
    return bottomLayers[0]


def _isMeanSquaredError(loss):
    return loss == 'mean_squared_error' or loss == 'mse' or loss == 'MSE'

def _appendInputLayerIfNecessary(kerasModel):
    """ Append an Input layer if not present: required for versions 2.1.5 (works with 2.1.5, but not with 2.2.4) and return all the layers  """
    input_layer = []
    if not any([isinstance(l, keras.layers.InputLayer) for l in kerasModel.layers]):
        input_name = kerasModel.layers[0]._inbound_nodes[0].inbound_layers[0].name
        input_shape = kerasModel.layers[0].input_shape
        input_layer = [keras.layers.InputLayer(name=input_name, input_shape=input_shape)]
    return input_layer + kerasModel.layers

def _throwLossException(loss, lastLayerActivation=None):
    if lastLayerActivation is not None:
        activationMsg = ' (where last layer activation ' + lastLayerActivation + ')'
    else:
        activationMsg = ''
    raise Exception('Unsupported loss layer ' + str(loss) + activationMsg)

def convertKerasToCaffeNetwork(
        kerasModel, outCaffeNetworkFilePath, batch_size):
    _checkIfValid(kerasModel.layers, lambda layer: False if type(
        layer) in supportedLayers else True, 'Unsupported Layers:')
    with open(outCaffeNetworkFilePath, 'w') as f:
        layers = _appendInputLayerIfNecessary(kerasModel)
        # Write the parsed layers for all but the last layer
        _appendKerasLayers(f, layers[:-1], batch_size)
        # Now process the last layer with loss
        lastLayer = layers[-1]
        if _isMeanSquaredError(kerasModel.loss):
            # No need to inspect the last layer, just append EuclideanLoss after writing the last layer
            _appendKerasLayers(f, [lastLayer], batch_size)
            f.write(lossLayerStr('EuclideanLoss', lastLayer.name))
        elif kerasModel.loss == 'categorical_crossentropy':
            # Three cases:
            if isinstance(lastLayer, keras.layers.Softmax):
                # Case 1: Last layer is a softmax.
                f.write(lossLayerStr('SoftmaxWithLoss', _getExactlyOneBottomLayer(lastLayer)))
            else:
                lastLayerActivation = str(keras.activations.serialize(lastLayer.activation))
                if lastLayerActivation == 'softmax' and kerasModel.loss == 'categorical_crossentropy':
                    # Case 2: Last layer activation is softmax.
                    # First append the last layer without its activation and then append SoftmaxWithLoss
                    bottomLayer = _getExactlyOneBottomLayer(lastLayer) if isinstance(
                        lastLayer, keras.layers.Activation) else lastLayer.name
                    _appendKerasLayerWithoutActivation(f, lastLayer, batch_size)
                    f.write(lossLayerStr('SoftmaxWithLoss', bottomLayer))
                else:
                    # Case 3: Last layer activation is not softmax => Throw error
                    _throwLossException(kerasModel.loss, lastLayerActivation)
        else:
            _throwLossException(kerasModel.loss)


def getNumPyMatrixFromKerasWeight(param):
    x = np.array(param)
    if len(x.shape) > 2:
        x = x.transpose(3, 2, 0, 1)
        return x.reshape(x.shape[0], -1)
    elif len(x.shape) == 1:
        return np.matrix(param).transpose()
    else:
        return x


defaultSolver = """
solver_mode: CPU
"""


def evaluateValue(val):
    if isinstance(val, int) or isinstance(val, float):
        return float(val)
    else:
        return K.eval(val)


def convertKerasToCaffeSolver(kerasModel, caffeNetworkFilePath, outCaffeSolverFilePath,
                              max_iter, test_iter, test_interval, display, lr_policy, weight_decay,
                              regularization_type):
    if isinstance(kerasModel.optimizer, keras.optimizers.SGD):
        solver = 'type: "Nesterov"\n' if kerasModel.optimizer.nesterov else 'type: "SGD"\n'
    elif isinstance(kerasModel.optimizer, keras.optimizers.Adagrad):
        solver = 'type: "Adagrad"\n'
    elif isinstance(kerasModel.optimizer, keras.optimizers.Adam):
        solver = 'type: "Adam"\n'
    else:
        raise Exception(
            'Only sgd (with/without momentum/nesterov), Adam and Adagrad supported.')
    base_lr = evaluateValue(
        kerasModel.optimizer.lr) if hasattr(
        kerasModel.optimizer,
        'lr') else 0.01
    gamma = evaluateValue(
        kerasModel.optimizer.decay) if hasattr(
        kerasModel.optimizer,
        'decay') else 0.0
    with open(outCaffeSolverFilePath, 'w') as f:
        f.write('net: "' + caffeNetworkFilePath + '"\n')
        f.write(defaultSolver)
        f.write(solver)
        f.write('lr_policy: "' + lr_policy + '"\n')
        f.write('regularization_type: "' + str(regularization_type) + '"\n')
        f.write('weight_decay: ' + str(weight_decay) + '\n')
        f.write(
            'max_iter: ' +
            str(max_iter) +
            '\ntest_iter: ' +
            str(test_iter) +
            '\ntest_interval: ' +
            str(test_interval) +
            '\n')
        f.write('display: ' + str(display) + '\n')
        f.write('base_lr: ' + str(base_lr) + '\n')
        f.write('gamma: ' + str(gamma) + '\n')
        if isinstance(kerasModel.optimizer, keras.optimizers.SGD):
            momentum = evaluateValue(
                kerasModel.optimizer.momentum) if hasattr(
                kerasModel.optimizer,
                'momentum') else 0.0
            f.write('momentum: ' + str(momentum) + '\n')
        elif isinstance(kerasModel.optimizer, keras.optimizers.Adam):
            momentum = evaluateValue(
                kerasModel.optimizer.beta_1) if hasattr(
                kerasModel.optimizer,
                'beta_1') else 0.9
            momentum2 = evaluateValue(
                kerasModel.optimizer.beta_2) if hasattr(
                kerasModel.optimizer,
                'beta_2') else 0.999
            delta = evaluateValue(
                kerasModel.optimizer.epsilon) if hasattr(
                kerasModel.optimizer,
                'epsilon') else 1e-8
            f.write('momentum: ' + str(momentum) + '\n')
            f.write('momentum2: ' + str(momentum2) + '\n')
            f.write('delta: ' + str(delta) + '\n')
        elif isinstance(kerasModel.optimizer, keras.optimizers.Adagrad):
            delta = evaluateValue(
                kerasModel.optimizer.epsilon) if hasattr(
                kerasModel.optimizer,
                'epsilon') else 1e-8
            f.write('delta: ' + str(delta) + '\n')
        else:
            raise Exception(
                'Only sgd (with/without momentum/nesterov), Adam and Adagrad supported.')

def getInputMatrices(layer):
    if isinstance(layer, keras.layers.SimpleRNN):
        weights = layer.get_weights()
        return [np.vstack((weights[0], weights[1])), np.matrix(weights[2])]
    elif isinstance(layer, keras.layers.LSTM):
        weights = layer.get_weights()
        W, U, b =  weights[0], weights[1], weights[2]
        units = int(W.shape[1]/4)
        if W.shape[1] != U.shape[1]:
            raise Exception('Number of hidden units of the kernel and the recurrent kernel doesnot match')
        # Note: For the LSTM layer, Keras weights are laid out in [i, f, c, o] format;
        # whereas SystemML weights are laid out in [i, f, o, c] format.
        W_i = W[:, :units]
        W_f = W[:, units: units * 2]
        W_c = W[:, units * 2: units * 3]
        W_o = W[:, units * 3:]
        U_i = U[:, :units]
        U_f = U[:, units: units * 2]
        U_c = U[:, units * 2: units * 3]
        U_o = U[:, units * 3:]
        b_i = b[:units]
        b_f = b[units: units * 2]
        b_c = b[units * 2: units * 3]
        b_o = b[units * 3:]
        return [np.vstack((np.hstack((W_i, W_f, W_o, W_c)), np.hstack((U_i, U_f, U_o, U_c)))).reshape((-1, 4*units)), np.hstack((b_i, b_f, b_o, b_c)).reshape((1, -1))]
    elif isinstance(layer, keras.layers.Conv2D):
        weights = layer.get_weights()
        #filter = np.swapaxes(weights[0].T, 2, 3)  # convert RSCK => KCRS format
        filter = np.swapaxes(np.swapaxes(np.swapaxes(weights[0], 1, 3), 0, 1), 1, 2)
        return [ filter.reshape((filter.shape[0], -1)) , getNumPyMatrixFromKerasWeight(weights[1])]
    else:
        return [getNumPyMatrixFromKerasWeight(
            param) for param in layer.get_weights()]


def convertKerasToSystemMLModel(spark, kerasModel, outDirectory):
    _checkIfValid(
        kerasModel.layers,
        lambda layer: False if len(
            layer.get_weights()) <= 4 or len(
            layer.get_weights()) != 3 else True,
        'Unsupported number of weights:')
    layers = [
        layer for layer in kerasModel.layers if len(
            layer.get_weights()) > 0]
    sc = spark._sc
    biasToTranspose = [keras.layers.Dense]
    dmlLines = []
    script_java = sc._jvm.org.apache.sysml.api.mlcontext.ScriptFactory.dml('')
    for layer in layers:
        inputMatrices = getInputMatrices(layer)
        potentialVar = [
            layer.name + '_weight',
            layer.name + '_bias',
            layer.name + '_1_weight',
            layer.name + '_1_bias']
        for i in range(len(inputMatrices)):
            dmlLines = dmlLines + \
                       ['write(' + potentialVar[i] + ', "' + outDirectory +
                        '/' + potentialVar[i] + '.mtx", format="binary");\n']
            mat = inputMatrices[i].transpose() if (
                    i == 1 and type(layer) in biasToTranspose) else inputMatrices[i]
            py4j.java_gateway.get_method(script_java, "in")(
                potentialVar[i], convertToMatrixBlock(sc, mat))
    script_str = ''.join(dmlLines)
    if script_str.strip() != '':
        # Only execute if the script is not empty
        script_java.setScriptString(script_str)
        ml = sc._jvm.org.apache.sysml.api.mlcontext.MLContext(sc._jsc)
        ml.execute(script_java)
