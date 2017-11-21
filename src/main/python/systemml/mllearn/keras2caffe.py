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
from itertools import chain, imap
from ..converters import *
from ..classloader import *
import keras

try:
    import py4j.java_gateway
    from py4j.java_gateway import JavaObject
    from pyspark import SparkContext
except ImportError:
    raise ImportError('Unable to import `pyspark`. Hint: Make sure you are running with PySpark.')

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
# --------------------------------------------------------------------------------------

supportedCaffeActivations = {'relu':'ReLU', 'softmax':'Softmax', 'sigmoid':'Sigmoid' }
supportedLayers = {
    keras.layers.InputLayer: 'Data',
    keras.layers.Dense: 'InnerProduct',
    keras.layers.Dropout: 'Dropout',
    keras.layers.Add: 'Eltwise',
    keras.layers.Concatenate: 'Concat',
    keras.layers.Conv2DTranspose: 'Deconvolution',
    keras.layers.Conv2D: 'Convolution',
    keras.layers.MaxPooling2D: 'Pooling',
    keras.layers.AveragePooling2D: 'Pooling',
	keras.layers.Flatten: 'None',
    keras.layers.BatchNormalization: 'None',
    keras.layers.Activation: 'None'
    }

def _getInboundLayers(layer):
    in_names = []
    for node in layer.inbound_nodes:  # get inbound nodes to current layer
        node_list = node.inbound_layers  # get layers pointing to this node
        in_names = in_names + node_list
    if any('flat' in s.name for s in in_names):  # For Caffe2DML to reroute any use of Flatten layers
        return _getInboundLayers([s for s in in_names if 'flat' in s.name][0])
    return in_names


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

str_keys = [ 'name', 'type', 'top', 'bottom' ]
def toKV(key, value):
	return str(key) + ': "' + str(value) + '"' if key in str_keys else str(key) + ': ' + str(value)
	

def _parseJSONObject(obj):
	rootName = obj.keys()[0]
	ret = ['\n', rootName, ' {']
	for key in obj[rootName]:
		if isinstance(obj[rootName][key], dict):
			ret = ret + [ '\n\t', key, ' {' ]
			for key1 in obj[rootName][key]:
				ret = ret + [ '\n\t\t', toKV(key1, obj[rootName][key][key1]) ]
			ret = ret + [ '\n\t', '}' ]
		elif isinstance(obj[rootName][key], list):
			for v in obj[rootName][key]:
				ret = ret + ['\n\t', toKV(key, v) ]
		else:
			ret = ret + ['\n\t', toKV(key, obj[rootName][key]) ]
	return ret + ['\n}' ]
	

def _getBottomLayers(layer):
    return [ bottomLayer.name for bottomLayer in _getInboundLayers(layer) ]


def _parseActivation(layer, customLayerName=None):
	kerasActivation = keras.activations.serialize(layer.activation)
	if kerasActivation not in supportedCaffeActivations:
		raise TypeError('Unsupported activation ' + kerasActivation + ' for the layer:' + layer.name)
	if customLayerName is not None:
		return { 'layer':{'name':customLayerName, 'type':supportedCaffeActivations[kerasActivation], 'top':layer.name, 'bottom':layer.name }}
	else:
		return { 'layer':{'name':layer.name, 'type':supportedCaffeActivations[kerasActivation], 'top':layer.name, 'bottom':_getBottomLayers(layer) }}



def _parseKerasLayer(layer):
	layerType = type(layer)
	if layerType in specialLayers:
		return specialLayers[layerType](layer)
	elif layerType == keras.layers.Activation:
		return [ _parseActivation(layer) ]
	param = layerParamMapping[layerType](layer)
	paramName = param.keys()[0]
	if layerType == keras.layers.InputLayer:
		ret = { 'layer': { 'name':layer.name, 'type':'Data', 'top':layer.name, paramName:param[paramName] } }
	else:
		ret = { 'layer': { 'name':layer.name, 'type':supportedLayers[layerType], 'bottom':_getBottomLayers(layer), 'top':layer.name, paramName:param[paramName] } }
	return [ ret, _parseActivation(layer, layer.name + '_activation') ] if hasattr(layer, 'activation') and keras.activations.serialize(layer.activation) != 'linear'  else [ ret ]


def _parseBatchNorm(layer):
	bnName = layer.name + '_1'
	config = layer.get_config()
	bias_term = 'true' if config['center'] else 'false'
	return [ { 'layer': { 'name':bnName, 'type':'BatchNorm', 'bottom':_getBottomLayers(layer), 'top':bnName, 'batch_norm_param':{'moving_average_fraction':layer.momentum, 'eps':layer.epsilon} } }, { 'layer': { 'name':layer.name, 'type':'Scale', 'bottom':bnName, 'top':layer.name, 'scale_param':{'bias_term':bias_term} } } ]

# The special are redirected to their custom parse function in _parseKerasLayer
specialLayers = {
    keras.layers.Flatten: lambda x: [],
    keras.layers.BatchNormalization: _parseBatchNorm
    }
	
batchSize = 64

def getConvParam(layer):
	stride = (1, 1) if layer.strides is None else layer.strides
	padding = [layer.kernel_size[0] / 2, layer.kernel_size[1] / 2] if layer.padding == 'same' else [0, 0]
	config = layer.get_config()
	return {'num_output':layer.filters,'bias_term':str(config['use_bias']).lower(),'kernel_h':layer.kernel_size[0], 'kernel_w':layer.kernel_size[1], 'stride_h':stride[0],'stride_w':stride[1],'pad_h':padding[0], 'pad_w':padding[1]}


def getPoolingParam(layer, pool='MAX'):
	stride = (1, 1) if layer.strides is None else layer.strides
	padding = [layer.pool_size[0] / 2, layer.pool_size[1] / 2] if layer.padding == 'same' else [0, 0]
	return {'pool':pool, 'kernel_h':layer.pool_size[0], 'kernel_w':layer.pool_size[1], 'stride_h':stride[0],'stride_w':stride[1],'pad_h':padding[0], 'pad_w':padding[1]}

# TODO: Update AveragePooling2D when we add maxpooling support 
layerParamMapping = {
    keras.layers.InputLayer: lambda l: \
        {'data_param': {'batch_size': batchSize}},
    keras.layers.Dense: lambda l: \
        {'inner_product_param': {'num_output': l.units}},
    keras.layers.Dropout: lambda l: \
        {'dropout_param': {'dropout_ratio': l.rate}},
    keras.layers.Add: lambda l: \
        {'eltwise_param': {'operation': 'SUM'}},
    keras.layers.Concatenate: lambda l: \
        {'concat_param': {'axis': _getCompensatedAxis(l)}},
    keras.layers.Conv2DTranspose: lambda l: \
        {'convolution_param': getConvParam(l)},
    keras.layers.Conv2D: lambda l: \
        {'convolution_param': getConvParam(l)},
    keras.layers.MaxPooling2D: lambda l: \
        {'pooling_param': getPoolingParam(l, 'MAX')},
    keras.layers.AveragePooling2D: lambda l: \
        {'pooling_param': getPoolingParam(l, 'MAX')},
    }

def _checkIfValid(myList, fn, errorMessage):
	bool_vals = np.array([ fn(elem) for elem in myList])
	unsupported_elems = np.where(bool_vals)[0]
	if len(unsupported_elems) != 0:
		raise ValueError(errorMessage + str(np.array(myList)[unsupported_elems]))

def convertKerasToCaffeNetwork(kerasModel, outCaffeNetworkFilePath):
	_checkIfValid(kerasModel.layers, lambda layer: False if type(layer) in supportedLayers else True, 'Unsupported Layers:')
	#unsupported_layers = np.array([False if type(layer) in supportedLayers else True for layer in kerasModel.layers])
	#if len(np.where(unsupported_layers)[0]) != 0:
	#	raise TypeError('Unsupported Layers:' + str(np.array(kerasModel.layers)[np.where(unsupported_layers)[0]]))
	# Core logic: model.layers.flatMap(layer => _parseJSONObject(_parseKerasLayer(layer)))
	jsonLayers = list(chain.from_iterable(imap(lambda layer: _parseKerasLayer(layer), kerasModel.layers)))
	parsedLayers = list(chain.from_iterable(imap(lambda layer: _parseJSONObject(layer), jsonLayers)))
	with open(outCaffeNetworkFilePath, 'w') as f:
		f.write(''.join(parsedLayers))


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
base_lr: 0.01
momentum: 0.9
weight_decay: 5e-4
lr_policy: "exp"
gamma: 0.95
display: 100
solver_mode: CPU
type: "SGD"
max_iter: 2000
test_iter: 10
test_interval: 500
"""

def convertKerasToCaffeSolver(kerasModel, caffeNetworkFilePath, outCaffeSolverFilePath):
	with open(outCaffeSolverFilePath, 'w') as f:
		f.write('net: "' + caffeNetworkFilePath + '"\n')
		f.write(defaultSolver)


def convertKerasToSystemMLModel(spark, kerasModel, outDirectory):
	_checkIfValid(kerasModel.layers, lambda layer: False if len(layer.get_weights()) <= 4 or len(layer.get_weights()) != 3 else True, 'Unsupported number of weights:')
	layers = [layer for layer in kerasModel.layers if len(layer.get_weights()) > 0]
	sc = spark._sc
	biasToTranspose = [ keras.layers.Dense ]
	dmlLines = []
	script_java = sc._jvm.org.apache.sysml.api.mlcontext.ScriptFactory.dml('')
	for layer in layers:
		inputMatrices = [ getNumPyMatrixFromKerasWeight(param) for param in layer.get_weights() ]
		potentialVar = [ layer.name + '_weight', layer.name + '_bias',  layer.name + '_1_weight', layer.name + '_1_bias' ]
		for i in range(len(inputMatrices)):
			dmlLines = dmlLines + [ 'write(' + potentialVar[i] + ', "' + outDirectory + '/' + potentialVar[i] + '.mtx", format="binary");\n' ]
			mat = inputMatrices[i].transpose() if (i == 1 and type(layer) in biasToTranspose) else inputMatrices[i]
			py4j.java_gateway.get_method(script_java, "in")(potentialVar[i], convertToMatrixBlock(sc, mat))
	script_java.setScriptString(''.join(dmlLines))
	ml = sc._jvm.org.apache.sysml.api.mlcontext.MLContext(sc._jsc)
	ml.execute(script_java)
