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
import keras
from itertools import chain, imap

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


def _parseJSONObject(obj):
	rootName = obj.keys()[0]
	ret = ['\n', rootName, ' {']
	for key in obj[rootName]:
		if isinstance(obj[rootName][key], dict):
			ret = ret + [ '\n\t', key, ' {' ]
			for key1 in obj[rootName][key]:
				ret = ret + [ '\n\t\t', key1, ': ', str(obj[rootName][key][key1]) ]
			ret = ret + [ '\n\t', '}' ]
		elif isinstance(obj[rootName][key], list):
			for v in obj[rootName][key]:
				ret = ret + ['\n\t', key, ': ', str(v) ]
		else:
			ret = ret + ['\n\t', key, ': ', str(obj[rootName][key]) ]
	return ret + ['\n}' ]
	

def _parseActivation(layer):
	kerasActivation = keras.activations.serialize(layer.activation)
	if kerasActivation not in supportedCaffeActivations:
		raise TypeError('Unsupported activation ' + kerasActivation + ' for the layer:' + kerasLayer.name)
	return { 'layer':{'name':layer.name, 'type':supportedCaffeActivations[kerasActivation], 'top':layer.name, 'bottom':layer.name }}


def _getBottomLayers(layer):
    return [ bottomLayer.name for bottomLayer in _getInboundLayers(layer) ]


def _parseKerasLayer(layer):
	layerType = type(layer)
	if layerType in specialLayers:
		return specialLayers[layerType](layerType)
	elif layerType == keras.layers.Activation:
		return [ _parseActivation(layer) ]
	param = layerParamMapping[layerType](layer)
	paramName = param.keys()[0]
	if layerType == keras.layers.InputLayer:
		ret = { 'layer': { 'name':layer.name, 'type':'Data', 'top':layer.name, paramName:param[paramName] } }
	else:
		ret = { 'layer': { 'name':layer.name, 'type':supportedLayers[layerType], 'bottom':_getBottomLayers(layer), 'top':layer.name, paramName:param[paramName] } }
	return [ ret, _parseActivation(layer) ] if hasattr(layer, 'activation') else [ ret ]


def _parseBatchNorm(layer):
	bnName = layer.name + '_batchNorm'
	return [ { 'layer': { 'name':bnName, 'type':'BatchNorm', 'bottom':_getBottomLayers(layer), 'top':bnName, 'batch_norm_param':{'moving_average_fraction':layer.momentum, 'eps':layer.epsilon} } }, { 'layer': { 'name':layer.name, 'type':'Scale', 'bottom':bnName, 'top':layer.name, 'scale_param':{'moving_average_fraction':layer.momentum, 'eps':layer.epsilon, 'bias_term':bias_term} } } ]

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
	padding = [layer.kernel_size[0] / 2, layer.kernel_size[1] / 2] if layer.padding == 'same' else [0, 0]
	return {'pool':pool, 'kernel_h':layer.pool_size[0], 'kernel_w':layer.pool_size[1], 'stride_h':stride[0],'stride_w':stride[1],'pad_h':padding[0], 'pad_w':padding[1]}


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
    }


def converKerasToCaffeNetwork(kerasModel, outCaffeNetworkFilePath):
	unsupported_layers = np.array([False if type(layer) in supportedLayers else True for layer in kerasModel.layers])
	if len(np.where(unsupported_layers)[0]) != 0:
		raise TypeError('Unsupported Layers:' + str(np.array(kerasModel.layers)[np.where(x)[0]]))
	# Core logic: model.layers.flatMap(layer => _parseJSONObject(_parseKerasLayer(layer)))
	jsonLayers = list(chain.from_iterable(imap(lambda layer: _parseKerasLayer(layer), kerasModel.layers)))
	parsedLayers = list(chain.from_iterable(imap(lambda layer: _parseJSONObject(layer), jsonLayers)))
	with open(outCaffeNetworkFilePath, 'w') as f:
		f.write(''.join(parsedLayers))