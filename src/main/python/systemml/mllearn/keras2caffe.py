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
import os, math
from itertools import chain, imap
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
    keras.layers.UpSampling2D: 'Upsample',
    keras.layers.MaxPooling2D: 'Pooling',
    keras.layers.AveragePooling2D: 'Pooling',
	keras.layers.SimpleRNN: 'RNN',
    keras.layers.LSTM: 'LSTM',
	keras.layers.Flatten: 'None',
    keras.layers.BatchNormalization: 'None',
    keras.layers.Activation: 'None'
    }

def _getInboundLayers(layer):
    in_names = []
    # get inbound nodes to current layer (support newer as well as older APIs)
    inbound_nodes = layer.inbound_nodes if hasattr(layer, 'inbound_nodes') else layer._inbound_nodes
    for node in inbound_nodes:
        node_list = node.inbound_layers  # get layers pointing to this node
        in_names = in_names + node_list
    # For Caffe2DML to reroute any use of Flatten layers
    return list(chain.from_iterable( [ _getInboundLayers(l) if isinstance(l, keras.layers.Flatten) else [ l ] for l in in_names ] ))

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


def _shouldParseActivation(layer):
    ignore_activation = [ keras.layers.SimpleRNN , keras.layers.LSTM ]
    return hasattr(layer, 'activation') and (type(layer) not in ignore_activation) and keras.activations.serialize(layer.activation) != 'linear'

def _parseKerasLayer(layer):
	layerType = type(layer)
	if layerType in specialLayers:
		return specialLayers[layerType](layer)
	elif layerType == keras.layers.Activation:
		return [ _parseActivation(layer) ]
	param = layerParamMapping[layerType](layer)
	paramName = param.keys()[0]
	if layerType == keras.layers.InputLayer:
		ret = { 'layer': { 'name':layer.name, 'type':'Data', paramName:param[paramName], 'top':layer.name, 'top':'label' } }
	else:
		ret = { 'layer': { 'name':layer.name, 'type':supportedLayers[layerType], 'bottom':_getBottomLayers(layer), 'top':layer.name, paramName:param[paramName] } }
	return [ ret, _parseActivation(layer, layer.name + '_activation') ] if _shouldParseActivation(layer)  else [ ret ]


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
	
def getConvParam(layer):
	stride = (1, 1) if layer.strides is None else layer.strides
	padding = [layer.kernel_size[0] / 2, layer.kernel_size[1] / 2] if layer.padding == 'same' else [0, 0]
	config = layer.get_config()
	return {'num_output':layer.filters,'bias_term':str(config['use_bias']).lower(),'kernel_h':layer.kernel_size[0], 'kernel_w':layer.kernel_size[1], 'stride_h':stride[0],'stride_w':stride[1],'pad_h':padding[0], 'pad_w':padding[1]}

def getUpSamplingParam(layer):
	return { 'size_h':layer.size[0], 'size_w':layer.size[1] }

def getPoolingParam(layer, pool='MAX'):
	stride = (1, 1) if layer.strides is None else layer.strides
	padding = [layer.pool_size[0] / 2, layer.pool_size[1] / 2] if layer.padding == 'same' else [0, 0]
	return {'pool':pool, 'kernel_h':layer.pool_size[0], 'kernel_w':layer.pool_size[1], 'stride_h':stride[0],'stride_w':stride[1],'pad_h':padding[0], 'pad_w':padding[1]}

def getRecurrentParam(layer):
	if(not layer.use_bias):
		raise Exception('Only use_bias=True supported for recurrent layers')
	if(keras.activations.serialize(layer.activation) != 'tanh'):
		raise Exception('Only tanh activation supported for recurrent layers')
	if(layer.dropout != 0 or layer.recurrent_dropout != 0):
		raise Exception('Only dropout not supported for recurrent layers')
	return {'num_output': layer.units, 'return_sequences': str(layer.return_sequences).lower() }

# TODO: Update AveragePooling2D when we add maxpooling support 
layerParamMapping = {
    keras.layers.InputLayer: lambda l: \
        {'data_param': {'batch_size': l.batch_size}},
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
    keras.layers.UpSampling2D: lambda l: \
        {'upsample_param': getUpSamplingParam(l)},
    keras.layers.Conv2D: lambda l: \
        {'convolution_param': getConvParam(l)},
    keras.layers.MaxPooling2D: lambda l: \
        {'pooling_param': getPoolingParam(l, 'MAX')},
    keras.layers.AveragePooling2D: lambda l: \
        {'pooling_param': getPoolingParam(l, 'AVE')},
    keras.layers.SimpleRNN: lambda l: \
        {'recurrent_param': getRecurrentParam(l)},
    keras.layers.LSTM: lambda l: \
        {'recurrent_param': getRecurrentParam(l)},
    }

def _checkIfValid(myList, fn, errorMessage):
	bool_vals = np.array([ fn(elem) for elem in myList])
	unsupported_elems = np.where(bool_vals)[0]
	if len(unsupported_elems) != 0:
		raise ValueError(errorMessage + str(np.array(myList)[unsupported_elems]))

def _transformLayer(layer, batch_size):
	if type(layer) == keras.layers.InputLayer:
		layer.batch_size = batch_size
	return [ layer ]

def _appendKerasLayers(fileHandle, kerasLayers, batch_size):
	if len(kerasLayers) >= 1:
		transformedLayers = list(chain.from_iterable(imap(lambda layer: _transformLayer(layer, batch_size), kerasLayers)))  
		jsonLayers = list(chain.from_iterable(imap(lambda layer: _parseKerasLayer(layer), transformedLayers)))
		parsedLayers = list(chain.from_iterable(imap(lambda layer: _parseJSONObject(layer), jsonLayers)))
		fileHandle.write(''.join(parsedLayers))
		fileHandle.write('\n')
	
def lossLayerStr(layerType, bottomLayer):
	return 'layer {\n  name: "loss"\n  type: "' + layerType + '"\n  bottom: "' + bottomLayer + '"\n  bottom: "label"\n  top: "loss"\n}\n'
	
def _appendKerasLayerWithoutActivation(fileHandle, layer, batch_size):
	if type(layer) != keras.layers.Activation:
		lastLayerActivation = layer.activation
		layer.activation = keras.activations.linear
		_appendKerasLayers(fileHandle, [layer], batch_size)
		layer.activation = lastLayerActivation

def _getExactlyOneBottomLayer(layer):
	bottomLayers = _getBottomLayers(layer)
	if len(bottomLayers) != 1:
		raise Exception('Expected only one bottom layer for ' + str(layer.name) + ', but found ' + str(bottomLayers))
	return bottomLayers[0]

def _isMeanSquaredError(loss):
	return loss == 'mean_squared_error' or loss == 'mse' or loss == 'MSE' 
	
def convertKerasToCaffeNetwork(kerasModel, outCaffeNetworkFilePath, batch_size):
	_checkIfValid(kerasModel.layers, lambda layer: False if type(layer) in supportedLayers else True, 'Unsupported Layers:')
	with open(outCaffeNetworkFilePath, 'w') as f:
		# Write the parsed layers for all but the last layer
		_appendKerasLayers(f, kerasModel.layers[:-1], batch_size)
		# Now process the last layer with loss
		lastLayer = kerasModel.layers[-1]
		if _isMeanSquaredError(kerasModel.loss):
			_appendKerasLayers(f, [ lastLayer ], batch_size)
			f.write(lossLayerStr('EuclideanLoss', lastLayer.name))
		elif kerasModel.loss == 'categorical_crossentropy':
			_appendKerasLayerWithoutActivation(f, lastLayer, batch_size)
			bottomLayer = _getExactlyOneBottomLayer(lastLayer) if type(lastLayer) == keras.layers.Activation else lastLayer.name  
			lastLayerActivation = str(keras.activations.serialize(lastLayer.activation))
			if lastLayerActivation == 'softmax' and kerasModel.loss == 'categorical_crossentropy':
				f.write(lossLayerStr('SoftmaxWithLoss', bottomLayer))
			else:
				raise Exception('Unsupported loss layer ' + str(kerasModel.loss) + ' (where last layer activation ' + lastLayerActivation + ').')
		else:
			raise Exception('Unsupported loss layer ' + str(kerasModel.loss) + ' (where last layer activation ' + lastLayerActivation + ').')


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
	if type(val) == int or type(val) == float:
		return float(val)
	else:
		return K.eval(val)
	
def convertKerasToCaffeSolver(kerasModel, caffeNetworkFilePath, outCaffeSolverFilePath, max_iter, test_iter, test_interval, display, lr_policy, weight_decay, regularization_type):
	if type(kerasModel.optimizer) == keras.optimizers.SGD:
		solver = 'type: "Nesterov"\n' if kerasModel.optimizer.nesterov else 'type: "SGD"\n'
	elif type(kerasModel.optimizer) == keras.optimizers.Adagrad:
		solver = 'type: "Adagrad"\n'
	elif type(kerasModel.optimizer) == keras.optimizers.Adam:
		solver = 'type: "Adam"\n'
	else:
		raise Exception('Only sgd (with/without momentum/nesterov), Adam and Adagrad supported.')
	base_lr = evaluateValue(kerasModel.optimizer.lr) if hasattr(kerasModel.optimizer, 'lr') else 0.01
	gamma = evaluateValue(kerasModel.optimizer.decay) if hasattr(kerasModel.optimizer, 'decay') else 0.0
	with open(outCaffeSolverFilePath, 'w') as f:
		f.write('net: "' + caffeNetworkFilePath + '"\n')
		f.write(defaultSolver)
		f.write(solver)
		f.write('lr_policy: "' + lr_policy + '"\n')
		f.write('regularization_type: "' + str(regularization_type) + '"\n')
		f.write('weight_decay: ' + str(weight_decay) + '\n')
		f.write('max_iter: ' + str(max_iter) + '\ntest_iter: ' + str(test_iter) + '\ntest_interval: ' + str(test_interval) + '\n')
		f.write('display: ' + str(display) + '\n')
		f.write('base_lr: ' + str(base_lr) + '\n')
		f.write('gamma: ' + str(gamma) + '\n')
		if type(kerasModel.optimizer) == keras.optimizers.SGD:
			momentum = evaluateValue(kerasModel.optimizer.momentum) if hasattr(kerasModel.optimizer, 'momentum') else 0.0
			f.write('momentum: ' + str(momentum) + '\n')
		elif type(kerasModel.optimizer) == keras.optimizers.Adam:
			momentum = evaluateValue(kerasModel.optimizer.beta_1) if hasattr(kerasModel.optimizer, 'beta_1') else 0.9
			momentum2 = evaluateValue(kerasModel.optimizer.beta_2) if hasattr(kerasModel.optimizer, 'beta_2') else 0.999
			delta = evaluateValue(kerasModel.optimizer.epsilon) if hasattr(kerasModel.optimizer, 'epsilon') else 1e-8
			f.write('momentum: ' + str(momentum) + '\n')
			f.write('momentum2: ' + str(momentum2) + '\n')
			f.write('delta: ' + str(delta) + '\n')
		elif type(kerasModel.optimizer) == keras.optimizers.Adagrad:
			delta = evaluateValue(kerasModel.optimizer.epsilon) if hasattr(kerasModel.optimizer, 'epsilon') else 1e-8
			f.write('delta: ' + str(delta) + '\n')
		else:
			raise Exception('Only sgd (with/without momentum/nesterov), Adam and Adagrad supported.')


def getInputMatrices(layer):
	if type(layer) == keras.layers.LSTM or type(layer) == keras.layers.SimpleRNN:
		weights = layer.get_weights()
		return [np.vstack((weights[0], weights[1])), np.matrix(weights[2]) ]
	else:
		return [ getNumPyMatrixFromKerasWeight(param) for param in layer.get_weights() ]

def convertKerasToSystemMLModel(spark, kerasModel, outDirectory):
	_checkIfValid(kerasModel.layers, lambda layer: False if len(layer.get_weights()) <= 4 or len(layer.get_weights()) != 3 else True, 'Unsupported number of weights:')
	layers = [layer for layer in kerasModel.layers if len(layer.get_weights()) > 0]
	sc = spark._sc
	biasToTranspose = [ keras.layers.Dense ]
	dmlLines = []
	script_java = sc._jvm.org.apache.sysml.api.mlcontext.ScriptFactory.dml('')
	for layer in layers:
		inputMatrices = getInputMatrices(layer)
		potentialVar = [ layer.name + '_weight', layer.name + '_bias',  layer.name + '_1_weight', layer.name + '_1_bias' ]
		for i in range(len(inputMatrices)):
			dmlLines = dmlLines + [ 'write(' + potentialVar[i] + ', "' + outDirectory + '/' + potentialVar[i] + '.mtx", format="binary");\n' ]
			mat = inputMatrices[i].transpose() if (i == 1 and type(layer) in biasToTranspose) else inputMatrices[i]
			py4j.java_gateway.get_method(script_java, "in")(potentialVar[i], convertToMatrixBlock(sc, mat))
	script_java.setScriptString(''.join(dmlLines))
	ml = sc._jvm.org.apache.sysml.api.mlcontext.MLContext(sc._jsc)
	ml.execute(script_java)
