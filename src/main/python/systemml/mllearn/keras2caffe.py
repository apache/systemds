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


#Script to generate caffe proto and .caffemodel files from Keras models
from caffe import *
import caffe
from caffe import layers as L
from caffe import params as P



import keras
from keras.models import load_model
from keras.models import model_from_json

import numpy as np


def load_keras_model(filepath):
    model = load_model(filepath)
    return model


def load_keras_skeleton_model(filepath):
    json_file = open(filepath, 'r')
    model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(model_json)
    return loaded_model


def load_weights_to_model(model,filepath):
    model.load_weights(filepath)
    return model


#Currently can only generate a Dense model
def generate_caffe_model(kModel, input_shape=None, phases=None):
    n = caffe.NetSpec()
    layers = kModel.layers

    for layer in layers:
        if type(layer) == keras.layers.InputLayer:
            #Grab the batchsize from i 0, shift over channels to index 1, and place the rest into the dictionary
            #TODO determine when to transform for layer types/input shape
            num = len(layer.batch_input_shape) - 1 #Range from 1st index to second last
            #TODO check for image_data_format to be channels_first or channels_last
            batch_list = [layer.batch_input_shape[0], layer.batch_input_shape[-1]]
            for i  in range(1 ,num):
                batch_list.append(layer.batch_input_shape[i])
            for i in range(len(batch_list)): #Set None dimensions to 0 for Caffe
                    if( batch_list[i] == None):
                        batch_list[i] = 0
                        #print("Batch size of InputLayer: ")
                        #print(batch_list[i])
            name = layer.name
            #TODO figure out having 2 tops, with n.label
            n[name] = L.Input(shape=[dict(dim=batch_list)])
            #print(name)
        elif type(layer) == keras.layers.Dense:
            #Pull name from Keras
            name = layer.name
            #Pull layer name of the layer passing to current layer
            in_names = getInboundLayers(layer)
            #Pipe names into caffe using unique Keras layer names
            #print("Current Keras layer" + layer.name + " Input Layer:") #debug to assure there are a correct number of layer names (1)
            #print(in_names)
            #print(in_names[0].name)
            n[name] = L.InnerProduct(n[in_names[0].name], num_output=layer.units) #TODO: Assert only 1
            if layer.activation is not None and layer.activation.__name__ != 'linear':
                name_act = name + "_activation_" + layer.activation.__name__ #get function string
                n[name_act] = getActivation(layer,n[name])
        elif type(layer) == keras.layers.Flatten:
            name = layer.name
            in_names= getInboundLayers(layer)
            n[name] = L.Flatten(n[in_names[0].name])

        elif type(layer) == keras.layers.Dropout:#TODO Random seed will be lost
            name = layer.name
            in_names= getInboundLayers(layer)
            n[name] = L.Dropout(n[in_names[0].name], dropout_ratio= layer.rate, in_place=True)
        #elif type(layer) == keras.Layers.LSTM:
        elif type(layer) == keras.layers.Add:
            name = layer.name
            in_names= getInboundLayers(layer)
            #turn list of names into network layers
            network_layers = []
            for ref in in_names:
                network_layers.append(n[ref.name])
            print(network_layers)
            #unpack the bottom layers
            n[name] = L.Eltwise(*network_layers, operation=1) #1 is SUM
        elif type(layer) == keras.layers.Multiply:
            name = layer.name
            in_names= getInboundLayers(layer)
            #turn list of names into network layers
            network_layers = []
            for ref in in_names:
                network_layers += n[ref.name]
            #unpack the bottom layers
            n[name] = L.Eltwise(*network_layers, operation=0)
        elif type(layer) == keras.layers.Concatenate:
            name = layer.name
            in_names= getInboundLayers(layer)
            #turn list of names into network layers
            network_layers = []
            for ref in in_names:
                network_layers += n[ref.name]
            axis = getCompensatedAxis(layer)
            n[name] = L.Concat(*network_layers, axis=axis)
        elif type(layer) == keras.layers.Maximum:
            name = layer.name
            in_names= getInboundLayers(layer)
            #turn list of names into network layers
            network_layers = []
            for ref in in_names:
                network_layers += n[ref.name]
            #unpack the bottom layers
            n[name] = L.Eltwise(*network_layers, operation=2)
        elif type(layer) == keras.layers.Conv2DTranspose:
            name = layer.name
            in_names= getInboundLayers(layer)
            #Stride
            if layer.strides is None:
                stride = (1,1)
            else:
                stride= layer.strides
            #Padding
            padding = [0,0] #If padding is valid(aka no padding)
            if layer.padding == 'same': #Calculate the padding for 'same'
                padding[0] = (layer.input_shape[0]*(stride[0] -1) + layer.kernel_size[0] - stride[0]) / 2

                padding[1] = (layer.input_shape[1]*(stride[1] -1) + layer.kernel_size[1] - stride[1]) / 2
            
            #TODO The rest of the arguements including bias, regulizers, dilation,
            
            n[name] = L.Deconvolution(n[in_names[0].name],kernel_h=layer.kernel_size[0],
                                    kernel_w=layer.kernel_size[1],stride_h=stride[0],
                                    stride_w=stride[-1], num_output=layer.filters, pad_h=padding[0], pad_w=padding[1])
            if layer.activation is not None and layer.activation.__name__ !='linear':
                name_act = name + "_activation_" + layer.activation.__name__ #get function string
                n[name_act] = getActivation(layer,n[name])
        elif type(layer) == keras.layers.BatchNormalization:
            name = layer.name
            in_names[0]=getInboundLayers(layer)
            n[name] = L.BatchNorm(n[in_names[0].name], moving_average_fraction= layer.momentum, eps = layer.epsilon)
        elif type(layer) == keras.layers.Conv1D:
            name = layer.name
            in_names[0]= getInboundLayers(layer)
            n[name] = L.Convolution()
        elif type(layer) == keras.layers.Conv2D:
            name = layer.name
            in_names= getInboundLayers(layer)
            #Stride
            if layer.strides is None:
                stride = (1,1)
            else:
                stride= layer.strides
            #Padding
            padding = [0,0] #If padding is valid(aka no padding)
            if layer.padding == 'same': #Calculate the padding for 'same'
                padding[0] = (layer.input_shape[0]*(stride[0] -1) + layer.kernel_size[0] - stride[0]) / 2

                padding[1] = (layer.input_shape[1]*(stride[1] -1) + layer.kernel_size[1] - stride[1]) / 2
            
            #TODO The rest of the arguements including bias, regulizers, dilation,
            
            n[name] = L.Convolution(n[in_names[0].name],kernel_h=layer.kernel_size[0],
                                    kernel_w=layer.kernel_size[1],stride_h=stride[0],
                                    stride_w=stride[-1], num_output=layer.filters, pad_h=padding[0], pad_w=padding[1])
            if layer.activation is not None and layer.activation.__name__ != 'linear':
                name_act = name + "_activation_" + layer.activation.__name__ #get function string
                n[name_act] = getActivation(layer,n[name])
        #elif type(layer) == keras.Layers.MaxPooling1D:

        elif type(layer) == keras.layers.MaxPooling2D:
            name = layer.name
            in_names= getInboundLayers(layer)
            #Padding
            #TODO The rest of the arguements including bias, regulizers, dilatin,
            if layer.strides is None:
                stride = (1,1)
            else:
                stride= layer.strides
            #Padding
            padding = [0,0] #If padding is valid(aka no padding)
            if layer.padding == 'same': #Calculate the padding for 'same'
                padding[0] = (layer.input_shape[0]*(stride[0] -1) + layer.pool_size[0] - stride[0]) / 2

                padding[1] = (layer.input_shape[1]*(stride[1] -1) + layer.pool_size[1] - stride[1]) / 2
            n[name] = L.Pooling(n[in_names[0].name],kernel_h=layer.pool_size[0],
                                    kernel_w=layer.pool_size[1],stride_h=stride[0],
                                    stride_w= stride[-1],pad_h = padding[0],pad_w= padding[1])
            #if layer.activation is not None:
                #name_act = name + "_activation_" + layer.activation.__name__ #get function string
                #n[name_act] = getActivation(layer,n[name])
        #Activation (wrapper for activations) and Advanced Activation Layers
        elif type(layer) == keras.layers.Activation:
            name = layer.name
            in_names = getInboundLayers(layer)
            n[name] = getActivation(layer,n[in_names[0].name]) #TODO: Assert only 1
        #Caffe lacks intializer, regulizer, and constraint params
        elif type(layer) == keras.layers.LeakyReLU:
            #TODO: figure out how to pass Leaky params
            name = layer.name
            in_names = getInboundLayers(layer)
            n[name] = L.PReLU(n[in_names[0].name])
        elif type(layer) == keras.layers.PReLU:
            name = layer.name
            in_names = getInboundLayers(layer)
            n[name] = L.PReLU(n[in_names[0].name])
        elif type(layer) == keras.layers.ELU:
            name = layer.name
            in_names = getInboundLayers(layer)
            n[name] = L.ELU(n[in_names[0].name],layer.alpha)
        else:
            RaiseError("Cannot convert model. " + layer.name + " is not supported.")

    #Determine the loss needed to be added
    for output in kModel.output_layers:
        if kModel.loss == 'categorical_crossentropy' and output.activation.__name__ == 'softmax':
            name = output.name + "_activation_" + output.activation.__name__
            n[name] = L.SoftmaxWithLoss(n[output.name])
        elif kModel.loss == 'binary_crossentropy' and output.activation.__name__ == 'sigmoid':
            name = output.name + "_activation_" + output.activation.__name__
            n[name] = L.SigmoidCrossEntropyLoss(n[output.name])
        else: #Map the rest of the loss functions to the end of the output layer in Keras
            if kModel.loss == 'hinge':
                name = kModel.name + 'hinge'
                n[name] = L.HingeLoss(n[output.name])
            elif kModel.loss == 'categorical_crossentropy':
                name = kModel.name + 'categorical_crossentropy'
                n[name] = L.MultinomialLogisticLoss(n[output.name])
                #TODO Post warning to use softmax before this loss
            elif kModel.loss == 'mean_squared_error':
                name = kModel.name + 'mean_squared_error'
                n[name] = L.EuclideanLoss(n[output.name])
            #TODO implement Infogain Loss
            else:
                RaiseError(kModel.loss + "is not supported")
    return n


def getInboundLayers(layer):
        in_names =[]
        for node in layer.inbound_nodes: #get inbound nodes to current layer
            node_list = node.inbound_layers #get layers pointing to this node
            in_names = in_names + node_list
        return in_names

#Only works with non Tensorflow functions! TODO
def getActivation(layer,bottom):
    if keras.activations.serialize(layer.activation) == 'relu':
        return L.relu(bottom, in_place=True)
    elif keras.activations.serialize(layer.activation) == 'softmax':
        return L.softmax(bottom) #Cannot extract axis from model, so default to -1
    elif keras.activations.serialize(layer.activation) == 'softsign':
        #Needs to be implemented in caffe2dml
        raise ValueError("softsign is not implemented")
    elif keras.activations.serialize(layer.activation) == 'elu':
        return L.ELU(bottom)
    elif keras.activations.serialize(layer.activation) == 'selu':
        #Needs to be implemented in caffe2dml 
        raise ValueError("SELU activation is not implemented")
    elif keras.activations.serialize(layer.activation) == 'sigmoid':
        return L.Sigmoid(bottom)
    elif keras.activations.serialize(layer.activation) == 'tanh':
        return L.TanH(bottom)
    #To add more acitvaiton functions, add more elif statements with
    #activation funciton __name__'s.

    
#def generate_caffe_weights(kModel):

#Params: keras Model, caffe prototxt filepath, filepath to save solver
def generate_caffe_solver(kModel,cModelPath,filepath):
    solver_param = CaffeSolver(trainnet_prototxt_path=cModelPath,
                               testnet_prototxt_path=cModelPath,
                               debug=True) #Currently train and test are the same protos
    solver_param.write(filepath)

#Params: NetSpec, filepath and filename
def write_caffe_model(cModel,filepath):
    with open(filepath, 'w') as f:
        f.write(str(cModel.to_proto()))


#Get compensated axis since Caffe has n,c,h,w and Keras has n,h,w,c for tensor dimensions
#Params: Current Keras layer
def getCompensatedAxis(layer):
    compensated_axis = layer.axis
    #Cover all cases for anything accessing the 0th index or the last index
    if layer.axis > 0 and layer.axis < layer.input[0].shape.ndims-1:
        compensated_axis = layer.axis + 1
    elif layer.axis < -1 and layer.axis > -(layer.input[0].shape.ndims):
        compensated_axis = layer.axis + 1
    elif layer.axis == -1 or layer.axis == layer.input[0].shape.ndims - 1:
        compensated_axis = 1
    return compensated_axis
#Copied from caffe repo in examples/pycaffe/tools.py
class CaffeSolver:

    """
    Caffesolver is a class for creating a solver.prototxt file. It sets default
    values and can export a solver parameter file.
    Note that all parameters are stored as strings. Strings variables are
    stored as strings in strings.
    """

    def __init__(self, testnet_prototxt_path="testnet.prototxt",
                 trainnet_prototxt_path="trainnet.prototxt", debug=False):

        self.sp = {}

        # critical:
        self.sp['base_lr'] = '0.001'
        self.sp['momentum'] = '0.9'
        self.sp['type'] = '"SGD"'

        # speed:
        self.sp['test_iter'] = '100'
        self.sp['test_interval'] = '250'

        # looks:
        self.sp['display'] = '25'
        self.sp['snapshot'] = '2500'
        self.sp['snapshot_prefix'] = '"snapshot"'  # string within a string!

        # learning rate policy
        self.sp['lr_policy'] = '"fixed"'

        # important, but rare:
        self.sp['gamma'] = '0.1'
        self.sp['weight_decay'] = '0.0005'
        #self.sp['train_net'] = '"' + trainnet_prototxt_path + '"'
        #self.sp['test_net'] = '"' + testnet_prototxt_path + '"'

        self.sp['net'] = '"' + testnet_prototxt_path + '"'

        # pretty much never change these.
        self.sp['max_iter'] = '100000'
        self.sp['test_initialization'] = 'false'
        self.sp['average_loss'] = '25'  # this has to do with the display.
        self.sp['iter_size'] = '1'  # this is for accumulating gradients

        if (debug):
            self.sp['max_iter'] = '12'
            self.sp['test_iter'] = '1'
            self.sp['test_interval'] = '4'
            self.sp['display'] = '1'

    def add_from_file(self, filepath):
        """
        Reads a caffe solver prototxt file and updates the Caffesolver
        instance parameters.
        """
        with open(filepath, 'r') as f:
            for line in f:
                if line[0] == '#':
                    continue
                splitLine = line.split(':')
                self.sp[splitLine[0].strip()] = splitLine[1].strip()

    def write(self, filepath):
        """
        Export solver parameters to INPUT "filepath". Sorted alphabetically.
        """
        f = open(filepath, 'w')
        for key, value in sorted(self.sp.items()):
            if not(type(value) is str):
                raise TypeError('All solver parameters must be strings')
            f.write('%s: %s\n' % (key, value))
