"""
Script to generate caffe proto and .caffemodel files from Keras models
"""
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


"""
Currently can only generate a Dense model
"""
def generate_caffe_model(kModel):
    n = caffe.NetSpec()
    layers = kModel.keras_model.layers

    for layer in layers:
        if type(layer) == keras.InputLayer:
            #Grab the batchsize from index 0, shift over channels to index 1, and place the rest into the dictionary
            num = len(layer.batch_input_shape) - 1 #Range from 1st index to second last
            batch_list = [layer.batch_input_shape[0], layer.batch_shape[-1]]
            for i  in range(1 ,num):
                batch_list.append(layer.batch_input_shape[i])
            n.data, n.label = L.Input(shape=[dict(dim=batch_list)])
        elif type(layer) == keras.Dense:
            #Pull name from Keras
            name = layer.name
            #Pull layer name of the layer passing to current layer
            in_names =[]
            for node in layer.inbound_nodes: #get inbound nodes to current layer
                node_list = node.inbound_layers #get layers pointing to this node
                in_names = in_names + node_list
            
            #Pipe names into caffe using unique Keras layer names
            print(in_names) #debug to assure there are a correct number of layer names (1)
            n[name] = L.InnerProduct()
    return n

#def generate_caffe_weights(kModel):


#def generate_caffe_solver(kModel):


