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

import time, os, argparse, sys, math
import numpy as np

from pyspark import SparkContext
sc = SparkContext()
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

parser=argparse.ArgumentParser("Testing deep networks for different batches")
parser.add_argument('--network', type=str, default='vgg16', choices=['vgg16', 'vgg19', 'resnet200', 'resnet1001', 'unet'])
parser.add_argument('--allocator', type=str, default='cuda', choices=['cuda', 'unified_memory'])
parser.add_argument('--batch_size', help='Batch size. Default: 64', type=int, default=64)
parser.add_argument('--num_images', help='Number of images. Default: 2048', type=int, default=2048)
parser.add_argument('--eviction_policy', help='Eviction policy. Default: align_memory', type=str, default='align_memory', choices=['align_memory', 'lru', 'fifo', 'min_evict', 'lfu', 'mru'])
parser.add_argument('--framework', help='The framework to use for running the benchmark. Default: systemml', type=str, default='systemml', choices=['systemml', 'tensorflow', 'systemml_force_gpu', 'tensorflow-gpu'])
parser.add_argument('--num_channels', help='Number of channels. Default: 3', type=int, default=3)
parser.add_argument('--height', help='Height. Default: 224', type=int, default=224)
parser.add_argument('--width', help='Width. Default: 224', type=int, default=224)
args=parser.parse_args()

#######################################################################
# Required to ensure that TF only uses exactly 1 GPU if framework is tensorflow-gpu, else no gpu
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if args.framework == 'tensorflow-gpu':
	os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
else:
	# Disable tensorflow from grabbing the entire GPU memory
	os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
	os.environ['CUDA_VISIBLE_DEVICES'] = ''
#######################################################################

# To discount the transfer time of batches, we use one randomly generated batch
# and scale the number of epochs
batch_size = args.batch_size
num_images = args.num_images
num_images = num_images - int(num_images % batch_size)
n_batches_for_epoch = num_images / batch_size

# Model-specific parameters
num_classes = 1000
input_shape = (args.num_channels, args.height, args.width)
if args.network == 'unet' and (input_shape[0] != 1 or input_shape[1] != 256 or input_shape[2] != 256):
	raise ValueError('Incorrect input shape for unet: ' + str(input_shape) + '. Supported input shape fo unet: (1, 256, 256)' )
num_pixels = input_shape[0]*input_shape[1]*input_shape[2]

import keras
from keras.utils import np_utils
from keras import backend as K
if args.framework.startswith('systemml'):
	K.set_image_data_format('channels_first')
import os 
import numpy as np
from keras.models import *
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, concatenate # merge
from keras.optimizers import *

#####################################################################################
# Ideally we would have preferred to compare the performance on double precision
# as SystemML's CPU backend only supports double precision. 
# But since TF 1.7 crashes with double precision, we only test with single precision 
use_double_precision = False 
if use_double_precision:
	K.set_floatx('float64')
if args.framework == 'tensorflow-gpu':
	import tensorflow as tf
	from keras.backend.tensorflow_backend import set_session
	tf_config = tf.ConfigProto()
	if args.allocator =='cuda':
		tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9
	elif args.allocator =='unified_memory':
		tf_config.gpu_options.allow_growth = True
	set_session(tf.Session(config=tf_config))
#####################################################################################

error_occured = False
print("Building model ... ")
if args.network == 'vgg16':
	model = keras.applications.vgg16.VGG16(weights='imagenet', classes=num_classes)
elif args.network == 'vgg19':
	model = keras.applications.vgg19.VGG19(weights='imagenet', classes=num_classes)
elif args.network == 'resnet200':
	import resnet
	model = resnet.ResnetBuilder.build_resnet_200(input_shape, num_classes)
elif args.network == 'resnet1001':
	import resnet
	model = resnet.ResnetBuilder.build_resnet_1001(input_shape, num_classes)
elif args.network == 'unet':
	def conv3x3(input, num_filters):
			conv = Conv2D(num_filters, 3, activation = 'relu', padding = 'same')(input)
			conv = Conv2D(num_filters, 3, activation = 'relu', padding = 'same')(conv)
			return conv
	num_filters = [64, 128, 256, 512, 1024]
	model_input = Input((input_shape[1], input_shape[2], input_shape[0]))
	input = model_input
	side_inputs = []
	for i in range(len(num_filters)):
			# Apply max pooling for all except first down_conv
			input = MaxPooling2D(pool_size=(2, 2))(input) if i != 0 else input
			input = conv3x3(input, num_filters[i])
			# Apply dropouts to only last 2 down_conv
			input = Dropout(0.5)(input) if i >= len(num_filters)-2 else input
			side_inputs.append(input)
	input = side_inputs.pop()
	num_filters.pop()
	for i in range(len(num_filters)):
			filters = num_filters.pop()
			input = Conv2D(filters, 3, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(input))
			#input = merge([side_inputs.pop(), input], mode = 'concat', concat_axis = 3)
			input = concatenate([side_inputs.pop(), input])
			input = conv3x3(input, filters)
	conv1 = Conv2D(2, 3, activation = 'relu', padding = 'same')(input)
	model_output = Conv2D(1, 1, activation = 'sigmoid')(conv1)
	model = Model(input = model_input, output = model_output)
else:
	raise ValueError('Unsupported network:' + args.network)
if args.network == 'unet':
	model.compile(optimizer = keras.optimizers.SGD(lr=1e-6, momentum=0.95, decay=5e-4, nesterov=True), loss = 'mean_squared_error')
else:
	model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=1e-6, momentum=0.95, decay=5e-4, nesterov=True))

#------------------------------------------------------------------------------------------
# Use this for baseline experiments:
# Alternate way to avoid eviction is to perform multiple forward/backward pass, aggregate gradients and finally perform update.
looped_minibatch = False
local_batch_size = batch_size
if looped_minibatch:
	if args.network == 'resnet200':
		local_batch_size = 16
	else:
		raise ValueError('looped_minibatch not yet implemented for ' + str(args.network))
	if batch_size % local_batch_size != 0:
		raise ValueError('local_batch_size = ' + str(local_batch_size) + ' should be multiple of batch size=' + str(batch_size))
#------------------------------------------------------------------------------------------

if args.framework.startswith('systemml'):
	print("Initializing Keras2DML.")
	from systemml.mllearn import Keras2DML
	should_load_weights=False
	sysml_model = Keras2DML(spark, model, load_keras_weights=should_load_weights, weights="tmp_weights1")
	if looped_minibatch:
		sysml_model.set(train_algo="looped_minibatch", parallel_batches=int(batch_size/local_batch_size), test_algo="batch") # systemml doesnot have a generator
		sysml_model.set(weight_parallel_batches=False)
	else:
		sysml_model.set(train_algo="batch", test_algo="batch") 
	sysml_model.set(perform_fused_backward_update=True)
	sysml_model.setStatistics(True).setStatisticsMaxHeavyHitters(100)
	# Since this script is used for measuring performance and not for printing script, inline the nn library
	sysml_model.set(inline_nn_library=True)
	# For apples-to-apples comparison, donot force set the allocated array to 0
	sysml_model.setConfigProperty("sysml.gpu.force.memSetZero", "false")
	# Use single GPU
	sysml_model.setConfigProperty("sysml.gpu.availableGPUs", "0")
	# Use user-specified allocator: cuda (default) or unified_memory
	sysml_model.setConfigProperty("sysml.gpu.memory.allocator", args.allocator);
	# Use user-specified eviction policy
	sysml_model.setConfigProperty("sysml.gpu.eviction.policy", args.eviction_policy)
	# Please consider allocating large enough JVM and using large CPU cache
	sysml_model.setConfigProperty("sysml.gpu.eviction.shadow.bufferSize", "0.5")
	sysml_model.setConfigProperty("sysml.caching.bufferSize", "1.0")
	# Use user-specified precision
	if not use_double_precision:
		sysml_model.setConfigProperty("sysml.floating.point.precision", "single")
	sysml_model.setGPU(True).setForceGPU(args.framework=='systemml_force_gpu')
	Xb = np.random.uniform(0,1,num_pixels*batch_size)
	Xb = Xb.reshape((batch_size, num_pixels))
	if args.network == 'unet':
		yb = np.random.randint(5, size=num_pixels*batch_size).reshape((batch_size, num_pixels))
		sysml_model.set(perform_one_hot_encoding=False)
	else:
		yb = np.random.randint(num_classes, size=batch_size)
	from py4j.protocol import Py4JJavaError
	start = time.time()
	try:
		print("Invoking fit")
		sysml_model.fit(Xb, yb, batch_size=local_batch_size, epochs=n_batches_for_epoch)
		print("Done with fit")
	except Py4JJavaError as e:
		error_occured = True
		print("Execution failed: " + str(e))
	except AttributeError as e1:
		error_occured = True
		print("Execution failed: " + str(e1))
elif args.framework.startswith('tensorflow'):
	Xb = np.random.randint(256, size=num_pixels*batch_size).reshape((batch_size, input_shape[1],input_shape[2], input_shape[0])) + 1
	if args.network == 'unet':
		yb = np.random.randint(5, size=num_pixels*batch_size).reshape((batch_size, input_shape[1],input_shape[2], input_shape[0]))
	else:
		yb = np.random.randint(num_classes, size=batch_size)
		yb = np_utils.to_categorical(yb, num_classes)
	start = time.time()
	model.fit(Xb, yb, batch_size=batch_size, epochs=n_batches_for_epoch)
K.clear_session()
end = time.time()
if not error_occured:
	with open('time.txt', 'a') as f:
		f.write(args.framework + ',' + args.network + ',synthetic_imagenet,1,' + str(batch_size) + ',1,' + str(num_images) + "," + str(end-start) + "," + args.eviction_policy + ',' + args.allocator + '\n')
