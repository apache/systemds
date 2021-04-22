<!--
{% comment %}
Licensed to the Apache Software Foundation (ASF) under one or more
contributor license agreements.  See the NOTICE file distributed with
this work for additional information regarding copyright ownership.
The ASF licenses this file to you under the Apache License, Version 2.0
(the "License"); you may not use this file except in compliance with
the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
{% endcomment %}
-->

# Initial prototype for GPU backend

The GPU backend implements two important abstract classes:
1. `org.apache.sysml.runtime.controlprogram.context.GPUContext`
2. `org.apache.sysml.runtime.controlprogram.context.GPUObject`

The `GPUContext` is responsible for GPU memory management and initialization/destruction of Cuda handles.
Currently, an active instance of the `GPUContext` class is made available globally and is used to store handles
of the allocated blocks on the GPU. A count is kept per block for the number of instructions that need it.
When the count is 0, the block may be evicted on a call to `GPUObject.evict()`.

A `GPUObject` (like RDDObject and BroadcastObject) is stored in CacheableData object. It gets call-backs from SystemML's bufferpool on following methods
1. void acquireDeviceRead()
2. void acquireDeviceModifyDense()
3. void acquireDeviceModifySparse
4. void acquireHostRead()
5. void acquireHostModify()
6. void releaseInput()
7. void releaseOutput()

Sparse matrices on GPU are represented in `CSR` format. In the SystemML runtime, they are represented in `MCSR` or modified `CSR` format.
A conversion cost is incurred when sparse matrices are sent back and forth between host and device memory.

Concrete classes `JCudaContext` and `JCudaObject` (which extend `GPUContext` & `GPUObject` respectively) contain references to `org.jcuda.*`.

The `LibMatrixCUDA` class contains methods to invoke CUDA libraries (where available) and invoke custom kernels. 
Runtime classes (that extend `GPUInstruction`) redirect calls to functions in this class.
Some functions in `LibMatrixCUDA` need finer control over GPU memory management primitives. These are provided by `JCudaObject`.

### Setup instructions:

1. Follow the instructions from `https://developer.nvidia.com/cuda-downloads` and install CUDA 8.0.
2. Follow the instructions from `https://developer.nvidia.com/cudnn` and install CuDNN v5.1.

To use SystemML's GPU backend when using the jar or uber-jar
1. Add JCuda's jar into the classpath.
2. Use `-gpu` flag.

For example: to use GPU backend in standalone mode:
```bash
java -classpath $JAR_PATH:systemml-1.0.0-SNAPSHOT-standalone.jar org.apache.sysml.api.DMLScript -f MyDML.dml -gpu -exec singlenode ... 
```
