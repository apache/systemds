/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.sysml.runtime.instructions.gpu.context;

import jcuda.Sizeof;

// This indirection prepares SystemML's GPU backend to allow
// for arbitrary datatype
public abstract class JCudaKernels {
	private static JCudaKernels singletonObj = new DoublePrecisionKernels();
	public static JCudaKernels handle() {
		return singletonObj;
	}
	public abstract int cudnnDataType();
	public abstract int sizeOfDatatype();
}

class DoublePrecisionKernels extends JCudaKernels {

	@Override
	public int cudnnDataType() {
		return jcuda.jcudnn.cudnnDataType.CUDNN_DATA_DOUBLE;
	}

	@Override
	public int sizeOfDatatype() {
		return Sizeof.DOUBLE;
	}
	
}

class SinglePrecisionKernels extends JCudaKernels {

	@Override
	public int cudnnDataType() {
		return jcuda.jcudnn.cudnnDataType.CUDNN_DATA_FLOAT;
	}

	@Override
	public int sizeOfDatatype() {
		return Sizeof.FLOAT;
	}
	
}