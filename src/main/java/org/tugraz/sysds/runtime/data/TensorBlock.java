/*
 * Copyright 2018 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tugraz.sysds.runtime.data;

import java.io.Serializable;

public class TensorBlock implements Serializable
{
	private static final long serialVersionUID = -4205257127878517048L;
	
	protected int[] dims     = new int[2];
	protected boolean sparse = true;
	protected long nonZeros  = 0;
	
	//matrix data (sparse or dense)
	protected DenseBlock denseBlock   = null;
	protected SparseBlock sparseBlock = null;
	
	public TensorBlock() {
		
	}
	
	public double get(int[] ix) {
		return -1;
	}
	
	public void set(int[] ix, double v) {
		
	}
}
