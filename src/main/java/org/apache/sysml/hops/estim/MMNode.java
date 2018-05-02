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

package org.apache.sysml.hops.estim;

import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;

/**
 * Helper class to represent matrix multiply operators in a DAG
 * along with references to its abstract data handles.
 */
public class MMNode 
{
	private final MMNode _m1;
	private final MMNode _m2;
	private final MatrixBlock _data;
	private final MatrixCharacteristics _mc;
	private Object _synops = null;
	
	public MMNode(MatrixBlock in) {
		_m1 = null;
		_m2 = null;
		_data = in;
		_mc = in.getMatrixCharacteristics();
	}
	
	public MMNode(MMNode left, MMNode right) {
		_m1 = left;
		_m2 = right;
		_data = null;
		_mc = new MatrixCharacteristics(
			_m1.getRows(), _m2.getCols(), -1, -1);
	}
	
	public int getRows() {
		return (int)_mc.getRows();
	}
	
	public int getCols() {
		return (int)_mc.getCols();
	}
	
	public MatrixCharacteristics getMatrixCharacteristics() {
		return _mc;
	}
	
	public MMNode getLeft() {
		return _m1;
	}
	
	public MMNode getRight() {
		return _m2;
	}
	
	public boolean isLeaf() {
		return _data != null;
	}
	
	public MatrixBlock getData() {
		return _data;
	}
	
	public void setSynopsis(Object obj) {
		_synops = obj;
	}
	
	public Object getSynopsis() {
		return _synops;
	}
}
