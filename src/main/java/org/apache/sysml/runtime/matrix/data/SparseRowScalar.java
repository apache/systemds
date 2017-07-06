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


package org.apache.sysml.runtime.matrix.data;

import java.io.Serializable;

public final class SparseRowScalar extends SparseRow implements Serializable 
{
	private static final long serialVersionUID = 722193514969067477L;
	
	private int index;
	private double value;
	
	public SparseRowScalar() {
		index = -1;
		value = 0;
	}
	
	public SparseRowScalar(int ix, double val) {
		index = ix;
		value = val;
	}
	
	@Override
	public int size() {
		return (index < 0) ? 0 : 1;
	}
	
	@Override
	public boolean isEmpty() {
		return (index < 0);
	}
	
	@Override
	public double[] values() {
		return new double[]{value};
	}
	
	public int[] indexes() {
		return new int[]{index};
	}

	@Override
	public void reset(int estnns, int maxnns) {
		index = -1;
	}
	
	@Override
	public boolean set(int col, double v) {
		boolean ret = (index==col && v==0 
			|| index<0 && v!=0);
		if( index >= 0 && index != col )
			throw new RuntimeException(
				"Invalid set to sparse row scalar.");
		index = (v!=0) ? col : -1;
		value = v;
		return ret;
	}

	@Override
	public void append(int col, double v) {
		if( v == 0 )
			return;
		if( index >= 0 )
			throw new RuntimeException(
				"Invalid append to sparse row scalar.");
		index = col;
		value = v;
	}
	
	public double get(int col) {
		return (index==col) ? value : 0;
	}
	
	public void sort() {
		//do nothing
	}

	@Override
	public void compact() {
		index = (value!=0) ? index : -1;
	}
}
