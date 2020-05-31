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

package org.apache.sysds.runtime.compress.colgroup;

import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.utils.MemoryEstimates;

/**
 * This dictionary class aims to encapsulate the storage and operations over
 * unique floating point values of a column group. The primary reason for its
 * introduction was to provide an entry point for specialization such as shared
 * dictionaries, which require additional information.
 */
public class Dictionary {
	// linearized <numcol vals> <numcol vals>
	protected final double[] _values;
	
	public Dictionary(double[] values) {
		_values = values;
	}
	
	public double[] getValues() {
		return _values;
	}
	
	public double getValue(int i) {
		return _values[i];
	}
	
	public long getInMemorySize() {
		//object + values array
		return 16 + MemoryEstimates.doubleArrayCost(_values.length);
	}
	
	public int hasZeroTuple(int ncol) {
		int len = _values.length;
		for(int i = 0, off = 0; i < len; i++, off += ncol) {
			boolean allZeros = true;
			for(int j = 0; j < ncol; j++)
				allZeros &= (_values[off + j] == 0);
			if(allZeros)
				return i;
		}
		return -1;
	}
	
	public double aggregate(double init, Builtin fn) {
		//full aggregate can disregard tuple boundaries
		int len = _values.length;
		double ret = init;
		for(int i = 0; i < len; i++)
			ret = fn.execute(ret, _values[i]);
		return ret;
	}
	
	public double[] aggregateCols(double[] init, Builtin fn, int[] cols) {
		int ncol = cols.length;
		int vlen = _values.length / ncol;
		double[] ret = init;
		for(int k = 0; k < vlen; k++)
			for(int j = 0, valOff = k * ncol; j < ncol; j++)
				ret[j] = fn.execute(ret[j], _values[valOff + j]);
		return ret;
	}
	
	public Dictionary apply(ScalarOperator op) {
		//in-place modification of the dictionary
		int len = _values.length;
		for(int i = 0; i < len; i++)
			_values[i] = op.executeScalar(_values[i]);
		return this; //fluent API
	}
	
	@Override
	public Dictionary clone() {
		return new Dictionary(_values.clone());
	}
}
