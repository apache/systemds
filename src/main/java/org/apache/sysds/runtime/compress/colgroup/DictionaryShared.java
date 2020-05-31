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

import org.apache.commons.lang3.ArrayUtils;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.utils.MemoryEstimates;

/**
 * This dictionary class aims to encapsulate the storage and operations over
 * unique floating point values of a column group. The primary reason for its
 * introduction was to provide an entry point for specialization such as shared
 * dictionaries, which require additional information.
 */
public class DictionaryShared extends Dictionary {
	protected final int[] _colIndexes;
	// linearized <min/max> <min/max> of 
	// column groups that share the dictionary
	protected final double[] _extrema;
	
	public DictionaryShared(double[] values, int[] colIndexes, double[] extrema) {
		super(values);
		_colIndexes = colIndexes;
		_extrema = extrema;
	}
	
	@Override
	public long getInMemorySize() {
		return super.getInMemorySize()
			+ MemoryEstimates.intArrayCost(_colIndexes.length)
			+ MemoryEstimates.doubleArrayCost(_extrema.length);
	}
	
	@Override
	public double aggregate(double init, Builtin fn) {
		//full aggregate directly over extreme values
		int len = _extrema.length;
		int off = fn.getBuiltinCode() == BuiltinCode.MIN ? 0 : 1;
		double ret = init;
		for(int i = off; i < len; i+=2)
			ret = fn.execute(ret, _extrema[i]);
		return ret;
	}
	
	public double[] aggregateCols(double[] init, Builtin fn, int[] cols) {
		int ncol = cols.length;
		double[] ret = init;
		int off = fn.getBuiltinCode() == BuiltinCode.MIN ? 0 : 1;
		for(int i=0; i<ncol; i++) {
			int pos = ArrayUtils.indexOf(_colIndexes, cols[i]);
			ret[i] = fn.execute(ret[i], _extrema[2*pos+off]);
		}
		return ret;
	}
	
	@Override
	public DictionaryShared clone() {
		return new DictionaryShared(
			getValues().clone(), _colIndexes.clone(), _extrema.clone());
	}
}
