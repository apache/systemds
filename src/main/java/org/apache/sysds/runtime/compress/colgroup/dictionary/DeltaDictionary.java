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

package org.apache.sysds.runtime.compress.colgroup.dictionary;

import java.io.DataOutput;
import java.io.IOException;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.runtime.functionobjects.Divide;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;

/**
 * This dictionary class is a specialization for the DeltaDDCColgroup. Here the adjustments for operations for the delta
 * encoded values are implemented.
 */
public class DeltaDictionary extends ADictionary {
	private static final long serialVersionUID = -5700139221491143705L;

	private final int _numCols;

	protected final double[] _values;

	public DeltaDictionary(double[] values, int numCols) {
		_values = values;
		_numCols = numCols;
	}

	@Override 
	public double[] getValues(){
		return _values;
	}

	@Override
	public DeltaDictionary applyScalarOp(ScalarOperator op) {
		final double[] retV = new double[_values.length];
		if(op.fn instanceof Multiply || op.fn instanceof Divide) {
			for(int i = 0; i < _values.length; i++)
				retV[i] = op.executeScalar(_values[i]);
		}
		else if(op.fn instanceof Plus || op.fn instanceof Minus) {
			// With Plus and Minus only the first row needs to be updated when delta encoded
			for(int i = 0; i < _values.length; i++) {
				if(i < _numCols)
					retV[i] = op.executeScalar(_values[i]);
				else
					retV[i] = _values[i];
			}
		}
		else
			throw new NotImplementedException();

		return new DeltaDictionary(retV, _numCols);
	}

	@Override
	public long getInMemorySize() {
		return Dictionary.getInMemorySize(_values.length);
	}

	@Override
	public void write(DataOutput out) throws IOException {
		throw new NotImplementedException();
	}

	@Override
	public long getExactSizeOnDisk() {
		throw new NotImplementedException();
	}

	@Override
	public DictType getDictType() {
		throw new NotImplementedException();
	}

	@Override
	public int getNumberOfValues(int ncol) {
		return _values.length / ncol;
	}

	@Override
	public int getNumberOfColumns(int nrow){
		return _values.length / nrow;
	}

	@Override
	public String getString(int colIndexes) {
		throw new NotImplementedException();
	}

	@Override
	public long getNumberNonZeros(int[] counts, int nCol) {
		throw new NotImplementedException();
	}

	@Override
	public boolean equals(IDictionary o) {
		throw new NotImplementedException();
	}

	@Override
	public IDictionary clone() {
		throw new NotImplementedException();
	}
}
