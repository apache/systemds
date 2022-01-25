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

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.functionobjects.Divide;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;

/**
 * This dictionary class is a specialization for the DeltaDDCColgroup. Here the adjustments for operations for the delta
 * encoded values are implemented.
 */
public class DeltaDictionary extends Dictionary {

	private final int _numCols;

	public DeltaDictionary(double[] values, int numCols) {
		super(values);
		_numCols = numCols;
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
}
