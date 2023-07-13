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

package org.apache.sysds.runtime.functionobjects;


import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.runtime.meta.DataCharacteristics;

/**
 * This index function is NOT used for actual sorting but just as a reference
 * in ReorgOperator in order to identify sort operations.
 * 
 */
public class SortIndex extends IndexFunction
{
	private static final long serialVersionUID = -8446389232078905200L;

	private final int[] _cols;
	private final boolean _decreasing;
	private final boolean _ixreturn;
	
	public SortIndex(int col, boolean decreasing, boolean indexreturn) {
		this(new int[]{col}, decreasing, indexreturn);
	}
	
	public SortIndex(int[] cols, boolean decreasing, boolean indexreturn) {
		_cols = cols;
		_decreasing = decreasing;
		_ixreturn = indexreturn;
	}

	public int[] getCols() {
		return _cols;
	}
	
	public boolean getDecreasing() {
		return _decreasing;
	}
	
	public boolean getIndexReturn() {
		return _ixreturn;
	}
	
	@Override
	public boolean computeDimension(int row, int col, CellIndex retDim) {
		retDim.set(row, _ixreturn?1:col);
		return false;
	}

	@Override
	public void execute(MatrixIndexes in, MatrixIndexes out) {
		throw new NotImplementedException();
	}

	@Override
	public void execute(CellIndex in, CellIndex out) {
		throw new NotImplementedException();
	}

	@Override
	public boolean computeDimension(DataCharacteristics in, DataCharacteristics out) {
		throw new NotImplementedException();
	}
}
