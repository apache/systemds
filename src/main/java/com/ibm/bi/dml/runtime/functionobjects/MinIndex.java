/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.runtime.functionobjects;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixIndexes;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;


public class MinIndex extends IndexFunction
{

	private static final long serialVersionUID = -4159274805822230421L;

	private static MinIndex singleObj = null;
	
	private MinIndex() {
		// nothing to do here
	}
	
	public static MinIndex getMinIndexFnObject() {
		if ( singleObj == null )
			singleObj = new MinIndex();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}
	
	@Override
	public void execute(MatrixIndexes in, MatrixIndexes out) {
		long min = Math.min(in.getRowIndex(), in.getColumnIndex());
		out.setIndexes(min, min);
	}

	@Override
	public void execute(CellIndex in, CellIndex out) {
		int min = Math.min(in.row, in.column);
		out.set(min, min);
	}

	@Override
	public boolean computeDimension(int row, int col, CellIndex retDim) {
		int min=Math.min(row, col);
		retDim.set(min, min);
		return false;
	}
	
	public boolean computeDimension(MatrixCharacteristics in, MatrixCharacteristics out) throws DMLRuntimeException
	{
		long minMatrix=Math.min(in.getRows(), in.getCols());
		int minBlock=Math.min(in.getRowsPerBlock(), in.getColsPerBlock());
		out.set(minMatrix, minMatrix, minBlock, minBlock);
		return false;
	}

}
