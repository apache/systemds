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

package org.apache.sysml.runtime.functionobjects;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;


public class MaxIndex extends IndexFunction
{

	private static final long serialVersionUID = -4941564912238185729L;

	private static MaxIndex singleObj = null;
	
	private MaxIndex() {
		// nothing to do here
	}
	
	public static MaxIndex getMaxIndexFnObject() {
		if ( singleObj == null )
			singleObj = new MaxIndex();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}
	
	@Override
	public void execute(MatrixIndexes in, MatrixIndexes out) {
		long max = Math.max(in.getRowIndex(), in.getColumnIndex());
		out.setIndexes(max, max);
	}

	@Override
	public void execute(CellIndex in, CellIndex out) {
		int max = Math.max(in.row, in.column);
		out.set(max, max);
	}

	@Override
	public boolean computeDimension(int row, int col, CellIndex retDim) {
		int max=Math.max(row, col);
		retDim.set(max, max);
		return false;
	}
	
	public boolean computeDimension(MatrixCharacteristics in, MatrixCharacteristics out) throws DMLRuntimeException
	{
		long maxMatrix=Math.max(in.getRows(), in.getCols());
		int maxBlock=Math.max(in.getRowsPerBlock(), in.getColsPerBlock());
		out.set(maxMatrix, maxMatrix, maxBlock, maxBlock);
		return false;
	}

}
