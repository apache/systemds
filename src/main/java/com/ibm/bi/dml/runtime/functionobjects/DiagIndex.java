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


public class DiagIndex extends IndexFunction
{

	private static final long serialVersionUID = -5294771266108903886L;

	private static DiagIndex singleObj = null;
	
	private DiagIndex() {
		// nothing to do here
	}
	
	public static DiagIndex getDiagIndexFnObject() {
		if ( singleObj == null )
			singleObj = new DiagIndex();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}
	
	@Override
	public void execute(MatrixIndexes in, MatrixIndexes out) {
		//only used for V2M
		out.setIndexes(in.getRowIndex(), in.getRowIndex());
	}

	@Override
	public void execute(CellIndex in, CellIndex out) {
		//only used for V2M
		out.set(in.row, in.row);
	}

	@Override
	public boolean computeDimension(int row, int col, CellIndex retDim) {
		if( col == 1 ) //diagV2M
			retDim.set(row, row);
		else //diagM2V
			retDim.set(row, 1);
		return false;
	}
	
	public boolean computeDimension(MatrixCharacteristics in, MatrixCharacteristics out) throws DMLRuntimeException
	{
		if( in.getCols() == 1 ) //diagV2M
			out.set(in.getRows(), in.getRows(), in.getRowsPerBlock(), in.getRowsPerBlock());
		else //diagM2V
			out.set(in.getRows(), 1, in.getRowsPerBlock(), in.getRowsPerBlock());
		return false;
	}

}
