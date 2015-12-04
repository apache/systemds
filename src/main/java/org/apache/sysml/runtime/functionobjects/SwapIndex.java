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

package org.apache.sysml.runtime.functionobjects;

import java.io.Serializable;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;


public class SwapIndex extends IndexFunction implements Serializable
{

	private static final long serialVersionUID = -8898087610410746689L;

	private static SwapIndex singleObj = null;
	
	private SwapIndex() {
		// nothing to do here
	}
	
	public static SwapIndex getSwapIndexFnObject() {
		if ( singleObj == null )
			singleObj = new SwapIndex();
		return singleObj;
	}
	
	public Object clone() throws CloneNotSupportedException {
		// cloning is not supported for singleton classes
		throw new CloneNotSupportedException();
	}
	
	@Override
	public void execute(MatrixIndexes in, MatrixIndexes out) {
		out.setIndexes(in.getColumnIndex(), in.getRowIndex());
	}

	@Override
	public void execute(CellIndex in, CellIndex out) {
		int temp=in.row;
		out.row=in.column;
		out.column=temp;
	}

	@Override
	public boolean computeDimension(int row, int col, CellIndex retDim) {
		retDim.set(col, row);
		return false;
	}

	public boolean computeDimension(MatrixCharacteristics in, MatrixCharacteristics out) throws DMLRuntimeException
	{
		out.set(in.getCols(), in.getRows(), in.getColsPerBlock(), in.getRowsPerBlock());
		return false;
	}
}
