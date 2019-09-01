/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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

package org.tugraz.sysds.runtime.functionobjects;

import org.tugraz.sysds.runtime.matrix.data.MatrixIndexes;
import org.tugraz.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.tugraz.sysds.runtime.meta.DataCharacteristics;


public class OffsetColumnIndex extends IndexFunction
{
	private static final long serialVersionUID = 1523769994005450946L;

	//private static OffsetColumnIndex singleObj = null;
	private int offset, numRowsInOutput, numColumnsInOutput;
	
	private OffsetColumnIndex(int offset) {
		this.offset = offset;
	}
	
	public static OffsetColumnIndex getOffsetColumnIndexFnObject(int offset) {
		return new OffsetColumnIndex(offset);
		//if ( singleObj == null )
		//	singleObj = new OffsetColumnIndex(offset);
		//return singleObj;
	}

	public void setOffset(int offset){
		this.offset = offset;
	}
	
	@Override
	public void execute(MatrixIndexes in, MatrixIndexes out) {
		out.setIndexes(in.getRowIndex(), in.getColumnIndex()+offset);
	}

	@Override
	public void execute(CellIndex in, CellIndex out) {
		out.row=in.row;
		out.column=offset+in.column;
	}

	@Override
	public boolean computeDimension(int row, int col, CellIndex retDim) {
		retDim.set(numRowsInOutput, numColumnsInOutput);
		return false;
	}

	@Override
	public boolean computeDimension(DataCharacteristics in, DataCharacteristics out) {
		out.set(numRowsInOutput, numColumnsInOutput, in.getBlocksize());
		return false;
	}
}
