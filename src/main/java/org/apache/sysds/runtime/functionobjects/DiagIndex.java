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

import org.apache.sysds.runtime.matrix.data.MatrixIndexes;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.runtime.meta.DataCharacteristics;


public class DiagIndex extends IndexFunction
{
	private static final long serialVersionUID = -5294771266108903886L;
	private final boolean diagV2M;
	
	private DiagIndex(boolean v2m) {
		diagV2M = v2m;
	}
	
	public static DiagIndex getDiagIndexFnObject() {
		return getDiagIndexFnObject(true);
	}
	
	public static DiagIndex getDiagIndexFnObject(boolean v2m) {
		return new DiagIndex(v2m);
	}
	
	@Override
	public void execute(MatrixIndexes in, MatrixIndexes out) {
		out.setIndexes(in.getRowIndex(), diagV2M ? in.getRowIndex() : 1);
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
	
	@Override
	public boolean computeDimension(DataCharacteristics in, DataCharacteristics out) {
		if( in.getCols() == 1 ) //diagV2M
			out.set(in.getRows(), in.getRows(), in.getBlocksize(), in.getBlocksize());
		else //diagM2V
			out.set(in.getRows(), 1, in.getBlocksize(), in.getBlocksize());
		return false;
	}
}
