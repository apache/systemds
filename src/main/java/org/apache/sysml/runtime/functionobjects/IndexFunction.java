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

import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;

public abstract class IndexFunction extends FunctionObject implements Serializable 
{
	private static final long serialVersionUID = -7672111359444767237L;
	
	//compute output indexes
	
	public abstract void execute(MatrixIndexes in, MatrixIndexes out);
	
	public abstract void execute(CellIndex in, CellIndex out);
	
	//determine of dimension has been reduced
	public abstract boolean computeDimension(int row, int col, CellIndex retDim);

	//compute output dimensions
	public abstract boolean computeDimension(MatrixCharacteristics in, MatrixCharacteristics out);
}
