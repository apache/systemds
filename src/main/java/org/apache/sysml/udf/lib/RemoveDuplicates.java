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

package org.apache.sysml.udf.lib;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.instructions.cp.ListObject;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.udf.FunctionParameter;
import org.apache.sysml.udf.List;
import org.apache.sysml.udf.Matrix;
import org.apache.sysml.udf.PackageFunction;
import org.apache.sysml.udf.Matrix.ValueType;

/**
 * Use this class to remove duplicate matrices from list of matrices.
 * It also returns the indexes which maps the original input list to the output list.
 * 
 * Usage:
 * <pre>
 * <code>
 * distinct = externalFunction(list[unknown] inL) return (list[unknown] outL, matrix[double] idx) implemented in (classname="org.apache.sysml.udf.lib.RemoveDuplicates", exectype="mem");
 * X = rand(rows=10, cols=10)
 * Y = X*sum(X);
 * Z = sum(X)*X;
 * W = X*sum(X);
 * inL = list(Y, Z, W)
 * [outL, idx] = distinct(inL);
 * print(toString(idx));
 * </code>
 * </pre>
 * 
 * The above code prints:
 * 1.000
 * 2.000
 * 1.000
 */
public class RemoveDuplicates extends PackageFunction {
	private static final long serialVersionUID = -3905212831582648882L;

	private List outputList;
	private Matrix indexes;
	private Random rand = new Random();
	
	@Override
	public int getNumFunctionOutputs() {
		return 2;
	}

	@Override
	public FunctionParameter getFunctionOutput(int pos) {
		if(pos == 0)
			return outputList;
		else if(pos == 1)
			return indexes;
		throw new RuntimeException("Invalid function output being requested");
	}
	
	private int indexOf(java.util.List<MatrixBlock> list, MatrixBlock mb) {
//		Caused by: java.lang.RuntimeException: equals should never be called for matrix blocks.
//		at org.apache.sysml.runtime.matrix.data.MatrixBlock.equals(MatrixBlock.java:5644)
//		return list.indexOf(mb);
		for(int i = 0; i < list.size(); i++) {
			if(list.get(i) == mb) {
				return i;
			}
		}
		return -1;
	}

	@Override
	public void execute() {
		java.util.List<Data> inputData = ((List)getFunctionInput(0)).getListObject().getData();
		java.util.List<Data> outputData = new ArrayList<>();
		java.util.List<MatrixBlock> outputMB = new ArrayList<>();
		indexes = new Matrix( "tmp_" + rand.nextLong(), inputData.size(), 1, ValueType.Double );
		MatrixBlock indexesMB = allocateDenseMatrixBlock(indexes);
		double [] indexesData = indexesMB.getDenseBlockValues();
		
		for(int i = 0; i < inputData.size(); i++) {
			Data elem = inputData.get(i);
			if(elem instanceof MatrixObject) {
				MatrixBlock mb = ((MatrixObject)elem).acquireRead();
				int index = indexOf(outputMB, mb);
				if(index >= 0) {
					indexesData[i] = indexOf(outputMB, mb) + 1;
				}
				else {
					outputMB.add(mb);
					outputData.add(elem);
					indexesData[i] = outputMB.size();
				}
				((MatrixObject)elem).release();
			}
			else {
				throw new RuntimeException("Only list of matrices is supported in RemoveDuplicates");
			}
		}
		indexesMB.setNonZeros(indexesData.length);
		try {
			indexes.setMatrixDoubleArray(indexesMB, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo);
		} catch (IOException e) {
			throw new RuntimeException("Exception while executing RemoveDuplicates", e);
		}
		outputList = new List(new ListObject(outputData));
	}
	
	private static MatrixBlock allocateDenseMatrixBlock(Matrix mat) {
		int rows = (int) mat.getNumRows();
		int cols = (int) mat.getNumCols();
		MatrixBlock mb = new MatrixBlock(rows, cols, false);
		mb.allocateDenseBlock();
		return mb;
	}
}
