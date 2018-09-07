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
import java.util.Iterator;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.IJV;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.udf.FunctionParameter;
import org.apache.sysml.udf.Matrix;
import org.apache.sysml.udf.PackageFunction;
import org.apache.sysml.udf.Scalar;
import org.apache.sysml.udf.Matrix.ValueType;

/**
 * This external built-in function addresses following two common scenarios:
 * 1. cbind (cbind (cbind ( X1, X2 ), X3 ), X4)
 * 2. With spagetization: cbind (cbind (cbind ( matrix(X1, rows=length(X1), cols=1), matrix(X2, rows=length(X2), cols=1) ), matrix(X3, rows=length(X3), cols=1) ), matrix(X4, rows=length(X4), cols=1))
 * 
 * The API of this external built-in function is as follows:
 * 
 * func = externalFunction(int numInputs, boolean spagetize, matrix[double] X1, matrix[double] X2,  matrix[double] X3, matrix[double] X4) return (matrix[double] out) 
 * implemented in (classname="org.apache.sysml.udf.lib.MultiInputCbind",exectype="mem");
 * 
 */
public class MultiInputCbind extends PackageFunction {
	private static final long serialVersionUID = -4266180315672563097L;

	private Matrix ret;
	private MatrixBlock retMB;
	long numRetRows; long numRetCols;
	boolean spagetize;
	
	@Override
	public int getNumFunctionOutputs() {
		return 1;
	}

	@Override
	public FunctionParameter getFunctionOutput(int pos) {
		if(pos == 0)
			return ret;
		else
			throw new RuntimeException("MultiInputCbind produces only one output");
	}

	@Override
	public void execute() {
		int numInputs = Integer.parseInt(((Scalar)getFunctionInput(0)).getValue());
		spagetize = Boolean.parseBoolean(((Scalar)getFunctionInput(1)).getValue());
		
		// Compute output dimensions
		numRetCols = 0;
		if(spagetize) {
			// Assumption the inputs are of same shape
			MatrixBlock in = ((Matrix) getFunctionInput(2)).getMatrixObject().acquireRead();
			numRetRows = in.getNumRows()*in.getNumColumns();
			numRetCols = numInputs;
			((Matrix) getFunctionInput(2)).getMatrixObject().release();
		}
		else {
			for(int inputID = 2; inputID < numInputs + 2; inputID++) {
				MatrixBlock in = ((Matrix) getFunctionInput(inputID)).getMatrixObject().acquireRead();
				numRetRows = in.getNumRows();
				numRetCols += in.getNumColumns();
				((Matrix) getFunctionInput(inputID)).getMatrixObject().release();
			}
		}
		
		allocateOutput();
		
		// Performs cbind (cbind (cbind ( X1, X2 ), X3 ), X4)
		double [] retData = retMB.getDenseBlockValues();
		int startColumn = 0;
		for(int inputID = 2; inputID < numInputs + 2; inputID++) {
			MatrixBlock in = ((Matrix) getFunctionInput(inputID)).getMatrixObject().acquireRead();
			if(spagetize && in.getNumRows()*in.getNumColumns() != numRetRows) {
				throw new RuntimeException("Expected the inputs to be of same size when spagetization is turned on.");
			}
			int inputNumCols = in.getNumColumns();
			if(in.isInSparseFormat()) {
				Iterator<IJV> iter = in.getSparseBlockIterator();
				while(iter.hasNext()) {
					IJV ijv = iter.next();
					if(spagetize) {
						// Perform matrix(X1, rows=length(X1), cols=1) operation before cbind
						// Output Column ID = inputID-2 for all elements of inputs
						int outputRowIndex = ijv.getI()*inputNumCols + ijv.getJ();
						int outputColIndex = inputID-2;
						retData[(int) (outputRowIndex*retMB.getNumColumns() + outputColIndex)] = ijv.getV();
					}
					else {
						// Traditional cbind
						// Row ID remains the same as that of input
						int outputRowIndex = ijv.getI();
						int outputColIndex = ijv.getJ() + startColumn;
						retData[(int) (outputRowIndex*retMB.getNumColumns() + outputColIndex)] = ijv.getV();
					}
				}
			}
			else {
				double [] denseBlock = in.getDenseBlockValues();
				if(denseBlock != null) {
					if(spagetize) {
						// Perform matrix(X1, rows=length(X1), cols=1) operation before cbind
						// Output Column ID = inputID-2 for all elements of inputs
						int j = inputID-2;
						for(int i = 0; i < numRetRows; i++) {
							retData[(int) (i*numRetCols + j)] = denseBlock[i];
						}
					}
					else {
						// Traditional cbind
						// Row ID remains the same as that of input
						for(int i = 0; i < retMB.getNumRows(); i++) {
							for(int j = 0; j < inputNumCols; j++) {
								int outputColIndex = j + startColumn;
								retData[(int) (i*numRetCols + outputColIndex)] = denseBlock[i*inputNumCols + j];
							}
						}
					}
				}
			}
			((Matrix) getFunctionInput(inputID)).getMatrixObject().release();
			startColumn += inputNumCols;
		}
	
		retMB.recomputeNonZeros();
		try {
			retMB.examSparsity();
			ret.setMatrixDoubleArray(retMB, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo);
		} catch (DMLRuntimeException e) {
			throw new RuntimeException("Error while executing MultiInputCbind", e);
		} catch (IOException e) {
			throw new RuntimeException("Error while executing MultiInputCbind", e);
		}	
	}
	
	private void allocateOutput() {
		String dir = createOutputFilePathAndName( "TMP" );
		ret = new Matrix( dir, numRetRows, numRetCols, ValueType.Double );
		retMB = new MatrixBlock((int) numRetRows, (int) numRetCols, false);
		retMB.allocateDenseBlock();
	}
	

}
