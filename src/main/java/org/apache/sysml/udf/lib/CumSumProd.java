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
 * Variant of cumsum:<br>
 * Computes following two functions:<br>
 * <pre>
 * <code>
 * cumsum_prod = function (Matrix[double] X, Matrix[double] C, double start)  return (Matrix[double] Y)
 * # Computes the following recurrence in log-number of steps:
 * # Y [1, ] = X [1, ] + C [1, ] * start;
 * # Y [i+1, ] = X [i+1, ] + C [i+1, ] * Y [i, ]
 * {
 * 	Y = X; P = C; m = nrow(X); k = 1;
 * 	Y [1, ] = Y [1, ] + C [1, ] * start;
 * 	while (k &lt; m) {
 * 		Y [k+1 : m, ] = Y [k+1 : m, ] + Y [1 : m-k, ] * P [k+1 : m, ];
 * 		P [k+1 : m, ] = P [1 : m-k, ] * P [k+1 : m, ];
 * 		k = 2 * k;
 * 	}
 * }
 * 
 * cumsum_prod_reverse = function (Matrix[double] X, Matrix[double] C, double start) return (Matrix[double] Y)
 * # Computes the reverse recurrence in log-number of steps:
 * # Y [m, ] = X [m, ] + C [m, ] * start;
 * # Y [i-1, ] = X [i-1, ] + C [i-1, ] * Y [i, ]
 * {
 * 	Y = X; P = C; m = nrow(X); k = 1;
 * 	Y [m, ] = Y [m, ] + C [m, ] * start;
 * 	while (k &lt; m) {
 * 		Y [1 : m-k, ] = Y [1 : m-k, ] + Y [k+1 : m, ] * P [1 : m-k, ];
 * 		P [1 : m-k, ] = P [k+1 : m, ] * P [1 : m-k, ];
 * 		k = 2 * k;
 * 	}
 * }
 * </code>
 * </pre>
 * 
 * The API of this external built-in function is as follows:<br>
 * <pre>
 * <code>
 * func = externalFunction(matrix[double] X, matrix[double] C,  double start, boolean isReverse) return (matrix[double] Y) 
 * implemented in (classname="org.apache.sysml.udf.lib.CumSumProd",exectype="mem");
 * </code>
 * </pre>
 */
public class CumSumProd extends PackageFunction {

	private static final long serialVersionUID = -7883258699548686065L;
	private Matrix ret;
	private MatrixBlock retMB, X, C;
	private double start;
	private boolean isReverse;

	@Override
	public int getNumFunctionOutputs() {
		return 1;
	}

	@Override
	public FunctionParameter getFunctionOutput(int pos) {
		if(pos == 0)
			return ret;
		else
			throw new RuntimeException("CumSumProd produces only one output");
	}

	@Override
	public void execute() {
		X = ((Matrix) getFunctionInput(0)).getMatrixObject().acquireRead();
		C = ((Matrix) getFunctionInput(1)).getMatrixObject().acquireRead();
		if(X.getNumRows() != C.getNumRows())
			throw new RuntimeException("Number of rows of X and C should match");
		if( X.getNumColumns() != C.getNumColumns() && C.getNumColumns() != 1 )
			throw new RuntimeException("Incorrect Number of columns of X and C (Expected C to be of same dimension or a vector)");
		start = Double.parseDouble(((Scalar)getFunctionInput(2)).getValue());
		isReverse = Boolean.parseBoolean(((Scalar)getFunctionInput(3)).getValue()); 
		
		numRetRows = X.getNumRows();
		numRetCols = X.getNumColumns();
		allocateOutput();
		
		// Copy X to Y
		denseBlock = retMB.getDenseBlockValues();
		if(X.isInSparseFormat()) {
			Iterator<IJV> iter = X.getSparseBlockIterator();
			while(iter.hasNext()) {
				IJV ijv = iter.next();
				denseBlock[ijv.getI()*numRetCols + ijv.getJ()] = ijv.getV();
			}
		}
		else {
			if(X.getDenseBlock() != null)
				System.arraycopy(X.getDenseBlockValues(), 0, denseBlock, 0, denseBlock.length);
		}
		
		if(!isReverse) {
			// Y [1, ] = X [1, ] + C [1, ] * start;
			// Y [i+1, ] = X [i+1, ] + C [i+1, ] * Y [i, ]
			addCNConstant(0, start);
			for(int i = 1; i < numRetRows; i++) {
				addC(i, true);
			}
		}
		else {
			// Y [m, ] = X [m, ] + C [m, ] * start;
			// Y [i-1, ] = X [i-1, ] + C [i-1, ] * Y [i, ]
			addCNConstant(numRetRows-1, start);
			for(int i = numRetRows - 2; i >= 0; i--) {
				addC(i, false);
			}
		}
		
		((Matrix) getFunctionInput(1)).getMatrixObject().release();
		((Matrix) getFunctionInput(0)).getMatrixObject().release();
		
		retMB.recomputeNonZeros();
		try {
			retMB.examSparsity();
			ret.setMatrixDoubleArray(retMB, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo);
		} catch (DMLRuntimeException e) {
			throw new RuntimeException("Error while executing CumSumProd", e);
		} catch (IOException e) {
			throw new RuntimeException("Error while executing CumSumProd", e);
		}
	}
	
	int numRetRows; int numRetCols;
	double [] denseBlock; 
	
	private void addCNConstant(int i, double constant) {
		boolean isCVector = C.getNumColumns() != ret.getNumCols();
		if(C.isInSparseFormat()) {
			Iterator<IJV> iter = C.getSparseBlockIterator(i, i+1);
			while(iter.hasNext()) {
				IJV ijv = iter.next();
				if(!isCVector)
					denseBlock[ijv.getI()*numRetCols + ijv.getJ()] += ijv.getV() * constant;
				else {
					double val = ijv.getV();
					for(int j = ijv.getI()*numRetCols; j < (ijv.getI()+1)*numRetCols; j++) {
						denseBlock[j] += val*constant;
					}
				}
			}
		}
		else {
			double [] CBlk = C.getDenseBlockValues();
			if(CBlk != null) {
				if(!isCVector) {
					for(int j = i*numRetCols; j < (i+1)*numRetCols; j++) {
						denseBlock[j] += CBlk[j]*constant;
					}
				}
				else {
					for(int j = i*numRetCols; j < (i+1)*numRetCols; j++) {
						denseBlock[j] += CBlk[i]*constant;
					}
				}
			}
		}
	}
	
	private void addC(int i, boolean addPrevRow) {
		boolean isCVector = C.getNumColumns() != ret.getNumCols();
		if(C.isInSparseFormat()) {
			Iterator<IJV> iter = C.getSparseBlockIterator(i, i+1);
			while(iter.hasNext()) {
				IJV ijv = iter.next();
				if(!isCVector) {
					if(addPrevRow)
						denseBlock[ijv.getI()*numRetCols + ijv.getJ()] += ijv.getV() * denseBlock[(ijv.getI()-1)*numRetCols + ijv.getJ()];
					else
						denseBlock[ijv.getI()*numRetCols + ijv.getJ()] += ijv.getV() * denseBlock[(ijv.getI()+1)*numRetCols + ijv.getJ()];
				}
				else {
					double val = ijv.getV();
					for(int j = ijv.getI()*numRetCols; j < (ijv.getI()+1)*numRetCols; j++) {
						double val1 = addPrevRow ? denseBlock[(ijv.getI()-1)*numRetCols + ijv.getJ()] : denseBlock[(ijv.getI()+1)*numRetCols + ijv.getJ()];
						denseBlock[j] += val*val1;
					}
				}
			}
		}
		else {
			double [] CBlk = C.getDenseBlockValues();
			if(CBlk != null) {
				if(!isCVector) {
					for(int j = i*numRetCols; j < (i+1)*numRetCols; j++) {
						double val1 = addPrevRow ? denseBlock[j-numRetCols] : denseBlock[j+numRetCols];
						denseBlock[j] += CBlk[j]*val1;
					}
				}
				else {
					for(int j = i*numRetCols; j < (i+1)*numRetCols; j++) {
						double val1 = addPrevRow ? denseBlock[j-numRetCols] : denseBlock[j+numRetCols];
						denseBlock[j] += CBlk[i]*val1;
					}
				}
			}
		}
	}
	
	private void allocateOutput() {
		String dir = createOutputFilePathAndName( "TMP" );
		ret = new Matrix( dir, numRetRows, numRetCols, ValueType.Double );
		retMB = new MatrixBlock((int) numRetRows, (int) numRetCols, false);
		retMB.allocateDenseBlock();
	}
}
