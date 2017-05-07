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
import java.util.Random;

import org.apache.sysml.runtime.controlprogram.caching.CacheException;
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
 * Use this class to perform an SGD update with Nesterov momentum in CP.
 * Assumption: the input batch fits in CP (which is also the assumption of most deep learning systems).
 * 
 * Usage:
 * update_nesterov = externalFunction(matrix[double] X, matrix[double] dX, double lr, double mu, matrix[double] v, double lambda) return (matrix[double] X, matrix[double] v) implemented in (classname="org.apache.sysml.udf.lib.SGDNesterovUpdate",exectype="mem");
 * [X, v] = update_nesterov(X, dX, lr, mu, v);
 * 
 * 
 * This class eliminates the unnecessary instruction overhead as well as memory pressure. 
 * 
 */
public class SGDNesterovUpdate extends PackageFunction {
	private static final long serialVersionUID = -3905212831582648882L;

	private Matrix updatedX;
	private Matrix updatedV;
	private Random rand = new Random();
	
	@Override
	public int getNumFunctionOutputs() {
		return 2;
	}

	@Override
	public FunctionParameter getFunctionOutput(int pos) {
		if(pos == 0)
			return updatedX;
		else if(pos == 1)
			return updatedV;
		
		throw new RuntimeException("Invalid function output being requested");
	}
	
	boolean isDense(MatrixBlock X) {
		return !X.isInSparseFormat() && X.getDenseBlock() != null;
	}

	@Override
	public void execute() {
		try {
			MatrixBlock X = ((Matrix) getFunctionInput(0)).getMatrixObject().acquireRead();
			MatrixBlock dX = ((Matrix) getFunctionInput(1)).getMatrixObject().acquireRead();
			double lr = Double.parseDouble(((Scalar)getFunctionInput(2)).getValue());
			double mu = Double.parseDouble(((Scalar)getFunctionInput(3)).getValue());
			MatrixBlock v = ((Matrix) getFunctionInput(4)).getMatrixObject().acquireRead();
			
			double lambda = Double.parseDouble(((Scalar)getFunctionInput(5)).getValue());
			
			// v = mu * v - lr * dX - lr*lambda*X
			updatedV = new Matrix( "tmp_" + rand.nextLong(), v.getNumRows(), v.getNumColumns(), ValueType.Double );
			MatrixBlock updatedVMB = allocateDenseMatrixBlock(updatedV);
			double [] updatedVData = updatedVMB.getDenseBlock();
			if(isDense(v) && isDense(dX) && isDense(X)) {
				double [] vArr = v.getDenseBlock();
				double [] dXArr = dX.getDenseBlock();
				double [] XArr = X.getDenseBlock();
				int nnz = 0;
				for(int i = 0; i < updatedVData.length; i++) {
					updatedVData[i] = mu*vArr[i] - lr*dXArr[i] - lr*lambda*XArr[i];
					nnz += (updatedVData[i]!=0) ? 1 : 0;
				}
				updatedVMB.setNonZeros(nnz); 
			}
			else {
				multiplyByConstant(v, mu, updatedVData);
				multiplyByConstant(dX, -lr, updatedVData);
				multiplyByConstant(X, -lr*lambda, updatedVData);
				updatedVMB.recomputeNonZeros();
			}
			
			updatedV.setMatrixDoubleArray(updatedVMB, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo);
			
			// X = X - mu * v_prev + (1 + mu) * v
			updatedX = new Matrix( "tmp_" + rand.nextLong(), X.getNumRows(), X.getNumColumns(), ValueType.Double );
			MatrixBlock updatedXMB = allocateDenseMatrixBlock(updatedX);
			double [] updatedXData = updatedXMB.getDenseBlock();
			if(isDense(X) && isDense(v)) {
				double [] XArr = X.getDenseBlock();
				double [] vPrevArr = v.getDenseBlock();
				int nnz = 0; double muPlus1 = mu+1;
				for(int i = 0; i < updatedXData.length; i++) {
					updatedXData[i] = XArr[i] - mu*vPrevArr[i] + muPlus1*updatedVData[i];
					nnz += (updatedXData[i]!=0) ? 1 : 0;
				}
				updatedXMB.setNonZeros(nnz); 
			}
			else if(isDense(v)) {
				copy(X, updatedXData);
				double [] vPrevArr = v.getDenseBlock();
				int nnz = 0; double muPlus1 = mu+1;
				for(int i = 0; i < updatedXData.length; i++) {
					updatedXData[i] += - mu*vPrevArr[i] + muPlus1*updatedVData[i];
					nnz += (updatedXData[i]!=0) ? 1 : 0;
				}
				updatedXMB.setNonZeros(nnz);
			}
			else {
				copy(X, updatedXData);
				multiplyByConstant(v, -mu, updatedXData);
				multiplyByConstant(updatedVData, 1+mu, updatedXData);
				updatedXMB.recomputeNonZeros();
			}
			updatedX.setMatrixDoubleArray(updatedXMB, OutputInfo.BinaryBlockOutputInfo, InputInfo.BinaryBlockInputInfo);
			
			((Matrix) getFunctionInput(0)).getMatrixObject().release();
			((Matrix) getFunctionInput(1)).getMatrixObject().release();
			((Matrix) getFunctionInput(4)).getMatrixObject().release();
		} catch (CacheException e) {
			throw new RuntimeException("Exception while executing SGDNesterovUpdate", e);
		} catch (IOException e) {
			throw new RuntimeException("Exception while executing SGDNesterovUpdate", e);
		}
	}
	
	private MatrixBlock allocateDenseMatrixBlock(Matrix mat) {
		int rows = (int) mat.getNumRows();
		int cols = (int) mat.getNumCols();
		MatrixBlock mb = new MatrixBlock(rows, cols, false);
		mb.allocateDenseBlock();
		return mb;
	}
	
	
	// out += constant*in
	private void multiplyByConstant(double [] in, double constant, double [] out) {
		for(int i = 0; i < out.length; i++) {
			out[i] += in[i]*constant;
		}
	}
	
	// out += constant*in
	private void multiplyByConstant(MatrixBlock in, double constant, double [] out) {
		if(in.isInSparseFormat()) {
			Iterator<IJV> iter = in.getSparseBlockIterator();
			while(iter.hasNext()) {
				IJV ijv = iter.next();
				out[ijv.getI()*ijv.getJ()] += ijv.getV() * constant;
			}
		}
		else {
			double [] denseBlock = in.getDenseBlock();
			if(denseBlock != null) {
				// If not empty block
				for(int i = 0; i < out.length; i++) {
					out[i] += denseBlock[i]*constant;
				}
			}
		}
	}
	
	// Assumption dest is zero-ed out.
	private void copy(MatrixBlock src, double [] dest) {
		if(src.isInSparseFormat()) {
			Iterator<IJV> iter = src.getSparseBlockIterator();
			while(iter.hasNext()) {
				IJV ijv = iter.next();
				dest[ijv.getI()*ijv.getJ()] = ijv.getV();
			}
		}
		else {
			double [] denseBlock = src.getDenseBlock();
			if(denseBlock != null) {
				// If not empty block
				System.arraycopy(denseBlock, 0, dest, 0, dest.length);
			}
		}
	}
}
