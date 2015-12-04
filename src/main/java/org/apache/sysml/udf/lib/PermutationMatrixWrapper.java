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

import java.util.Arrays;
import java.util.Comparator;

import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.udf.FunctionParameter;
import org.apache.sysml.udf.Matrix;
import org.apache.sysml.udf.PackageFunction;
import org.apache.sysml.udf.PackageRuntimeException;
import org.apache.sysml.udf.Scalar;
import org.apache.sysml.udf.Matrix.ValueType;

/**
 * Wrapper class for Sorting and Creating of a Permutation Matrix
 * 
 * Sort single-column matrix and produce a permutation matrix. Pre-multiplying
 * the input matrix with the permutation matrix produces a sorted matrix. A
 * permutation matrix is a matrix where each row and each column as exactly one
 * 1: To From 1
 * 
 * Input: (n x 1)-matrix, and true/false for sorting in descending order Output:
 * (n x n)- matrix
 * 
 * permutation_matrix= externalFunction(Matrix[Double] A, Boolean desc) return
 * (Matrix[Double] P) implemented in
 * (classname="org.apache.sysml.udf.lib.PermutationMatrixWrapper"
 * ,exectype="mem"); A = read( "Data/A.mtx"); P = permutation_matrix( A[,2],
 * false); B = P %*% A
 * 
 */

public class PermutationMatrixWrapper extends PackageFunction 
{
	
	private static final long serialVersionUID = 1L;
	private static final String OUTPUT_FILE = "TMP";

	// return matrix
	private Matrix _ret;

	@Override
	public int getNumFunctionOutputs() {
		return 1;
	}

	@Override
	public FunctionParameter getFunctionOutput(int pos) {
		if (pos == 0)
			return _ret;

		throw new PackageRuntimeException(
				"Invalid function output being requested");
	}

	@Override
	public void execute() {
		try {
			Matrix inM = (Matrix) getFunctionInput(0);
			double[][] inData = inM.getMatrixAsDoubleArray();
			boolean desc = Boolean.parseBoolean(((Scalar) getFunctionInput(1))
					.getValue());

			// add index column as first column
			double[][] idxData = new double[(int) inM.getNumRows()][2];
			for (int i = 0; i < idxData.length; i++) {
				idxData[i][0] = i;
				idxData[i][1] = inData[i][0];
			}

			// sort input matrix (in-place)
			if (!desc) // asc
				Arrays.sort(idxData, new AscRowComparator(1));
			else
				// desc
				Arrays.sort(idxData, new DescRowComparator(1));

			// create and populate sparse matrixblock for result
			MatrixBlock mb = new MatrixBlock(idxData.length, idxData.length,
					true, idxData.length);
			for (int i = 0; i < idxData.length; i++) {
				mb.quickSetValue(i, (int) idxData[i][0], 1.0);
			}
			mb.examSparsity();

			// set result
			String dir = createOutputFilePathAndName(OUTPUT_FILE);
			_ret = new Matrix(dir, mb.getNumRows(), mb.getNumColumns(),
					ValueType.Double);
			_ret.setMatrixDoubleArray(mb, OutputInfo.BinaryBlockOutputInfo,
					InputInfo.BinaryBlockInputInfo);

		} catch (Exception e) {
			throw new PackageRuntimeException(
					"Error executing external permutation_matrix function", e);
		}
	}

	/**
	 * 
	 *
	 */
	private static class AscRowComparator implements Comparator<double[]> {
		private int _col = -1;

		public AscRowComparator(int col) {
			_col = col;
		}

		@Override
		public int compare(double[] arg0, double[] arg1) {
			return (arg0[_col] < arg1[_col] ? -1
					: (arg0[_col] == arg1[_col] ? 0 : 1));
		}
	}

	/**
	 * 
	 * 
	 */
	private static class DescRowComparator implements Comparator<double[]> {
		private int _col = -1;

		public DescRowComparator(int col) {
			_col = col;
		}

		@Override
		public int compare(double[] arg0, double[] arg1) {
			return (arg0[_col] > arg1[_col] ? -1
					: (arg0[_col] == arg1[_col] ? 0 : 1));
		}
	}
}
