package com.ibm.bi.dml.packagesupport;

import java.util.Arrays;
import java.util.Comparator;

import com.ibm.bi.dml.packagesupport.Matrix;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.packagesupport.Matrix.ValueType;
import com.ibm.bi.dml.packagesupport.PackageFunction;
import com.ibm.bi.dml.packagesupport.PackageRuntimeException;

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
 * (classname="com.ibm.bi.dml.packagesupport.PermutationMatrixWrapper"
 * ,exectype="mem"); A = read( "Data/A.mtx"); P = permutation_matrix( A[,2],
 * false); B = P %*% A
 * 
 */

public class PermutationMatrixWrapper extends PackageFunction {
	private static final long serialVersionUID = 1L;
	private final String OUTPUT_FILE = "TMP";

	// return matrix
	private Matrix _ret;

	@Override
	public int getNumFunctionOutputs() {
		return 1;
	}

	@Override
	public FIO getFunctionOutput(int pos) {
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
	private class AscRowComparator implements Comparator<double[]> {
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
	private class DescRowComparator implements Comparator<double[]> {
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
