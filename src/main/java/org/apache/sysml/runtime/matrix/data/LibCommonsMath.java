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

package org.apache.sysml.runtime.matrix.data;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.CholeskyDecomposition;
import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.QRDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.util.DataConverter;

/**
 * Library for matrix operations that need invocation of 
 * Apache Commons Math library. 
 * 
 * This library currently supports following operations:
 * matrix inverse, matrix decompositions (QR, LU, Eigen), solve 
 */
public class LibCommonsMath 
{	
	private LibCommonsMath() {
		//prevent instantiation via private constructor
	}
	
	public static boolean isSupportedUnaryOperation( String opcode ) {
		return ( opcode.equals("inverse") || opcode.equals("cholesky") );
	}
	
	public static boolean isSupportedMultiReturnOperation( String opcode ) {
		return ( opcode.equals("qr") || opcode.equals("lu") || opcode.equals("eigen") );
	}
	
	public static boolean isSupportedMatrixMatrixOperation( String opcode ) {
		return ( opcode.equals("solve") );
	}
		
	public static MatrixBlock unaryOperations(MatrixObject inj, String opcode) 
		throws DMLRuntimeException 
	{
		Array2DRowRealMatrix matrixInput = DataConverter.convertToArray2DRowRealMatrix(inj);
		if(opcode.equals("inverse"))
			return computeMatrixInverse(matrixInput);
		else if (opcode.equals("cholesky"))
			return computeCholesky(matrixInput);		
		return null;
	}
	
	public static MatrixBlock[] multiReturnOperations(MatrixObject in, String opcode) 
		throws DMLRuntimeException 
	{
		if(opcode.equals("qr"))
			return computeQR(in);
		else if (opcode.equals("lu"))
			return computeLU(in);
		else if (opcode.equals("eigen"))
			return computeEigen(in);
		return null;
	}
	
	public static MatrixBlock matrixMatrixOperations(MatrixObject in1, MatrixObject in2, String opcode) 
		throws DMLRuntimeException 
	{
		if(opcode.equals("solve"))
			return computeSolve(in1, in2);
		return null;
	}
	
	/**
	 * Function to solve a given system of equations.
	 * 
	 * @param in1
	 * @param in2
	 * @return
	 * @throws DMLRuntimeException
	 */
	private static MatrixBlock computeSolve(MatrixObject in1, MatrixObject in2) 
		throws DMLRuntimeException 
	{
		Array2DRowRealMatrix matrixInput = DataConverter.convertToArray2DRowRealMatrix(in1);
		Array2DRowRealMatrix vectorInput = DataConverter.convertToArray2DRowRealMatrix(in2);
		
		/*LUDecompositionImpl ludecompose = new LUDecompositionImpl(matrixInput);
		DecompositionSolver lusolver = ludecompose.getSolver();
		RealMatrix solutionMatrix = lusolver.solve(vectorInput);*/
		
		// Setup a solver based on QR Decomposition
		QRDecomposition qrdecompose = new QRDecomposition(matrixInput);
		DecompositionSolver solver = qrdecompose.getSolver();
		// Invoke solve
		RealMatrix solutionMatrix = solver.solve(vectorInput);
		
		return DataConverter.convertToMatrixBlock(solutionMatrix.getData());
	}
	
	/**
	 * Function to perform QR decomposition on a given matrix.
	 * 
	 * @param in
	 * @return
	 * @throws DMLRuntimeException
	 */
	private static MatrixBlock[] computeQR(MatrixObject in) 
		throws DMLRuntimeException 
	{
		Array2DRowRealMatrix matrixInput = DataConverter.convertToArray2DRowRealMatrix(in);
		
		// Perform QR decomposition
		QRDecomposition qrdecompose = new QRDecomposition(matrixInput);
		RealMatrix H = qrdecompose.getH();
		RealMatrix R = qrdecompose.getR();
		
		// Read the results into native format
		MatrixBlock mbH = DataConverter.convertToMatrixBlock(H.getData());
		MatrixBlock mbR = DataConverter.convertToMatrixBlock(R.getData());

		return new MatrixBlock[] { mbH, mbR };
	}
	
	/**
	 * Function to perform LU decomposition on a given matrix.
	 * 
	 * @param in
	 * @return
	 * @throws DMLRuntimeException
	 */
	private static MatrixBlock[] computeLU(MatrixObject in) 
		throws DMLRuntimeException 
	{
		if ( in.getNumRows() != in.getNumColumns() ) {
			throw new DMLRuntimeException("LU Decomposition can only be done on a square matrix. Input matrix is rectangular (rows=" + in.getNumRows() + ", cols="+ in.getNumColumns() +")");
		}
		
		Array2DRowRealMatrix matrixInput = DataConverter.convertToArray2DRowRealMatrix(in);
		
		// Perform LUP decomposition
		LUDecomposition ludecompose = new LUDecomposition(matrixInput);
		RealMatrix P = ludecompose.getP();
		RealMatrix L = ludecompose.getL();
		RealMatrix U = ludecompose.getU();
		
		// Read the results into native format
		MatrixBlock mbP = DataConverter.convertToMatrixBlock(P.getData());
		MatrixBlock mbL = DataConverter.convertToMatrixBlock(L.getData());
		MatrixBlock mbU = DataConverter.convertToMatrixBlock(U.getData());

		return new MatrixBlock[] { mbP, mbL, mbU };
	}
	
	/**
	 * Function to perform Eigen decomposition on a given matrix.
	 * Input must be a symmetric matrix.
	 * 
	 * @param in
	 * @return
	 * @throws DMLRuntimeException
	 */
	private static MatrixBlock[] computeEigen(MatrixObject in)
		throws DMLRuntimeException 
	{
		if ( in.getNumRows() != in.getNumColumns() ) {
			throw new DMLRuntimeException("Eigen Decomposition can only be done on a square matrix. Input matrix is rectangular (rows=" + in.getNumRows() + ", cols="+ in.getNumColumns() +")");
		}
		
		Array2DRowRealMatrix matrixInput = DataConverter.convertToArray2DRowRealMatrix(in);
		
		EigenDecomposition eigendecompose = new EigenDecomposition(matrixInput);
		RealMatrix eVectorsMatrix = eigendecompose.getV();
		double[][] eVectors = eVectorsMatrix.getData();
		double[] eValues = eigendecompose.getRealEigenvalues();
		
		//Sort the eigen values (and vectors) in increasing order (to be compatible w/ LAPACK.DSYEVR())
		int n = eValues.length;
		for (int i = 0; i < n; i++) {
		    int k = i;
		    double p = eValues[i];
		    for (int j = i + 1; j < n; j++) {
		        if (eValues[j] < p) {
		            k = j;
		            p = eValues[j];
		        }
		    }
		    if (k != i) {
		        eValues[k] = eValues[i];
		        eValues[i] = p;
		        for (int j = 0; j < n; j++) {
		            p = eVectors[j][i];
		            eVectors[j][i] = eVectors[j][k];
		            eVectors[j][k] = p;
		        }
		    }
		}

		MatrixBlock mbValues = DataConverter.convertToMatrixBlock(eValues, true);
		MatrixBlock mbVectors = DataConverter.convertToMatrixBlock(eVectors);

		return new MatrixBlock[] { mbValues, mbVectors };
	}
	
	/**
	 * Function to compute matrix inverse via matrix decomposition.
	 * 
	 * @param in
	 * @return
	 * @throws DMLRuntimeException
	 */
	private static MatrixBlock computeMatrixInverse(Array2DRowRealMatrix in) 
		throws DMLRuntimeException 
	{	
		if ( !in.isSquare() )
			throw new DMLRuntimeException("Input to inv() must be square matrix -- given: a " + in.getRowDimension() + "x" + in.getColumnDimension() + " matrix.");
		
		QRDecomposition qrdecompose = new QRDecomposition(in);
		DecompositionSolver solver = qrdecompose.getSolver();
		RealMatrix inverseMatrix = solver.getInverse();

		return DataConverter.convertToMatrixBlock(inverseMatrix.getData());
	}

	/**
	 * Function to compute Cholesky decomposition of the given input matrix. 
	 * The input must be a real symmetric positive-definite matrix.
	 * 
	 * @param in
	 * @return
	 * @throws DMLRuntimeException
	 */
	private static MatrixBlock computeCholesky(Array2DRowRealMatrix in) 
		throws DMLRuntimeException 
	{	
		if ( !in.isSquare() )
			throw new DMLRuntimeException("Input to cholesky() must be square matrix -- given: a " + in.getRowDimension() + "x" + in.getColumnDimension() + " matrix.");

		CholeskyDecomposition cholesky = new CholeskyDecomposition(in);
		RealMatrix rmL = cholesky.getL();
		
		return DataConverter.convertToMatrixBlock(rmL.getData());
	}
}
