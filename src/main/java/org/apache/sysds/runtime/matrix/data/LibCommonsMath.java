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

package org.apache.sysds.runtime.matrix.data;

import static org.apache.sysds.runtime.matrix.data.LibMatrixFourier.fft;
import static org.apache.sysds.runtime.matrix.data.LibMatrixFourier.fft_linearized;
import static org.apache.sysds.runtime.matrix.data.LibMatrixFourier.ifft;
import static org.apache.sysds.runtime.matrix.data.LibMatrixFourier.ifft_linearized;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.commons.math3.exception.MaxCountExceededException;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.CholeskyDecomposition;
import org.apache.commons.math3.linear.DecompositionSolver;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.QRDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.codegen.CodegenUtils;
import org.apache.sysds.runtime.codegen.SpoofOperator.SideInput;
import org.apache.sysds.runtime.compress.utils.IntArrayList;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Divide;
import org.apache.sysds.runtime.functionobjects.MinusMultiply;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.SwapIndex;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.LeftScalarOperator;
import org.apache.sysds.runtime.matrix.operators.ReorgOperator;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.TernaryOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.UtilFunctions;

/**
 * Library for matrix operations that need invocation of 
 * Apache Commons Math library. 
 * 
 * This library currently supports following operations:
 * matrix inverse, matrix decompositions (QR, LU, Eigen), solve 
 */
public class LibCommonsMath 
{
	private static final Log LOG = LogFactory.getLog(LibCommonsMath.class.getName());
	private static final double RELATIVE_SYMMETRY_THRESHOLD = 1e-6;
	private static final double EIGEN_LAMBDA = 1e-8;

	private LibCommonsMath() {
		//prevent instantiation via private constructor
	}
	
	public static boolean isSupportedUnaryOperation( String opcode ) {
		return opcode.equals(Opcodes.INVERSE.toString())
			|| opcode.equals(Opcodes.CHOLESKY.toString()) 
			|| opcode.equals(Opcodes.SQRT_MATRIX_JAVA.toString());
	}
	
	public static boolean isSupportedMultiReturnOperation( String opcode ) {

		switch (opcode) {
			case "eigen":
			case "fft":
			case "fft_linearized":
			case "ifft":
			case "ifft_linearized":
			case "lu":
			case "qr":
			case "rcm":
			case "stft":
			case "svd": return true;
			default: return false;
		}

	}
	
	public static boolean isSupportedMatrixMatrixOperation( String opcode ) {
		return ( opcode.equals(Opcodes.SOLVE.toString()) );
	}
		
	public static MatrixBlock unaryOperations(MatrixBlock inj, String opcode) {
		if (opcode.equals(Opcodes.SQRT_MATRIX_JAVA.toString()))
			return computeSqrt(inj);
		else {
			Array2DRowRealMatrix matrixInput = DataConverter.convertToArray2DRowRealMatrix(inj);
			if(opcode.equals(Opcodes.INVERSE.toString()))
				return computeMatrixInverse(matrixInput);
			else if (opcode.equals(Opcodes.CHOLESKY.toString()))
				return computeCholesky(matrixInput);
		}
		return null;
	}

	// public static MatrixBlock[] multiReturnOperations(MatrixBlock in, String opcode) {
	// 	return multiReturnOperations(in, opcode, 1, 1);
	// }

	public static MatrixBlock[] multiReturnOperations(MatrixBlock in, String opcode, int threads) {
		return multiReturnOperations(in, opcode, threads, 1);
	}

	public static MatrixBlock[] multiReturnOperations(MatrixBlock in1, MatrixBlock in2, String opcode) {
		return multiReturnOperations(in1, in2, opcode, 1, 1);
	}

	public static MatrixBlock[] multiReturnOperations(MatrixBlock in, String opcode, int threads, int num_iterations, double tol) {
		if(opcode.equals("eigen_qr"))
			return computeEigenQR(in, num_iterations, tol, threads);
		else
			return multiReturnOperations(in, opcode, threads, 1);
	}

	public static MatrixBlock[] multiReturnOperations(MatrixBlock in, String opcode, int threads, long seed) {
		if(opcode.equals(Opcodes.QR.toString()))
			return computeQR(in);
		else if (opcode.equals("qr2"))
			return computeQR2(in, threads);
		else if (opcode.equals(Opcodes.LU.toString()))
			return computeLU(in);
		else if (opcode.equals(Opcodes.EIGEN.toString()))
			return computeEigen(in);
		else if (opcode.equals("eigen_lanczos"))
			return computeEigenLanczos(in, threads, seed);
		else if (opcode.equals("eigen_qr"))
			return computeEigenQR(in, threads);
		else if (opcode.equals(Opcodes.SVD.toString()))
			return computeSvd(in);
		else if (opcode.equals(Opcodes.FFT.toString()))
			return computeFFT(in, threads);
		else if (opcode.equals(Opcodes.IFFT.toString()))
			return computeIFFT(in, threads);
		else if (opcode.equals(Opcodes.FFT_LINEARIZED.toString()))
			return computeFFT_LINEARIZED(in, threads);
		else if (opcode.equals(Opcodes.IFFT_LINEARIZED.toString()))
			return computeIFFT_LINEARIZED(in, threads);
		return null;
	}

	public static MatrixBlock[] multiReturnOperations(MatrixBlock in1, MatrixBlock in2, String opcode, int threads,
			long seed) {

		switch (opcode) {
			case "ifft":
				return computeIFFT(in1, in2, threads);
			case "ifft_linearized":
				return computeIFFT_LINEARIZED(in1, in2, threads);
			case "rcm":
				return computeRCM(in1, in2);
			default:
				return null;
		}

	}

	public static MatrixBlock matrixMatrixOperations(MatrixBlock in1, MatrixBlock in2, String opcode) {
		if(opcode.equals(Opcodes.SOLVE.toString())) {
			if (in1.getNumRows() != in1.getNumColumns())
				throw new DMLRuntimeException("The A matrix, in solve(A,b) should have squared dimensions.");
			return computeSolve(in1, in2);
		}
		return null;
	}
	
	/**
	 * Function to solve a given system of equations.
	 * 
	 * @param in1 matrix object 1
	 * @param in2 matrix object 2
	 * @return matrix block
	 */
	private static MatrixBlock computeSolve(MatrixBlock in1, MatrixBlock in2) {
		//convert to commons math BlockRealMatrix instead of Array2DRowRealMatrix
		//to avoid unnecessary conversion as QR internally creates a BlockRealMatrix
		BlockRealMatrix matrixInput = DataConverter.convertToBlockRealMatrix(in1);
		BlockRealMatrix vectorInput = DataConverter.convertToBlockRealMatrix(in2);
		
		/*LUDecompositionImpl ludecompose = new LUDecompositionImpl(matrixInput);
		DecompositionSolver lusolver = ludecompose.getSolver();
		RealMatrix solutionMatrix = lusolver.solve(vectorInput);*/
		
		// Setup a solver based on QR Decomposition
		QRDecomposition qrdecompose = new QRDecomposition(matrixInput);
		DecompositionSolver solver = qrdecompose.getSolver();
		// Invoke solve
		RealMatrix solutionMatrix = solver.solve(vectorInput);
		
		return DataConverter.convertToMatrixBlock(solutionMatrix);
	}
	
	/**
	 * Function to perform QR decomposition on a given matrix.
	 * 
	 * @param in matrix object
	 * @return array of matrix blocks
	 */
	private static MatrixBlock[] computeQR(MatrixBlock in) {
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
	 * @param in matrix object
	 * @return array of matrix blocks
	 */
	private static MatrixBlock[] computeLU(MatrixBlock in) {
		if(in.getNumRows() != in.getNumColumns()) {
			throw new DMLRuntimeException(
				"LU Decomposition can only be done on a square matrix. Input matrix is rectangular (rows="
					+ in.getNumRows() + ", cols=" + in.getNumColumns() + ")");
		}
		
		Array2DRowRealMatrix matrixInput = DataConverter.convertToArray2DRowRealMatrix(in);

		// Perform LUP decomposition
		LUDecomposition ludecompose = new LUDecomposition(matrixInput);

		// check for singular matrix otherwise ludecompose.getP() will return null
		if(ludecompose.getDeterminant() == 0){
			throw new DMLRuntimeException("LU Decomposition can only be done on a non-singular matrix.");
		}

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
	 * @param in matrix object
	 * @return array of matrix blocks
	 */
	private static MatrixBlock[] computeEigen(MatrixBlock in) {
		if ( in.getNumRows() != in.getNumColumns() ) {
			throw new DMLRuntimeException("Eigen Decomposition can only be done on a square matrix. "
				+ "Input matrix is rectangular (rows=" + in.getNumRows() + ", cols="+ in.getNumColumns() +")");
		}
		
		EigenDecomposition eigendecompose = null;
		try {
			Array2DRowRealMatrix matrixInput = DataConverter.convertToArray2DRowRealMatrix(in);
			eigendecompose = new EigenDecomposition(matrixInput);
		}
		catch(MaxCountExceededException ex) {
			LOG.warn("Eigen: "+ ex.getMessage()+". Falling back to regularized eigen factorization.");
			eigendecompose = computeEigenRegularized(in);
		}
		
		RealMatrix eVectorsMatrix = eigendecompose.getV();
		double[][] eVectors = eVectorsMatrix.getData();
		double[] eValues = eigendecompose.getRealEigenvalues();

		return sortEVs(eValues, eVectors);
	}

	private static EigenDecomposition computeEigenRegularized(MatrixBlock in) {
		if( in == null || in.isEmptyBlock(false) )
			throw new DMLRuntimeException("Invalid empty block");
		
		//slightly modify input for regularization (pos/neg)
		MatrixBlock in2 = new MatrixBlock(in, false);
		DenseBlock a = in2.getDenseBlock();
		for( int i=0; i<in2.rlen; i++ ) {
			double[] avals = a.values(i);
			int apos = a.pos(i);
			for( int j=0; j<in2.clen; j++ ) {
				double v = avals[apos+j];
				avals[apos+j] += Math.signum(v) * EIGEN_LAMBDA;
			}
		}
		
		//run eigen decomposition
		return new EigenDecomposition(
			DataConverter.convertToArray2DRowRealMatrix(in2));
	}

	/**
	 * Function to perform FFT on a given matrix.
	 *
	 * @param re      matrix object
	 * @param threads number of threads
	 * @return array of matrix blocks
	 */
	private static MatrixBlock[] computeFFT(MatrixBlock re, int threads) {
		if(re == null)
			throw new DMLRuntimeException("Invalid empty block");
		if(re.isEmptyBlock(false)) {
			// Return the original matrix as the result
			// Assuming you need to return two matrices: the real part and an imaginary part initialized to 0.
			return new MatrixBlock[] {re, new MatrixBlock(re.getNumRows(), re.getNumColumns(), true)};
		}
		// run fft
		re.sparseToDense();
		return fft(re, threads);
	}

	/**
	 * Function to perform IFFT on a given matrix.
	 *
	 * @param re      matrix object
	 * @param im      matrix object
	 * @param threads number of threads
	 * @return array of matrix blocks
	 */
	private static MatrixBlock[] computeIFFT(MatrixBlock re, MatrixBlock im, int threads) {
		if(re == null)
			throw new DMLRuntimeException("Invalid empty block");

		// run ifft
		if(im != null && !im.isEmptyBlock(false)) {
			re.sparseToDense();
			im.sparseToDense();
			return ifft(re, im, threads);
		}
		else {
			if(re.isEmptyBlock(false)) {
				// Return the original matrix as the result
				// Assuming you need to return two matrices: the real part and an imaginary part initialized to 0.
				return new MatrixBlock[] {re, new MatrixBlock(re.getNumRows(), re.getNumColumns(), true)};
			}
			re.sparseToDense();
			return ifft(re, threads);
		}
	}

	/**
	 * Function to perform IFFT on a given matrix.
	 *
	 * @param re      matrix object
	 * @param threads number of threads
	 * @return array of matrix blocks
	 */
	private static MatrixBlock[] computeIFFT(MatrixBlock re, int threads) {
		return computeIFFT(re, null, threads);
	}

	/**
	 * Function to perform FFT_LINEARIZED on a given matrix.
	 *
	 * @param re      matrix object
	 * @param threads number of threads
	 * @return array of matrix blocks
	 */
	private static MatrixBlock[] computeFFT_LINEARIZED(MatrixBlock re, int threads) {
		if(re == null)
			throw new DMLRuntimeException("Invalid empty block");
		if(re.isEmptyBlock(false)) {
			// Return the original matrix as the result
			// Assuming you need to return two matrices: the real part and an imaginary part initialized to 0.
			return new MatrixBlock[] {re, new MatrixBlock(re.getNumRows(), re.getNumColumns(), true)};
		}
		// run fft
		re.sparseToDense();
		return fft_linearized(re, threads);
	}

	/**
	 * Function to perform IFFT_LINEARIZED on a given matrix.
	 *
	 * @param re      matrix object
	 * @param im      matrix object
	 * @param threads number of threads
	 * @return array of matrix blocks
	 */
	private static MatrixBlock[] computeIFFT_LINEARIZED(MatrixBlock re, MatrixBlock im, int threads) {
		if(re == null)
			throw new DMLRuntimeException("Invalid empty block");

		// run ifft
		if(im != null && !im.isEmptyBlock(false)) {
			re.sparseToDense();
			im.sparseToDense();
			return ifft_linearized(re, im, threads);
		}
		else {
			if(re.isEmptyBlock(false)) {
				// Return the original matrix as the result
				// Assuming you need to return two matrices: the real part and an imaginary part initialized to 0.
				return new MatrixBlock[] {re, new MatrixBlock(re.getNumRows(), re.getNumColumns(), true)};
			}
			re.sparseToDense();
			return ifft_linearized(re, threads);
		}
	}

	/**
	 * Function to perform IFFT_LINEARIZED on a given matrix
	 * 
	 * @param re      matrix object
	 * @param threads number of threads
	 * @return array of matrix blocks
	 */
	private static MatrixBlock[] computeIFFT_LINEARIZED(MatrixBlock re, int threads) {
		return computeIFFT_LINEARIZED(re, null, threads);
	}


	// /**
	//  * Function to perform STFT on a given matrix.
	//  *
	//  * @param re matrix object
	//  * @param im matrix object
	//  * @param windowSize of stft
	//  * @param overlap of stft
	//  * @return array of matrix blocks
	//  */
	// private static MatrixBlock[] computeSTFT(MatrixBlock re, MatrixBlock im, int windowSize, int overlap, int threads) {
	// 	if (re == null) {
	// 		throw new DMLRuntimeException("Invalid empty block");
	// 	} else if (im != null && !im.isEmptyBlock(false)) {
	// 		re.sparseToDense();
	// 		im.sparseToDense();
	// 		return stft(re, im, windowSize, overlap, threads);
	// 	} else {
	// 		if (re.isEmptyBlock(false)) {
	// 			// Return the original matrix as the result
	// 			int rows = re.getNumRows();
	// 			int cols = re.getNumColumns();

	// 			int stepSize = windowSize - overlap;
	// 			if (stepSize == 0) {
	// 				throw new IllegalArgumentException("windowSize - overlap is zero");
	// 			}

	// 			int numberOfFramesPerRow = (cols - overlap + stepSize - 1) / stepSize;
	// 			int rowLength= numberOfFramesPerRow * windowSize;
	// 			int out_len = rowLength * rows;

	// 			double[] out_zero = new double[out_len];

	// 			return new MatrixBlock[]{new MatrixBlock(rows, rowLength, out_zero), new MatrixBlock(rows, rowLength, out_zero)};
	// 			}
	// 		re.sparseToDense();
	// 		return stft(re, windowSize, overlap, threads);
	// 	}
	// }

	// /**
	//  * Function to perform STFT on a given matrix.
	//  *
	//  * @param re matrix object
	//  * @return array of matrix blocks
	//  */
	// private static MatrixBlock[] computeSTFT(MatrixBlock re, int windowSize, int overlap, int threads) {
	// 	return computeSTFT(re, null, windowSize, overlap, threads);
	// }

	/**
	 * Performs Singular Value Decomposition. Calls Apache Commons Math SVD.
	 * X = U * Sigma * Vt, where X is the input matrix,
	 * U is the left singular matrix, Sigma is the singular values matrix returned as a
	 * column matrix and Vt is the transpose of the right singular matrix V.
	 * However, the returned array has  { U, Sigma, V}
	 * 
	 * @param in Input matrix
	 * @return An array containing U, Sigma & V
	 */
	private static MatrixBlock[] computeSvd(MatrixBlock in) {
		Array2DRowRealMatrix matrixInput = DataConverter.convertToArray2DRowRealMatrix(in);

		SingularValueDecomposition svd = new SingularValueDecomposition(matrixInput);
		double[] sigma = svd.getSingularValues();
		RealMatrix u = svd.getU();
		RealMatrix v = svd.getV();
		MatrixBlock U = DataConverter.convertToMatrixBlock(u.getData());
		MatrixBlock Sigma = DataConverter.convertToMatrixBlock(sigma, true);
		Sigma = LibMatrixReorg.diag(Sigma, new MatrixBlock(Sigma.rlen, Sigma.rlen, true));
		MatrixBlock V = DataConverter.convertToMatrixBlock(v.getData());

		return new MatrixBlock[] { U, Sigma, V };
	}
	
	/**
	 * Computes the square root of a matrix Calls Apache Commons Math EigenDecomposition.
	 *
	 * @param in Input matrix
	 * @return matrix block
	 */
	private static MatrixBlock computeSqrt(MatrixBlock in) {
		Array2DRowRealMatrix matrixInput = DataConverter.convertToArray2DRowRealMatrix(in);
		EigenDecomposition ed = new EigenDecomposition(matrixInput);
		return DataConverter.convertToMatrixBlock(ed.getSquareRoot());
	}
	
	/**
	 * Function to compute matrix inverse via matrix decomposition.
	 * 
	 * @param in commons-math3 Array2DRowRealMatrix
	 * @return matrix block
	 */
	private static MatrixBlock computeMatrixInverse(Array2DRowRealMatrix in) {
		if(!in.isSquare())
			throw new DMLRuntimeException("Input to inv() must be square matrix -- given: a " + in.getRowDimension()
				+ "x" + in.getColumnDimension() + " matrix.");

		QRDecomposition qrdecompose = new QRDecomposition(in);
		DecompositionSolver solver = qrdecompose.getSolver();
		RealMatrix inverseMatrix = solver.getInverse();

		return DataConverter.convertToMatrixBlock(inverseMatrix.getData());
	}

	/**
	 * Function to compute Cholesky decomposition of the given input matrix. 
	 * The input must be a real symmetric positive-definite matrix.
	 * 
	 * @param in commons-math3 Array2DRowRealMatrix
	 * @return matrix block
	 */
	private static MatrixBlock computeCholesky(Array2DRowRealMatrix in) {
		if(!in.isSquare())
			throw new DMLRuntimeException("Input to cholesky() must be square matrix -- given: a "
				+ in.getRowDimension() + "x" + in.getColumnDimension() + " matrix.");
		CholeskyDecomposition cholesky = new CholeskyDecomposition(in, RELATIVE_SYMMETRY_THRESHOLD,
			CholeskyDecomposition.DEFAULT_ABSOLUTE_POSITIVITY_THRESHOLD);
		RealMatrix rmL = cholesky.getL();
		return DataConverter.convertToMatrixBlock(rmL.getData());
	}

	/**
	 * Creates a random normalized vector with dim elements.
	 *
	 * @param dim dimension of the created vector
	 * @param threads number of threads
	 * @param seed seed for the random MatrixBlock generation
	 * @return random normalized vector
	 */
	private static MatrixBlock randNormalizedVect(int dim, int threads, long seed) {
		MatrixBlock v1 = MatrixBlock.randOperations(dim, 1, 1.0, 0.0, 1.0, "UNIFORM", seed);

		double v1_sum = v1.sum();
		ScalarOperator op_div_scalar = new RightScalarOperator(Divide.getDivideFnObject(), v1_sum, threads);
		v1 = v1.scalarOperations(op_div_scalar, new MatrixBlock());
		UnaryOperator op_sqrt = new UnaryOperator(Builtin.getBuiltinFnObject(Builtin.BuiltinCode.SQRT), threads, true);
		v1 = v1.unaryOperations(op_sqrt, new MatrixBlock());

		if(Math.abs(v1.sumSq() - 1.0) >= 1e-7)
			throw new DMLRuntimeException("v1 not correctly normalized (maybe try changing the seed)");

		return v1;
	}

	/**
	 * Function to perform the Lanczos algorithm and then computes the Eigendecomposition.
	 * Caution: Lanczos is not numerically stable (see https://en.wikipedia.org/wiki/Lanczos_algorithm)
	 * Input must be a symmetric (and square) matrix.
	 *
	 * @param in matrix object
	 * @param threads number of threads
	 * @param seed seed for the random MatrixBlock generation
	 * @return array of matrix blocks
	 */
	private static MatrixBlock[] computeEigenLanczos(MatrixBlock in, int threads, long seed) {
		if(in.getNumRows() != in.getNumColumns()) {
			throw new DMLRuntimeException(
				"Lanczos algorithm and Eigen Decomposition can only be done on a square matrix. "
					+ "Input matrix is rectangular (rows=" + in.getNumRows() + ", cols=" + in.getNumColumns() + ")");
		}

		int m = in.getNumRows();
		MatrixBlock v0 = new MatrixBlock(m, 1, 0.0);
		MatrixBlock v1 = randNormalizedVect(m, threads, seed);

		MatrixBlock T = new MatrixBlock(m, m, 0.0);
		MatrixBlock TV = new MatrixBlock(m, m, 0.0);
		MatrixBlock w1;

		ReorgOperator op_t = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), threads);
		TernaryOperator op_minus_mul = new TernaryOperator(MinusMultiply.getFnObject(), threads);
		AggregateBinaryOperator op_mul_agg = InstructionUtils.getMatMultOperator(threads);
		ScalarOperator op_div_scalar = new RightScalarOperator(Divide.getDivideFnObject(), 1, threads);

		MatrixBlock beta = new MatrixBlock(1, 1, 0.0);
		for(int i = 0; i < m; i++) {
			v1.putInto(TV, 0, i, false);
			w1 = in.aggregateBinaryOperations(in, v1, op_mul_agg);
			MatrixBlock alpha = w1.aggregateBinaryOperations(v1.reorgOperations(op_t, new MatrixBlock(), 0, 0, m), w1, op_mul_agg);
			if(i < m - 1) {
				w1 = w1.ternaryOperations(op_minus_mul, v1, alpha, new MatrixBlock());
				w1 = w1.ternaryOperations(op_minus_mul, v0, beta, new MatrixBlock());
				beta.set(0, 0, Math.sqrt(w1.sumSq()));
				v0.copy(v1);
				op_div_scalar = op_div_scalar.setConstant(beta.getDouble(0, 0));
				w1.scalarOperations(op_div_scalar, v1);

				T.set(i + 1, i, beta.get(0, 0));
				T.set(i, i + 1, beta.get(0, 0));
			}
			T.set(i, i, alpha.get(0, 0));
		}

		MatrixBlock[] e = computeEigen(T);
		TV.setNonZeros((long) m*m);
		e[1] = TV.aggregateBinaryOperations(TV, e[1], op_mul_agg);
		return e;
	}

	/**
	 * Function to perform the QR decomposition.
	 * Input must be a square matrix.
	 * TODO: use Householder transformation and implicit shifts to further speed up QR decompositions
	 *
	 * @param in matrix object
	 * @param threads number of threads
	 * @return array of matrix blocks [Q, R]
	 */
	private static MatrixBlock[] computeQR2(MatrixBlock in, int threads) {
		if(in.getNumRows() != in.getNumColumns()) {
			throw new DMLRuntimeException("QR2 Decomposition can only be done on a square matrix. "
				+ "Input matrix is rectangular (rows=" + in.getNumRows() + ", cols=" + in.getNumColumns() + ")");
		}

		int m = in.rlen;

		MatrixBlock A_n = new MatrixBlock();
		A_n.copy(in);

		MatrixBlock Q_n = new MatrixBlock(m, m, true);
		for(int i = 0; i < m; i++) {
			Q_n.set(i, i, 1.0);
		}

		ReorgOperator op_t = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), threads);
		AggregateBinaryOperator op_mul_agg = InstructionUtils.getMatMultOperator(threads);
		BinaryOperator op_sub = InstructionUtils.parseExtendedBinaryOperator("-");
		ScalarOperator op_div_scalar = new RightScalarOperator(Divide.getDivideFnObject(), 1, threads);
		ScalarOperator op_mult_2 = new LeftScalarOperator(Multiply.getMultiplyFnObject(), 2, threads);

		for(int k = 0; k < m; k++) {
			MatrixBlock z = A_n.slice(k, m - 1, k, k);
			MatrixBlock uk = new MatrixBlock(m - k, 1, 0.0);
			uk.copy(z);
			uk.set(0, 0, uk.get(0, 0) + Math.signum(z.get(0, 0)) * Math.sqrt(z.sumSq()));
			op_div_scalar = op_div_scalar.setConstant(Math.sqrt(uk.sumSq()));
			uk = uk.scalarOperations(op_div_scalar, new MatrixBlock());

			MatrixBlock vk = new MatrixBlock(m, 1, 0.0);
			vk.copy(k, m - 1, 0, 0, uk, true);

			MatrixBlock vkt = vk.reorgOperations(op_t, new MatrixBlock(), 0, 0, m);
			MatrixBlock vkt2 = vkt.scalarOperations(op_mult_2, new MatrixBlock());
			MatrixBlock vkvkt2 = vk.aggregateBinaryOperations(vk, vkt2, op_mul_agg);

			A_n = A_n.binaryOperations(op_sub, A_n.aggregateBinaryOperations(vkvkt2, A_n, op_mul_agg));
			Q_n = Q_n.binaryOperations(op_sub, Q_n.aggregateBinaryOperations(Q_n, vkvkt2, op_mul_agg));
		}
		// QR decomp: Q: Q_n; R: A_n
		return new MatrixBlock[] {Q_n, A_n};
	}

	/**
	 * Function that computes the Eigen Decomposition using the QR algorithm.
	 * Caution: check if the QR algorithm is converged, if not increase iterations
	 * Caution: if the input matrix has complex eigenvalues results will be incorrect
	 *
	 * @param in Input matrix
	 * @param threads number of threads
	 * @return array of matrix blocks
	 */
	private static MatrixBlock[] computeEigenQR(MatrixBlock in, int threads) {
		return computeEigenQR(in, 100, 1e-10, threads);
	}

	private static MatrixBlock[] computeEigenQR(MatrixBlock in, int num_iterations, double tol, int threads) {
		if(in.getNumRows() != in.getNumColumns()) {
			throw new DMLRuntimeException("Eigen Decomposition (QR) can only be done on a square matrix. "
				+ "Input matrix is rectangular (rows=" + in.getNumRows() + ", cols=" + in.getNumColumns() + ")");
		}

		int m = in.rlen;
		AggregateBinaryOperator op_mul_agg = InstructionUtils.getMatMultOperator(threads);

		MatrixBlock Q_prod = new MatrixBlock(m, m, 0.0);
		for(int i = 0; i < m; i++) {
			Q_prod.set(i, i, 1.0);
		}

		for(int i = 0; i < num_iterations; i++) {
			MatrixBlock[] QR = computeQR2(in, threads);
			Q_prod = Q_prod.aggregateBinaryOperations(Q_prod, QR[0], op_mul_agg);
			in = QR[1].aggregateBinaryOperations(QR[1], QR[0], op_mul_agg);
		}

		// Is converged if all values are below tol and the there only is values on the diagonal.

		double[] check = in.getDenseBlockValues();
		double[] eval = new double[m];
		for(int i = 0; i < m; i++)
			eval[i] = check[i*m+i];
		
		double[] evec = Q_prod.getDenseBlockValues();
		return sortEVs(eval, evec);
	}

	/**
	 * Function to compute the Householder transformation of a Matrix.
	 *
	 * @param in Input Matrix
	 * @param threads number of threads
	 * @return transformed matrix
	 */
	@SuppressWarnings("unused")
	private static MatrixBlock computeHouseholder(MatrixBlock in, int threads) {
		int m = in.rlen;

		MatrixBlock A_n = new MatrixBlock(m, m, 0.0);
		A_n.copy(in);

		for(int k = 0; k < m - 2; k++) {
			MatrixBlock ajk = A_n.slice(0, m - 1, k, k);
			for(int i = 0; i <= k; i++) {
				ajk.set(i, 0, 0.0);
			}
			double alpha = Math.sqrt(ajk.sumSq());
			double ak1k = A_n.getDouble(k + 1, k);
			if(ak1k > 0.0)
				alpha *= -1;
			double r = Math.sqrt(0.5 * (alpha * alpha - ak1k * alpha));
			MatrixBlock v = new MatrixBlock(m, 1, 0.0);
			v.copy(ajk);
			v.set(k + 1, 0, ak1k - alpha);
			ScalarOperator op_div_scalar = new RightScalarOperator(Divide.getDivideFnObject(), 2 * r, threads);
			v = v.scalarOperations(op_div_scalar, new MatrixBlock());

			MatrixBlock P = new MatrixBlock(m, m, 0.0);
			for(int i = 0; i < m; i++) {
				P.set(i, i, 1.0);
			}

			ReorgOperator op_t = new ReorgOperator(SwapIndex.getSwapIndexFnObject(), threads);
			AggregateBinaryOperator op_mul_agg = InstructionUtils.getMatMultOperator(threads);
			BinaryOperator op_add = InstructionUtils.parseExtendedBinaryOperator("+");
			BinaryOperator op_sub = InstructionUtils.parseExtendedBinaryOperator("-");

			MatrixBlock v_t = v.reorgOperations(op_t, new MatrixBlock(), 0, 0, m);
			v_t = v_t.binaryOperations(op_add, v_t);
			MatrixBlock v_v_t_2 = A_n.aggregateBinaryOperations(v, v_t, op_mul_agg);
			P = P.binaryOperations(op_sub, v_v_t_2);
			A_n = A_n.aggregateBinaryOperations(P, A_n.aggregateBinaryOperations(A_n, P, op_mul_agg), op_mul_agg);
		}
		return A_n;
	}

	/**
	 * Sort the eigen values (and vectors) in increasing order (to be compatible w/ LAPACK.DSYEVR())
	 *
	 * @param eValues  Eigenvalues
	 * @param eVectors Eigenvectors
	 * @return array of matrix blocks
	 */
	private static MatrixBlock[] sortEVs(double[] eValues, double[][] eVectors) {
		int n = eValues.length;
		for(int i = 0; i < n; i++) {
			int k = i;
			double p = eValues[i];
			for(int j = i + 1; j < n; j++) {
				if(eValues[j] < p) {
					k = j;
					p = eValues[j];
				}
			}
			if(k != i) {
				eValues[k] = eValues[i];
				eValues[i] = p;
				for(int j = 0; j < n; j++) {
					p = eVectors[j][i];
					eVectors[j][i] = eVectors[j][k];
					eVectors[j][k] = p;
				}
			}
		}

		MatrixBlock eval = DataConverter.convertToMatrixBlock(eValues, true);
		MatrixBlock evec = DataConverter.convertToMatrixBlock(eVectors);
		return new MatrixBlock[] {eval, evec};
	}

	private static MatrixBlock[] sortEVs(double[] eValues, double[] eVectors) {
		int n = eValues.length;
		for(int i = 0; i < n; i++) {
			int k = i;
			double p = eValues[i];
			for(int j = i + 1; j < n; j++) {
				if(eValues[j] < p) {
					k = j;
					p = eValues[j];
				}
			}
			if(k != i) {
				eValues[k] = eValues[i];
				eValues[i] = p;
				for(int j = 0; j < n; j++) {
					p = eVectors[j*n+i];
					eVectors[j*n+i] = eVectors[j*n+k];
					eVectors[j*n+k] = p;
				}
			}
		}

		MatrixBlock eval = DataConverter.convertToMatrixBlock(eValues, true);
		MatrixBlock evec = new MatrixBlock(n, n, false);
		evec.init(eVectors, n, n);
		return new MatrixBlock[] {eval, evec};
	}
	
	/**
	 * Performs following operation:
	 * Computes the intersection ("meet") of equivalence classes for
	 * each row of A and B, excluding 0-valued cells.
	 * INPUT:
	 *   A, B = matrices whose rows contain that row's class labels;
	 *          for each i, rows A [i, ] and B [i, ] define two
	 *          equivalence relations on some of the columns, which
	 *          we want to intersect
	 *   A [i, j] == A [i, k] != 0 if and only if (j ~ k) as defined
	 *          by row A [i, ];
	 *   A [i, j] == 0 means that j is excluded by A [i, ]
	 *   B [i, j] is analogous
	 *   NOTE 1: Either nrow(A) == nrow(B), or exactly one of A or B
	 *   has one row that "applies" to each row of the other matrix.
	 *   NOTE 2: If ncol(A) != ncol(B), we pad extra 0-columns up to
	 *   max (ncol(A), ncol(B)).
	 * OUTPUT:
	 *   Both C and N have the same size as (the max of) A and B.
	 *   C = matrix whose rows contain class labels that represent
	 *       the intersection (coarsest common refinement) of the
	 *       corresponding rows of A and B.
	 *   C [i, j] == C [i, k] != 0 if and only if (j ~ k) as defined
	 *       by both A [i, ] and B [j, ]
	 *   C [i, j] == 0 if and only if A [i, j] == 0 or B [i, j] == 0
	 *       Additionally, we guarantee that non-0 labels in C [i, ]
	 *       will be integers from 1 to max (C [i, ]) without gaps.
	 *       For A and B the labels can be arbitrary.
	 *   N = matrix with class-size information for C-cells
	 *   N [i, j] = count of {C [i, k] | C [i, j] == C [i, k] != 0}
	 *
	 * @param A first input matrix
	 * @param B second input matrix
	 * @return output matrices C and N
	 */
	private static MatrixBlock[] computeRCM(MatrixBlock A, MatrixBlock B) {
		int nr = Math.max(A.getNumRows(), B.getNumRows());
		int nc = Math.max(A.getNumColumns(), B.getNumColumns());
		MatrixBlock C = new MatrixBlock(nr, nc, false).allocateBlock();
		MatrixBlock N = new MatrixBlock(nr, nc, false).allocateBlock();
		double[] dC = C.getDenseBlockValues();
		double[] dN = N.getDenseBlockValues();
		//wrap both A and B into side inputs for efficient sparse access
		SideInput sB = CodegenUtils.createSideInput(B);
		boolean mv = (B.getNumRows() == 1);
		int numCols = Math.min(A.getNumColumns(), B.getNumColumns());

		Map<ClassLabel, IntArrayList> classLabelMapping = new HashMap<>();
		for(int i=0, ai=0; i < A.getNumRows(); i++, ai+=A.getNumColumns()) {
			classLabelMapping.clear(); sB.reset();
			if( A.isInSparseFormat() ) {
				if(A.getSparseBlock()==null || A.getSparseBlock().isEmpty(i))
					continue;
				int alen = A.getSparseBlock().size(i);
				int apos = A.getSparseBlock().pos(i);
				int[] aix = A.getSparseBlock().indexes(i);
				double[] avals = A.getSparseBlock().values(i);
				for(int k=apos; k<apos+alen; k++) {
					if( aix[k] >= numCols ) break;
					int bval = (int)sB.getValue(mv?0:i, aix[k]);
					if( bval != 0 ) {
						ClassLabel key = new ClassLabel((int)avals[k], bval);
						if(!classLabelMapping.containsKey(key))
							classLabelMapping.put(key, new IntArrayList());
						classLabelMapping.get(key).appendValue(aix[k]);
					}
				}
			}
			else {
				double [] denseBlk = A.getDenseBlockValues();
				if(denseBlk == null) break;
				for(int j = 0; j < numCols; j++) {
					int aVal = (int) denseBlk[ai+j];
					int bVal = (int) sB.getValue(mv?0:i, j);
					if(aVal != 0 && bVal != 0) {
						ClassLabel key = new ClassLabel(aVal, bVal);
						if(!classLabelMapping.containsKey(key))
							classLabelMapping.put(key, new IntArrayList());
						classLabelMapping.get(key).appendValue(j);
					}
				}
			}

			int labelID = 1;
			for(Entry<ClassLabel, IntArrayList> entry : classLabelMapping.entrySet()) {
				int nVal = entry.getValue().size();
				int[] list = entry.getValue().extractValues();
				for(int k=0, off=i*nc; k<nVal; k++) {
					dN[off+list[k]] = nVal;
					dC[off+list[k]] = labelID;
				}
				labelID++;
			}
		}

		//prepare outputs
		C.recomputeNonZeros(); C.examSparsity();
		N.recomputeNonZeros(); N.examSparsity();
		return new MatrixBlock[] {C, N};
	}
	
	private static class ClassLabel {
		public int aVal;
		public int bVal;
		public ClassLabel(int aVal, int bVal) {
			this.aVal = aVal;
			this.bVal = bVal;
		}
		@Override
		public int hashCode() {
			return UtilFunctions.intHashCode(aVal, bVal);
		}
		@Override
		public boolean equals(Object o) {
			if( !(o instanceof ClassLabel) )
				return false;
			ClassLabel that = (ClassLabel) o;
			return aVal == that.aVal && bVal == that.bVal;
		}
	}
}
