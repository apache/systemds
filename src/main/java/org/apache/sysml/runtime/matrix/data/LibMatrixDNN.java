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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicLong;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.compress.CompressedMatrixBlock;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject.UpdateType;
import org.apache.sysml.runtime.functionobjects.Builtin;
import org.apache.sysml.runtime.functionobjects.KahanPlus;
import org.apache.sysml.runtime.functionobjects.Multiply;
import org.apache.sysml.runtime.functionobjects.Plus;
import org.apache.sysml.runtime.functionobjects.ValueFunction;
import org.apache.sysml.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysml.runtime.instructions.cp.KahanObject;
import org.apache.sysml.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysml.runtime.matrix.operators.AggregateOperator;
import org.apache.sysml.runtime.matrix.operators.BinaryOperator;
import org.apache.sysml.runtime.matrix.operators.UnaryOperator;
import org.apache.sysml.runtime.util.CommonThreadPool;
import org.apache.sysml.runtime.util.DnnUtils;
import org.apache.sysml.runtime.util.IndexRange;

/*
 * This class allows users to invoke deep learning related operations 
 * (such as conv2d, conv2d_backward_data, conv2d_backward_filter, maxpooling, maxpooling_backward, bias_add)
 * using multiple threads.
 * 
 * The methods accept the input matrices as MatrixBlock and the parameters using ConvolutionParameters.
 * 
 * To run in single thread, please set ConvolutionParameters.numThreads to 1.
 * 
 * DESIGN:
 * 
 * 1. LibMatrixDNN contains the user-facing methods for deep learning related operations. 
 * 2. The deep learning tasks are executed in parallel using java's ExecutorService. The key pattern
 * followed by the above mentioned functions are as follows:
 *   execute(LibMatrixDNNHelper.get__Workers(params), params);
 * 3. LibMatrixDNN's execute() method ensures the creation and shutdown of the ExecutorService.
 * 4. LibMatrixDNN__.getWorkers creates appropriate workers based on the runtime characteristics of
 * the input data (for example: input activations, filter, dout, ...). For code maintenance, these workers
 * are placed in the respective LibMatrixDNN__Helper files.
 * 5. The above mentioned workers may also use additional workers such as im2col and rotate180.
 * We have created similar get__Workers methods to return the appropriate worker based on the
 * runtime characteristics.
 * 6. As opposed to earlier implementation, this design reduces branch misprediction as well 
 * as instruction cache misses. It also allows us to experiment with new operators for different
 * data characteristics without affecting the performance of other operators. 
 * 7. This class assumes that the caller (for CP ConvolutionCPInstruction) deals with the empty block cases.  
 * 
 */
public class LibMatrixDNN {
	
	protected static final Log LOG =  LogFactory.getLog(LibMatrixDNN.class.getName());
	public static enum PoolingType {
		MAX, AVG
	}
	
	//library configurations and external contracts
	// ------------------------------------------------------------------------------------------------
	private static AtomicLong conv2dSparseCount = new AtomicLong(0);
	private static AtomicLong conv2dDenseCount = new AtomicLong(0);
	private static AtomicLong conv2dBwdFilterSparseCount = new AtomicLong(0);
	private static AtomicLong conv2dBwdFilterDenseCount = new AtomicLong(0);
	private static AtomicLong conv2dBwdDataSparseCount = new AtomicLong(0);
	private static AtomicLong conv2dBwdDataDenseCount = new AtomicLong(0);
	private static AtomicLong im2colSparseCount = new AtomicLong(0);
	private static AtomicLong im2colDenseCount = new AtomicLong(0);
	private static AtomicLong maxPoolBwdSparseCount = new AtomicLong(0);
	private static AtomicLong maxPoolBwdDenseCount = new AtomicLong(0);
	static AtomicLong loopedConvMatMultTime = new AtomicLong(0);
	static AtomicLong loopedConvIm2ColTime = new AtomicLong(0);
	static AtomicLong loopedConvBwdFilterMatMultTime = new AtomicLong(0);
	static AtomicLong loopedConvBwdFilterIm2ColTime = new AtomicLong(0);
	static AtomicLong loopedConvBwdDataMatMultTime = new AtomicLong(0);
	static AtomicLong loopedConvBwdDataCol2ImTime = new AtomicLong(0);
	
	public static void appendStatistics(StringBuilder sb) {
		if(ConfigurationManager.isFinegrainedStatistics()) {
			sb.append("LibMatrixDNN dense count (conv/bwdF/bwdD/im2col/maxBwd):\t" 
					+ conv2dDenseCount.get() + "/"
					+ conv2dBwdFilterDenseCount.get() + "/"
					+ conv2dBwdDataDenseCount.get() + "/"
					+ im2colDenseCount.get() + "/"
					+ maxPoolBwdDenseCount.get() + ".\n");
			sb.append("LibMatrixDNN sparse count (conv/bwdF/bwdD/im2col/maxBwd):\t" 
					+ conv2dSparseCount.get() + "/"
					+ conv2dBwdFilterSparseCount.get() + "/"
					+ conv2dBwdDataSparseCount.get() + "/"
					+ im2colSparseCount.get() + "/"
					+ maxPoolBwdSparseCount.get() + ".\n");
			sb.append("LibMatrixDNN conv(im2col/matmult), bwdF (im2col/matmult), bwdD (col2im/matmult) time:\t" +
					String.format("%.3f", loopedConvIm2ColTime.get()*1e-9) + "/" +
					String.format("%.3f", loopedConvMatMultTime.get()*1e-9) + "/" + 
					String.format("%.3f", loopedConvBwdFilterIm2ColTime.get()*1e-9) + "/" +
					String.format("%.3f", loopedConvBwdFilterMatMultTime.get()*1e-9) + "/" +
					String.format("%.3f", loopedConvBwdDataCol2ImTime.get()*1e-9) + "/" +
					String.format("%.3f", loopedConvBwdDataMatMultTime.get()*1e-9) + " sec.\n");
		}
	}
	public static void resetStatistics() {
		conv2dDenseCount.set(0);
		conv2dBwdFilterDenseCount.set(0);
		conv2dBwdDataDenseCount.set(0);
		im2colDenseCount.set(0);
		maxPoolBwdDenseCount.set(0);
		
		conv2dSparseCount.set(0);
		conv2dBwdFilterSparseCount.set(0);
		conv2dBwdDataSparseCount.set(0);
		im2colSparseCount.set(0);
		maxPoolBwdSparseCount.set(0);
		
		loopedConvIm2ColTime.set(0);
		loopedConvMatMultTime.set(0);
		loopedConvBwdFilterMatMultTime.set(0);
		loopedConvBwdFilterIm2ColTime.set(0);
		loopedConvBwdDataMatMultTime.set(0);
		loopedConvBwdDataCol2ImTime.set(0);
	}

	// ------------------------------------------------------------------------------------------------
	
	/**
	 * This method performs convolution (i.e. cross-correlation) operation on input
	 * 
	 * @param input input batch 
	 * @param filter filter
	 * @param outputBlock output of convolution
	 * @param params convolution parameters
	 */
	public static void conv2d(MatrixBlock input, MatrixBlock filter, MatrixBlock outputBlock, DnnParameters params) {
		LibMatrixDNN.checkInputsConv2d(input, filter, outputBlock, params);
		if(params.bias != null && params.bias.isInSparseFormat())
			params.bias.sparseToDense(); // Since bias is extremely small array
		
		long nnz = execute(LibMatrixDNNConv2d.getConv2dWorkers(params), params);
		
		//post-processing: maintain nnz
		outputBlock.setNonZeros(nnz);
		outputBlock.examSparsity();
	}
	
	/**
	 * This method computes the backpropogation errors for previous layer of convolution operation
	 * 
	 * @param filter filter used in conv2d 
	 * @param dout errors from next layer
	 * @param outputBlock  output errors
	 * @param params convolution parameters
	 
*/
	public static void conv2dBackwardData(MatrixBlock filter, MatrixBlock dout, MatrixBlock outputBlock, DnnParameters params) {
		checkInputsConv2dBackwardData(filter, dout, outputBlock, params);
		
		long nnz = execute(LibMatrixDNNConv2d.getConv2dBackwardDataWorkers(params), params);
		
		//post-processing: maintain nnz
		outputBlock.setNonZeros(nnz);
		outputBlock.examSparsity();
	}
	
	/**
	 * This method computes the backpropogation errors for filter of convolution operation
	 * 
	 * @param input input image 
	 * @param dout errors from next layer
	 * @param outputBlock  output errors
	 * @param params convolution parameters
	 */
	public static void conv2dBackwardFilter(MatrixBlock input, MatrixBlock dout, MatrixBlock outputBlock, DnnParameters params) {
		checkInputsConv2dBackwardFilter(input, dout, outputBlock, params);
		
		execute(LibMatrixDNNConv2d.getConv2dBackwardFilterWorkers(params), params);
		
		//post-processing: maintain nnz
		outputBlock.recomputeNonZeros(); 
		outputBlock.examSparsity();
	}
	
	public static void pooling(MatrixBlock input, MatrixBlock output, DnnParameters params, PoolingType poolType) {
		params.input1 = input;
		params.output = output;
		
		if(input.getNumColumns() != params.C*params.H*params.W || input.getNumRows() != params.N) {
			throw new DMLRuntimeException("Incorrect input dimensions in maxpooling:" + input.getNumRows() + " " 
				+ input.getNumColumns() + " " + params.N + " " + params.C*params.H*params.W);
		}
		
		//materialize indexes unless basic case with stride=1 and pad=0
		if( !params.isStride1Pad0() || input.sparse )
			fillIndexesArray(params);
		
		long nnz = execute(LibMatrixDNNPooling.getPoolingWorkers(params, poolType), params);
		
		// post-processing: maintain nnz
		output.setNonZeros(nnz);
		output.examSparsity();
	}
	

	/**
	 * This method computes the backpropogation errors for previous layer of pooling operation
	 * 
	 * @param input input matrix
	 * @param dout dout matrix
	 * @param outputBlock output matrix
	 * @param params convolution parameters
	 * @param performReluBackward perform ReLU backward
	 * @param poolType type of pooling
	 */
	public static void poolingBackward(MatrixBlock input, MatrixBlock dout, MatrixBlock outputBlock, 
			DnnParameters params, boolean performReluBackward, PoolingType poolType) {
		params.input1 = input;
		params.input2 = dout;
		params.output = outputBlock;
		
		if(poolType == PoolingType.MAX && (input.getNumColumns() != params.C*params.H*params.W || input.getNumRows() != params.N)) {
			throw new DMLRuntimeException("Incorrect input dimensions in maxpooling_backward:" + input.getNumRows() + " " + input.getNumColumns() + " " + params.N + " " + params.K*params.P*params.Q);
		}

		if(dout.getNumColumns() != params.C*params.P*params.Q || dout.getNumRows() != params.N) {
			throw new DMLRuntimeException("Incorrect dout dimensions in pooling_backward:" + input.getNumRows() + " " + input.getNumColumns() + " " + params.N + " " + params.K*params.P*params.Q);
		}
		
		if(ConfigurationManager.isFinegrainedStatistics()) {
			boolean isSparse = (poolType == PoolingType.MAX) ? (input.isInSparseFormat() || dout.isInSparseFormat()) : dout.isInSparseFormat();
			if(isSparse)
				maxPoolBwdSparseCount.addAndGet(1);
			else
				maxPoolBwdDenseCount.addAndGet(1);
		}
		
		if (params.output.isInSparseFormat())
			throw new DMLRuntimeException("Sparse pooling_backward is not supported");

		if(poolType == PoolingType.AVG) {
			fillIndexesArray(params); 
		}
		else {
			if( !(params.input1.isInSparseFormat() && !params.input2.isInSparseFormat()) )
				fillIndexesArray(params); //not needed for sparse-dense	 
		}
		long nnz = execute(LibMatrixDNNPooling.getPoolingBackwardWorkers(params, performReluBackward, poolType), params);
		//post-processing: maintain nnz 
		outputBlock.setNonZeros(nnz);
		outputBlock.examSparsity();
	}
	
	private static MatrixBlock matmult(MatrixBlock matBlock1, MatrixBlock matBlock2, int numThreads) {
		AggregateBinaryOperator ab_op = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), 
				new AggregateOperator(0, Plus.getPlusFnObject()), numThreads);
		MatrixBlock main = (matBlock2 instanceof CompressedMatrixBlock) ? matBlock2 : matBlock1;
		MatrixBlock ret = main.aggregateBinaryOperations(matBlock1, matBlock2, new MatrixBlock(), ab_op);
		return ret;
	}
	
	private static MatrixBlock add(MatrixBlock matBlock1, MatrixBlock matBlock2) {
		return (MatrixBlock) matBlock1.binaryOperations(new BinaryOperator(Plus.getPlusFnObject()), matBlock2, new MatrixBlock());
	}
	private static MatrixBlock multiply(MatrixBlock matBlock1, MatrixBlock matBlock2) {
		return (MatrixBlock) matBlock1.binaryOperations(new BinaryOperator(Multiply.getMultiplyFnObject()), matBlock2, new MatrixBlock());
	}
	
	// sigmoid(0)*c_prev + sigmoid(0)*tanh(0);
	
	private static Builtin sigmoidOp = Builtin.getBuiltinFnObject(BuiltinCode.SIGMOID);
	private static Builtin tanhOp = Builtin.getBuiltinFnObject(BuiltinCode.TANH);
	private static MatrixBlock sigmoid(MatrixBlock in, int numThreads, boolean inPlace) {
		return (MatrixBlock) in.unaryOperations(new UnaryOperator(sigmoidOp, numThreads, inPlace), new MatrixBlock());
	}
	private static MatrixBlock tanh(MatrixBlock in, int numThreads, boolean inPlace) {
		return (MatrixBlock) in.unaryOperations(new UnaryOperator(tanhOp, numThreads, inPlace), new MatrixBlock());
	}
	
	public static void lstm(MatrixBlock X, MatrixBlock W, MatrixBlock b, MatrixBlock out0, MatrixBlock c0, 
			boolean return_seq, int N, int T, int D, int M,
			MatrixBlock out, MatrixBlock c, // output 
			MatrixBlock cache_out, MatrixBlock cache_c, MatrixBlock cache_ifog, // if null, the cache values are not computed
			int numThreads) {
		MatrixBlock out_prev = out0;
		MatrixBlock c_prev = c0;
		
		MatrixBlock W1 = W.slice(0, D-1);
		MatrixBlock W2 = W.slice(D, D+M-1);
		MatrixBlock c_t = null;
		MatrixBlock out_t = null;
		for(int t = 1; t <= T; t++) {
			MatrixBlock X_t = X.slice(0, N-1, (t-1)*D, t*D-1, new MatrixBlock());
			MatrixBlock ifog_raw = add(add(matmult(X_t, W1, numThreads), matmult(out_prev, W2, numThreads)), b);
			MatrixBlock i = ifog_raw.slice(0, N-1, 0, M-1, new MatrixBlock());
			MatrixBlock f = ifog_raw.slice(0, N-1, M, 2*M-1, new MatrixBlock());
			MatrixBlock o = ifog_raw.slice(0, N-1, 2*M, 3*M-1, new MatrixBlock());
			MatrixBlock g = ifog_raw.slice(0, N-1, 3*M, 4*M-1, new MatrixBlock());
			i = sigmoid(i, numThreads, true);
			f = sigmoid(f, numThreads, true);
			o = sigmoid(o, numThreads, true);
			g = tanh(g, numThreads, true);
			// c_t = f*c_prev + i*g
			c_t = add(multiply(f, c_prev) , multiply(i, g));
			// out_t = o*tanh(c)
			out_t = multiply(o, tanh(c_t, numThreads, false));
			if(return_seq) {
				out = out.leftIndexingOperations(out_t, 0, N-1, (t-1)*M, t*M-1, new MatrixBlock(), UpdateType.INPLACE);
			}
			out_prev = out_t;
			c_prev = c_t;
			
			// TODO: Add this when implementing lstm_backward
//			cache_out[t,] = matrix(out_t, rows=1, cols=N*M)  # reshape
//		    cache_c[t,] = matrix(c, rows=1, cols=N*M)  # reshape
//		    cache_ifog[t,] = matrix(cbind(ifo, g), rows=1, cols=N*4*M)  # reshape
		}
		if(out_t != null && !return_seq)
			out.copy(out_t);
		if(c_t != null)
			c.copy(c_t);
		else
			c.copy(c0);
		
	}
	
	/**
	 * This method computes the backpropagation errors for previous layer of relu operation
	 * 
	 * @param input input matrix
	 * @param dout errors from next layer
	 * @param outputBlock output matrix
	 * @param numThreads number of threads
	 */
	public static void reluBackward(MatrixBlock input, MatrixBlock dout, MatrixBlock outputBlock, int numThreads) {
		int N = input.getNumRows();
		DnnParameters params = new DnnParameters(N, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, numThreads);
		params.input1 = input;
		params.input2 = dout;
		params.output = outputBlock;
		if(input.getNumRows() != dout.getNumRows() || input.getNumColumns() != dout.getNumColumns()) {
			throw new DMLRuntimeException("Incorrect dimensions for relu_backward:" + 
				input.getNumRows() + " != " + dout.getNumRows() + " || " + input.getNumColumns() + " != " + dout.getNumColumns());
		}
		
		long nnz = execute(LibMatrixDNNRelu.getReluBackwardWorkers(params), params);
		
		// post-processing: maintain nnz
		outputBlock.setNonZeros(nnz);
		outputBlock.examSparsity();
	}
	
	/**
	 * Performs the operation corresponding to the DML script:
	 * ones = matrix(1, rows=1, cols=Hout*Wout)		
	 * output = input + matrix(bias %*% ones, rows=1, cols=F*Hout*Wout)
	 * This operation is often followed by conv2d and hence we have introduced bias_add(input, bias) built-in function
	 * 
	 * @param input input matrix
	 * @param bias bias matrix
	 * @param outputBlock output matrix
	 * @param numThreads number of threads
	 */
	public static void biasAdd(MatrixBlock input, MatrixBlock bias, MatrixBlock outputBlock, int numThreads) {
		int N = input.getNumRows();
		int K = bias.getNumRows();
		int PQ = input.getNumColumns() / K;
		
		if(bias.getNumColumns() != 1 || input.getNumColumns() % K != 0) {
			throw new DMLRuntimeException("Incorrect inputs for bias_add: input[" + N + " X " + input.getNumColumns()  + "] and bias[" + K + " X " + bias.getNumColumns() + "]");
		}
		
		double [] outputArray = outputBlock.getDenseBlockValues();
		if(input.isEmptyBlock()) {
			for(int n = 0;  n < N; n++) 
				DnnUtils.fillBias(bias, outputArray, n, n+1, N, K, PQ);
		}
		else {
			// Handles both dense and sparse inputs and copies it to dense output
			outputBlock.copy(input, false);
			if(bias.isInSparseFormat())
				bias.sparseToDense(); // Since bias is extremely small array
			double [] biasArr = bias.getDenseBlockValues();
			addBias(outputArray, biasArr, 1, N, K, PQ);
		}
		
		//post-processing: maintain nnz
		outputBlock.recomputeNonZeros(); 
		outputBlock.examSparsity();
	}
	
	/**
	 * Perform channel sum operation
	 * 
	 * @param input input matrix block
	 * @param outputBlock output matrix block
	 * @param C number of channels
	 * @param HW height X width
	 */
	public static void channelSums(MatrixBlock input, MatrixBlock outputBlock, int C, int HW) {
		double [] output = outputBlock.getDenseBlockValues();
		if(input.isInSparseFormat()) {
			SparseBlock sblock = input.getSparseBlock();
			for(int n = 0; n < input.getNumRows(); n++) {
				if( sblock.isEmpty(n) )
					continue;
				int apos = sblock.pos(n);
				int alen = sblock.size(n);
				int[] aix = sblock.indexes(n);
				double[] avals = sblock.values(n);
				
				// Iterate over the sparse block
				for(int j=apos; j<apos+alen; j++) {
					// Note: the input is of shape [N, CHW]
					int chw = aix[j];
					
					// Get individual zero-based c,h,w indexes from zero-based 'chw'
					int c = chw / HW;
					output[c] += avals[j];
				}
			}
		}
		else {
			double [] inArr = input.getDenseBlockValues();
			if(inArr != null) {
				KahanPlus kplus = KahanPlus.getKahanPlusFnObject();
				for(int c = 0; c < C; c++) {
					KahanObject sum = new KahanObject(0.0, 0.0);
					for(int n = 0; n < input.getNumRows(); n++) {
						int index =  n*C*HW + c*HW;
						for(int hw = 0; hw < HW; hw++, index++) {
							kplus.execute2(sum, inArr[index]);
						}
					}
					output[c] = sum._sum; 
				}
			}
		}
		outputBlock.recomputeNonZeros();
	}
	
	public static void batchNorm2DBackward(MatrixBlock image, MatrixBlock dout, MatrixBlock scale, double epsilon,  
			MatrixBlock resultSaveMean, MatrixBlock resultSaveInvVariance,
			MatrixBlock dX, MatrixBlock dScale, MatrixBlock dBias) {
		int N = image.getNumRows();
		int K = scale.getNumRows();
		int PQ = image.getNumColumns() / K;
		channelSums(image, dBias, K, PQ);
		// Since output
		if(dBias.isInSparseFormat())
			dBias.sparseToDense();
		if(dScale.isInSparseFormat())
			dScale.sparseToDense();
		if(dX.isInSparseFormat())
			dX.sparseToDense();
		// Very small matrices
		if(resultSaveMean.isInSparseFormat())
			resultSaveMean.sparseToDense();
		if(resultSaveInvVariance.isInSparseFormat())
			resultSaveInvVariance.sparseToDense();
		if(scale.isInSparseFormat())
			scale.sparseToDense();
		double [] dBiasArr = dBias.getDenseBlockValues();
		double [] dScaleArr = dScale.getDenseBlockValues();
		double [] dXArr = dX.getDenseBlockValues();
		double [] mean = resultSaveMean.getDenseBlockValues();
		double [] invVar = resultSaveInvVariance.getDenseBlockValues();
		double [] scaleArr = scale.getDenseBlockValues();
		// since K is relatively small, it reduces code complexity. We can avoid this in subsequent commits.
		mean = (mean==null) ? new double[K] : mean; 
		invVar = (invVar==null) ? new double[K] : invVar;
		scaleArr = (scaleArr == null) ? new double[K] : scaleArr;
		
		// TODO: Handle sparse image and dout cases:
		if(image.isInSparseFormat())
			image.sparseToDense();
		if(dout.isInSparseFormat())
			dout.sparseToDense();
		
		if(!image.isInSparseFormat() && !dout.isInSparseFormat()) {
			double [] imageArr = image.getDenseBlockValues();
			double [] doutArr = dout.getDenseBlockValues();
			double constant1 = Math.pow(N*PQ, -1);
			int KPQ = K*PQ;
			for(int k = 0; k < K; k++) {
				double dvar = 0; 
				double dmean_norm_branch = 0; double dmean_var_branch  = 0;
				double sumDout = 0; double sum = 0;
				for(int n = 0; n < N; n++) {
					int index = n*KPQ + k*PQ; 
					for(int pq = 0; pq < PQ; pq++, index++) {
						double doutVal = doutArr != null ? doutArr[index] : 0;
						double centered = imageArr != null ? imageArr[index] : 0;
						centered -= mean[k];
						double dnorm = doutVal*scaleArr[k];
						dvar -= 0.5*centered*Math.pow(invVar[k], 3)*dnorm;
						dmean_norm_branch -= dnorm*invVar[k];
						sum += centered * invVar[k] * doutVal;
						sumDout += doutVal;
						dmean_var_branch -= 2*constant1*centered;
					}
				}
				dBiasArr[k] = sumDout;
				dScaleArr[k] = sum;
				dmean_var_branch *= dvar;
				double dmean = dmean_norm_branch + dmean_var_branch;
				double dX_mean_branch = constant1*dmean;
				
				for(int n = 0; n < N; n++) {
					int index = n*KPQ + k*PQ; 
					for(int pq = 0; pq < PQ; pq++, index++) {
						double doutVal = doutArr != null ? doutArr[index] : 0;
						double centered = imageArr != null ? imageArr[index] : 0;
						centered -= mean[k];
						double dnorm = doutVal*scaleArr[k];
						double dX_norm_branch = dnorm*invVar[k];
						double dX_var_branch = 2*constant1*centered*dvar;
						dXArr[index] = dX_norm_branch + dX_mean_branch + dX_var_branch;
					}
				}
			}
		}
		else {
			throw new DMLRuntimeException("Sparse format is not yet supported for batch norm backward");
		}
		dBias.recomputeNonZeros();
		dScale.recomputeNonZeros();
		dX.recomputeNonZeros();
	}
	
	public static void batchNorm2D(MatrixBlock image, MatrixBlock scale, MatrixBlock bias, MatrixBlock runningMean, 
			MatrixBlock runningVar, String phase, double epsilon, double mu,
			MatrixBlock ret, MatrixBlock retRunningMean, MatrixBlock retRunningVar, 
			MatrixBlock resultSaveMean, MatrixBlock resultSaveInvVariance) {
		// Since bias, scale, runningMean, runningVar are extremely small array
		if(bias.isInSparseFormat())
			bias.sparseToDense();
		double [] biasArr = bias.getDenseBlockValues();
		if(scale.isInSparseFormat())
			scale.sparseToDense();
		double [] scaleArr = scale.getDenseBlockValues();
		if(runningMean.isInSparseFormat())
			runningMean.sparseToDense();
		double [] runningMeanArr = runningMean.getDenseBlockValues(); // ema_mean
		if(runningVar.isInSparseFormat())
			runningVar.sparseToDense(); 
		double [] runningVarArr = runningVar.getDenseBlockValues(); // ema_var
		
		double [] retRunningMeanArr = retRunningMean.getDenseBlockValues(); // ema_mean_upd
		double [] retRunningVarArr = retRunningVar.getDenseBlockValues(); // ema_var_upd
		double [] resultSaveMeanArr = resultSaveMean.getDenseBlockValues(); // cache_mean
		double [] resultSaveInvVarianceArr = resultSaveInvVariance.getDenseBlockValues(); // cache_inv_var
		
		int N = image.getNumRows();
		int K = bias.getNumRows(); // number of output channels
		int PQ = image.getNumColumns() / K; // output height X output width
		
		if(phase.equalsIgnoreCase("train")) { 
			computeBiasSumAndSumSquares(image, resultSaveMeanArr, resultSaveInvVarianceArr, K, PQ);
			int NPQ = N*PQ;
			for(int k = 0; k < K; k++) {
				double mean = resultSaveMeanArr[k] / NPQ;
				double var = resultSaveInvVarianceArr[k]/NPQ - Math.pow(mean, 2.0);
				resultSaveMeanArr[k] = mean;
				resultSaveInvVarianceArr[k] = Math.pow(Math.sqrt(var + epsilon), -1.0);
				retRunningMeanArr[k] = mu*((runningMeanArr!=null)?runningMeanArr[k]:0) + (1-mu)*mean;
				retRunningVarArr[k] = mu*((runningVarArr!=null)?runningVarArr[k]:0) + (1-mu)*mean;
			}
		}
		else if(phase.equalsIgnoreCase("test")) {
			copy(runningMean, retRunningMeanArr); // ema_mean_upd = ema_mean
			copy(runningVar, retRunningVarArr); // ema_var_upd = ema_var
			copy(runningMean, resultSaveMeanArr); // cache_mean = ema_mean
			double invSqrtEps = Math.pow(Math.sqrt(epsilon), -1.0);
			double [] inArr = runningVar.getDenseBlockValues();
			if(inArr != null) {
				for(int i = 0; i < inArr.length; i++) {
					resultSaveInvVarianceArr[i] = Math.pow(Math.sqrt(inArr[i] + epsilon), -1.0);
				}
			}
			else {
				Arrays.fill(resultSaveInvVarianceArr, invSqrtEps);
			}
		}
		else {
			throw new DMLRuntimeException("Incorrect mode: Expected either train or test, but found " + phase);
		}
		
		// Normalize, shift, and scale
		double [] retArr = ret.getDenseBlockValues();
		copy(image, retArr);
		if(resultSaveMean != null && resultSaveInvVariance != null && biasArr != null && scaleArr != null) {
			// Common scenario:
			int index = 0;
			for(int n = 0; n < N; n++) {
				for(int k = 0; k < K; k++) {
					for(int pq = 0; pq < PQ; pq++, index++) {
						retArr[index] = (retArr[index]-resultSaveMeanArr[k])*resultSaveInvVarianceArr[k]*scaleArr[k] + biasArr[k];
					}
				}
			}
		}
		else {
			addBias(retArr, resultSaveMeanArr, -1, N, K, PQ);
			multBias(retArr, resultSaveInvVarianceArr, N, K, PQ);
			multBias(retArr, scaleArr, N, K, PQ);
			addBias(retArr, biasArr, 1, N, K, PQ);
		}
		ret.recomputeNonZeros();
		retRunningMean.recomputeNonZeros();
		retRunningVar.recomputeNonZeros();
		resultSaveMean.recomputeNonZeros();
		resultSaveInvVariance.recomputeNonZeros();
	}
	
	private static void copy(MatrixBlock input, double [] output) {
		if(input.isInSparseFormat()) {
			SparseBlock sblock = input.getSparseBlock();
			int numCols = input.getNumColumns();
			for(int n = 0; n < input.getNumRows(); n++) {
				if( sblock.isEmpty(n) )
					continue;
				int apos = sblock.pos(n);
				int alen = sblock.size(n);
				int[] aix = sblock.indexes(n);
				double[] avals = sblock.values(n);
				
				// Iterate over the sparse block
				for(int j=apos; j<apos+alen; j++) {
					output[n*numCols + aix[j]] = avals[j];
				}
			}
		}
		else {
			double [] inputArr = input.getDenseBlockValues();
			if(inputArr != null) {
				System.arraycopy(inputArr, 0, output, 0, inputArr.length);
			}
			else {
				Arrays.fill(output, 0);
			}
		}
	}
	
	public static void addBias(double[] a, double[] bias, double biasMultiplier, int N, int K, int PQ) {
		if( bias == null )
			return;
		int index = 0;
		for(int n = 0; n < N; n++) {
			for(int k = 0; k < K; k++) {
				double biasVal = biasMultiplier*bias[k];
				for(int pq = 0; pq < PQ; pq++, index++)
					a[index] += biasVal;
			}
		}
	}
	
	public static void multBias(double[] a, double[] bias, int N, int K, int PQ) {
		if( bias == null ) {
			Arrays.fill(a, 0);
			return;
		}
		int index = 0;
		for(int n = 0; n < N; n++) {
			for(int k = 0; k < K; k++) {
				double biasVal = bias[k];
				for(int pq = 0; pq < PQ; pq++, index++)
					a[index] *= biasVal;
			}
		}
	}
	
	private static void computeBiasSumAndSumSquares(MatrixBlock image, double [] sumArr, double [] sumSquaresArr, int K, int PQ) {
		if(sumArr.length != K) {
			throw new DMLRuntimeException("Expected the length of array to be " + K + ", but instead is " + sumArr.length);
		}
		if(sumSquaresArr.length != K) {
			throw new DMLRuntimeException("Expected the length of array to be " + K + ", but instead is " + sumSquaresArr.length);
		}
		if(image.isInSparseFormat()) {
			SparseBlock sblock = image.getSparseBlock();
			for(int r = 0; r < image.getNumRows(); r++) {
				if( sblock.isEmpty(r) )
					continue;
				int apos = sblock.pos(r);
				int alen = sblock.size(r);
				int[] aix = sblock.indexes(r);
				double[] avals = sblock.values(r);
				for(int j=apos; j<apos+alen; j++) {
					int k = aix[j] / PQ;
					sumArr[k] += avals[j];
					sumSquaresArr[k] += Math.pow(avals[j], 2.0);
				}
			}
		}
		else {
			double [] X = image.getDenseBlockValues();
			int N = image.getNumRows();
			if(X != null) {
				int index = 0;
				for(int n = 0; n < N; n++) {
					for(int k = 0; k < K; k++) {
						for(int pq = 0; pq < PQ; pq++, index++) {
							sumArr[k] += X[index];
							sumSquaresArr[k] += Math.pow(X[index], 2.0);
						}
					}
				}
			}
		}
	}
	
	
	/**
	 * Performs the operation corresponding to the DML script:
	 * ones = matrix(1, rows=1, cols=Hout*Wout)		
	 * output = input * matrix(bias %*% ones, rows=1, cols=F*Hout*Wout)
	 * This operation is often followed by conv2d and hence we have introduced bias_multiply(input, bias) built-in function
	 * 
	 * @param input input matrix
	 * @param bias bias matrix
	 * @param outputBlock output matrix
	 * @param numThreads number of threads
	 */
	public static void biasMultiply(MatrixBlock input, MatrixBlock bias, MatrixBlock outputBlock, int numThreads) {
		int N = input.getNumRows();
		int K = bias.getNumRows();
		int PQ = input.getNumColumns() / K;
		
		DnnParameters params = new DnnParameters(N, PQ, -1, -1, K, -1, -1, -1, -1, -1, -1, numThreads);
		params.input1 = input;
		params.input2 = bias;
		params.output = outputBlock;
		
		if(bias.getNumColumns() != 1 || input.getNumColumns() % K != 0) {
			throw new DMLRuntimeException("Incorrect inputs for bias_multiply: input[" + N + " X " + input.getNumColumns()  + "] and bias[" + K + " X " + bias.getNumColumns() + "]");
		}
		
		if(!input.isEmptyBlock() && !bias.isEmptyBlock()) {
			// Handles both dense and sparse inputs and copies it to dense output
			outputBlock.copy(input);
			if(bias.isInSparseFormat())
				bias.sparseToDense(); // Since bias is extremely small array
			double [] biasArr = bias.getDenseBlockValues();
			if(!input.isInSparseFormat()) {
				double [] outputArray = outputBlock.getDenseBlockValues();
				int index = 0;
				for(int n = 0; n < N; n++) {
					for(int k = 0; k < K; k++) {
						double biasVal = biasArr[k];
						for(int pq = 0; pq < PQ; pq++, index++) {
							outputArray[index] *= biasVal;
						}
					}
				}
			}
			else {
				SparseBlock sblock = outputBlock.sparseBlock;
				// First delete those elements which will become zero 
				for(int k = 0; k < K; k++) {
					if(biasArr[k] == 0) {
						for(int n = 0; n < N; n++) {
							if( sblock.isEmpty(n) ) continue;
							sblock.deleteIndexRange(n, k*PQ, (k+1)*PQ);
						}
					}
				}
				// Then perform bias_multiply for non-zero bias entries
				for(int n = 0; n < N; n++) {
					if( sblock.isEmpty(n) ) continue;
					int apos = sblock.pos(n);
					int alen = sblock.size(n);
					int[] aix = sblock.indexes(n);
					double[] avals = sblock.values(n);
					for(int j=apos; j<apos+alen; j++) {
						int k = aix[j] / PQ; //aix[j] KPQ
						if(biasArr[k] != 0)
							avals[j] *= biasArr[k];
					}
				}
			}
			
			//post-processing: maintain nnz
			params.output.recomputeNonZeros();
			params.output.examSparsity();
		}
		else {
			params.output.setNonZeros(0);
		}
	}
	
	/**
	 * Executes the tasks in parallel using java's ExecutorService.
	 *  
	 * @param tasks deep learning related tasks
	 * @param params convolution parameters
	 */
	private static long execute(ArrayList<Callable<Long>> tasks, DnnParameters params) {
		int k = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		long lnnz = 0;
		try {
			if(k == 1) {
				// Single-threaded execution when called in parfor
				// this avoid unnecessary creation of threadpool.
				for(Callable<Long> task : tasks) {
					lnnz += task.call();
				}
			}
			else {
				ExecutorService pool = CommonThreadPool.get( Math.min(k, params.N) );
				List<Future<Long>> taskret = pool.invokeAll(tasks);
				pool.shutdown();
				for( Future<Long> task : taskret )
					lnnz += task.get();
			}
		} 
		catch (Exception e) {
			throw new DMLRuntimeException("Error while executing multi-threaded tasks", e);
		}
		
		return lnnz;
	}
	
	private static void checkOrThrowException(String msg, long lhs, long rhs) {
		if(lhs != rhs)
			throw new DMLRuntimeException(msg + ":" + lhs + " != " + rhs);
	}
	private static void checkOrThrowException(String msg, long lhs, long rhs1, long rhs2, long rhs3) {
		if(lhs != (rhs1*rhs2*rhs3))
			throw new DMLRuntimeException(msg + ":" + lhs + " != (" + rhs1 + " * " + rhs2 + " * " + rhs3);
	}
	
	static void checkInputsConv2dBackwardData(MatrixBlock filter, MatrixBlock dout, MatrixBlock outputBlock, DnnParameters params) {
		params.input1 = filter;
		params.input2 = dout;
		params.output = outputBlock;
		checkOrThrowException("Incorrect input to conv2d_backward_data: Number of rows of input filter != "
				+ "number of filters in filter_shape", filter.getNumRows(), params.K);
		checkOrThrowException("Incorrect input to conv2d_backward_data: Number of columns of input filter != "
				+ "channels*filter_height*filter_height in filter_shape", filter.getNumColumns(), params.C, params.R, params.S);
		checkOrThrowException("Incorrect input to conv2d_backward_data: Number of rows of input errors != "
				+ "batch size in input_shape", dout.getNumRows(), params.N);
		checkOrThrowException("Incorrect input to conv2d_backward_data: Number of columns of input errors != "
				+ "expected input error channels*height*width", dout.getNumColumns(), params.K, params.P, params.Q);
		if(params.stride_h <= 0 || params.stride_w <= 0) 
			throw new DMLRuntimeException("Only positive strides supported:" + params.stride_h + ", " + params.stride_w);
		
		if(ConfigurationManager.isFinegrainedStatistics()) {
			if(filter.isInSparseFormat() || dout.isInSparseFormat()) {
				conv2dBwdDataSparseCount.addAndGet(1);
			}
			else {
				conv2dBwdDataDenseCount.addAndGet(1);
			}
		}
	}
	
	static void checkInputsConv2dBackwardFilter(MatrixBlock input, MatrixBlock dout, MatrixBlock outputBlock, DnnParameters params) {
		params.input1 = input;
		params.input2 = dout;
		params.output = outputBlock;
		checkOrThrowException("Incorrect input to conv2d_backward_filter: Number of rows of input data != "
				+ "batch size in input_shape", input.getNumRows(), params.N);
		checkOrThrowException("Incorrect input to conv2d_backward_filter: Number of columns of input data != "
				+ "channels*input_height*input_height in input_shape", input.getNumColumns(), params.C, params.H, params.W);
		checkOrThrowException("Incorrect input to conv2d_backward_filter: Number of rows of input errors != "
				+ "batch size in input_shape", dout.getNumRows(), params.N);
		checkOrThrowException("Incorrect input to conv2d_backward_filter: Number of columns of input errors != "
				+ "expected input error channels*height*width", dout.getNumColumns(), params.K, params.P, params.Q);
		if(params.stride_h <= 0 || params.stride_w <= 0) 
			throw new DMLRuntimeException("Only positive strides supported:" + params.stride_h + ", " + params.stride_w);
		
		if(ConfigurationManager.isFinegrainedStatistics()) {
			if(input.isInSparseFormat() || dout.isInSparseFormat()) {
				conv2dBwdFilterSparseCount.addAndGet(1);
			}
			else {
				conv2dBwdFilterDenseCount.addAndGet(1);
			}
		}
	}
	
	static void checkInputsConv2d(MatrixBlock input, MatrixBlock filter, MatrixBlock outputBlock, DnnParameters params) {
		params.input1 = input;
		params.input2 = filter;
		params.output = outputBlock;
		
		checkOrThrowException("Incorrect input to conv2d: Number of rows of input filter != "
				+ "number of filters in filter_shape", filter.getNumRows(), params.K);
		checkOrThrowException("Incorrect input to conv2d: Number of columns of input filter != "
				+ "channels*filter_height*filter_height in filter_shape", filter.getNumColumns(), params.C, params.R, params.S);
		checkOrThrowException("Incorrect input to conv2d: Number of rows of input data != "
				+ "batch size in input_shape", input.getNumRows(), params.N);
		checkOrThrowException("Incorrect input to conv2d: Number of columns of input data != "
				+ "channels*input_height*input_height in input_shape", input.getNumColumns(), params.C, params.H, params.W);
		if(params.stride_h <= 0 || params.stride_w <= 0) 
			throw new DMLRuntimeException("Only positive strides supported:" + params.stride_h + ", " + params.stride_w);
		
		if(ConfigurationManager.isFinegrainedStatistics()) {
			if(input.isInSparseFormat() || filter.isInSparseFormat()) {
				conv2dSparseCount.addAndGet(1);
			}
			else {
				conv2dDenseCount.addAndGet(1);
			}
		}
	}
	
	/**
	 * This method computes start and end indexes required for max_pool and max_pool_backward operations.
	 * This speeds up the performance of max_pool and  max_pool_backward
	 * 
	 * @param params parameters required for max_pool and max_pool_backward operations
	 */
	private static void fillIndexesArray(DnnParameters params) {
		params.start_indexes_h = new int[params.P];
		params.end_indexes_h = new int[params.P];
		params.start_indexes_w = new int[params.Q];
		params.end_indexes_w = new int[params.Q];
		for( int p=0, ix=-params.pad_h; p < params.P; p++, ix+=params.stride_h ) {
			// Note: We do not treat pad as zero
			params.start_indexes_h[p] = Math.max(ix, 0);
			params.end_indexes_h[p] = Math.min(ix+params.R, params.H);
		}
		for( int q=0, ix=-params.pad_w; q < params.Q; q++, ix+=params.stride_w) {
			// Note: We do not treat pad as zero
			params.start_indexes_w[q] = Math.max(ix, 0);
			params.end_indexes_w[q] = Math.min(ix+params.S, params.W);
		}
	}
}
