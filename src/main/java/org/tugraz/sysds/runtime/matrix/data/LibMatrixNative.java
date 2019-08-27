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
package org.tugraz.sysds.runtime.matrix.data;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.stream.IntStream;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.conf.ConfigurationManager;
import org.tugraz.sysds.conf.DMLConfig;
import org.tugraz.sysds.hops.OptimizerUtils;
import org.tugraz.sysds.utils.NativeHelper;
import org.tugraz.sysds.utils.Statistics;

public class LibMatrixNative
{
	private static final Log LOG = LogFactory.getLog(LibMatrixNative.class.getName());
	
	// ThreadLocal reuse of direct buffers for inputs/outputs (extended on demand).
	//   note: since we anyway have to convert from double to float, we use
	//   preallocated direct buffers (with thread-local reuse and resizing on demand)
	//   to ensure there are no additional copies created by the transfer over jni
	private static ThreadLocal<FloatBuffer> inBuff = new ThreadLocal<FloatBuffer>();
	private static ThreadLocal<FloatBuffer> biasBuff = new ThreadLocal<FloatBuffer>();
	private static ThreadLocal<FloatBuffer> filterBuff = new ThreadLocal<FloatBuffer>();
	private static ThreadLocal<FloatBuffer> outBuff = new ThreadLocal<FloatBuffer>();
	
	// We could encapsulate heuristics in this function
	// For now, we only consider matrix-vector operation to be memory bound
	public static boolean isMatMultMemoryBound(int m1Rlen, int m1Clen, int m2Clen) {
		return (m1Rlen == 1 || m1Clen == 1 || m2Clen == 1)
			&& (8L*m1Rlen*m1Clen > 16 * LibMatrixMult.L3_CACHESIZE 
				|| 8L*m1Clen*m2Clen > 16 * LibMatrixMult.L3_CACHESIZE);
	}

	/**
	 * Performs matrix multiplication using native library if BLAS is available or else falls back to
	 * Java BLAS.
	 * 
	 * @param m1 lhs matrix block
	 * @param m2 rhs matrix block
	 * @param ret output matrix block
	 * @param k number of threads
	 */
	public static void matrixMult(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int k) {
		matrixMult(m1, m2, ret, k, true);
	}
	
	public static void matrixMult(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int k, boolean examSparsity) {
		// Sanity check:
		k = k <= 0 ? NativeHelper.getMaxNumThreads() : k;
		
		// check inputs / outputs
		if (m1.isEmptyBlock(false) || m2.isEmptyBlock(false)){
			ret.setNonZeros(0);
			if(examSparsity)
				ret.examSparsity(); // turn empty dense into sparse
			return;
		}
		
		if( NativeHelper.isNativeLibraryLoaded()
			&& !isMatMultMemoryBound(m1.rlen, m1.clen, m2.clen) 
			&& !m1.isInSparseFormat() && !m2.isInSparseFormat()
			&& m1.getDenseBlock().isContiguous() && m2.getDenseBlock().isContiguous()
			&& 8L * ret.getLength() < Integer.MAX_VALUE ) //contiguous but not allocated
		{
			ret.sparse = false;
			ret.allocateDenseBlock();
			long start = DMLScript.STATISTICS ? System.nanoTime() : 0;
			boolean rccode = false;
			if( isSinglePrecision() ) {
				FloatBuffer fin1 = toFloatBuffer(m1.getDenseBlockValues(), inBuff, true);
				FloatBuffer fin2 = toFloatBuffer(m2.getDenseBlockValues(), filterBuff, true);
				FloatBuffer fout = toFloatBuffer(ret.getDenseBlockValues(), outBuff, false);
				rccode = NativeHelper.smmdd(fin1, fin2, fout, 
					m1.getNumRows(), m1.getNumColumns(), m2.getNumColumns(), k);
				fromFloatBuffer(outBuff.get(), ret.getDenseBlockValues());
			}
			else {
				rccode = NativeHelper.dmmdd(m1.getDenseBlockValues(), m2.getDenseBlockValues(),
					ret.getDenseBlockValues(), m1.getNumRows(), m1.getNumColumns(), m2.getNumColumns(), k);
			}
			if (rccode) {
				if(DMLScript.STATISTICS) {
					Statistics.nativeLibMatrixMultTime += System.nanoTime() - start;
					Statistics.numNativeLibMatrixMultCalls.increment();
				}
				ret.recomputeNonZeros();
				if(examSparsity)
					ret.examSparsity();
				return;
			}
			//else record failure and fallback to java
			Statistics.incrementNativeFailuresCounter();
		}
		if (k == 1)
			LibMatrixMult.matrixMult(m1, m2, ret, !examSparsity);
		else
			LibMatrixMult.matrixMult(m1, m2, ret, k);
	}
	
	public static void tsmm(MatrixBlock m1, MatrixBlock ret, boolean leftTrans, int k) {
		if( m1.isEmptyBlock(false) )
			return;
		if( NativeHelper.isNativeLibraryLoaded() && (ret.clen > 1 || ret.getLength()==1)
			&& (!m1.sparse && m1.getDenseBlock().isContiguous() ) ) {
			ret.sparse = false;
			ret.allocateDenseBlock();
			if( NativeHelper.tsmm(m1.getDenseBlockValues(), 
				ret.getDenseBlockValues(), m1.rlen, m1.clen, leftTrans, k) ) 
			{
				LOG.info("Using native TSMM()");
				long nnz = (ret.clen==1) ? ret.recomputeNonZeros() :
					LibMatrixMult.copyUpperToLowerTriangle(ret);
				ret.setNonZeros(nnz);
				ret.examSparsity();
				return;
			}
			//fallback to default java implementation
			LOG.info("Falling back to java TSMM()");
			Statistics.incrementNativeFailuresCounter();
		}
		if( k > 1 )
			LibMatrixMult.matrixMultTransposeSelf(m1, ret, leftTrans, k);
		else
			LibMatrixMult.matrixMultTransposeSelf(m1, ret, leftTrans);
	}
	
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
		params.numThreads = params.numThreads <= 0 ? NativeHelper.getMaxNumThreads() : params.numThreads;
		if(NativeHelper.isNativeLibraryLoaded() && !input.isInSparseFormat() && !filter.isInSparseFormat()) {
			setNumThreads(params);
			long start = DMLScript.STATISTICS ? System.nanoTime() : 0;
			int nnz = 0;
			if(params.bias == null) {
				nnz = NativeHelper.conv2dDense(input.getDenseBlockValues(), filter.getDenseBlockValues(),
						outputBlock.getDenseBlockValues(), params.N, params.C, params.H, params.W, 
						params.K, params.R, params.S, params.stride_h, params.stride_w, params.pad_h, params.pad_w, 
						params.P, params.Q, params.numThreads);
			}
			else {
				if(params.bias.isInSparseFormat())
					params.bias.sparseToDense(); // Bias matrix is usually extremely small
				//NOTE: We temporarily disable MKL FP32 conv2d_bias_add due to incorrect results on
				//newer processors with AVX2 and AVX-512 instruction set (library bug or alignment issue)
				//Experiments have shown that falling back to the MKL FP64 primitives is generally faster
				//than falling back to the custom openmp FP32 implementation.
				if( isSinglePrecision() && !NativeHelper.getCurrentBLAS().equalsIgnoreCase("mkl") ) {
					FloatBuffer finput = toFloatBuffer(input.getDenseBlockValues(), inBuff, true);
					FloatBuffer fbias = toFloatBuffer(params.bias.getDenseBlockValues(), biasBuff, true);
					FloatBuffer ffilter = toFloatBuffer(filter.getDenseBlockValues(), filterBuff, true);
					FloatBuffer foutput = toFloatBuffer(outputBlock.getDenseBlockValues(), outBuff, false);
					nnz = NativeHelper.sconv2dBiasAddDense(finput, fbias, ffilter, foutput,
						params.N, params.C, params.H, params.W, params.K, params.R, params.S,
						params.stride_h, params.stride_w, params.pad_h, params.pad_w, 
						params.P, params.Q, params.numThreads);
					if( nnz != -1 )
						fromFloatBuffer(outBuff.get(), outputBlock.getDenseBlockValues());
				}
				else { //Double
					nnz = NativeHelper.dconv2dBiasAddDense(input.getDenseBlockValues(), params.bias.getDenseBlockValues(),
						filter.getDenseBlockValues(), outputBlock.getDenseBlockValues(),
						params.N, params.C, params.H, params.W, params.K, params.R, params.S,
						params.stride_h, params.stride_w, params.pad_h, params.pad_w, 
						params.P, params.Q, params.numThreads);	
				}
			}
			//post processing and error handling
			if(nnz != -1) {
				if(DMLScript.STATISTICS) {
					Statistics.nativeConv2dTime += System.nanoTime() - start;
					Statistics.numNativeConv2dCalls.increment();
				}
				outputBlock.setNonZeros(nnz);
				return;
			}
			else {
				// Fall back to Java in case of failures, reset output to ensure correctness
				LOG.warn("Native conv2d call returned with error - falling back to java operator.");
				if( !(isSinglePrecision() && params.bias!=null) )
					outputBlock.reset();
				Statistics.incrementNativeFailuresCounter();
			}
		}
		
		// Fall back to Java when failures or sparse
		LibMatrixDNN.conv2d(input, filter, outputBlock, params);
	}
	
	private static void setNumThreads(DnnParameters params) {
		params.numThreads = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		if (!(params.isOutputThreadSafe() && params.numThreads > 1))
			params.numThreads = 1;
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
		LibMatrixDNN.checkInputsConv2dBackwardFilter(input, dout, outputBlock, params);
		params.numThreads = params.numThreads <= 0 ? NativeHelper.getMaxNumThreads() : params.numThreads;
		if(NativeHelper.isNativeLibraryLoaded() && !dout.isInSparseFormat() && !input.isInSparseFormat()) {
			setNumThreads(params);
			long start = DMLScript.STATISTICS ? System.nanoTime() : 0;
			int nnz = NativeHelper.conv2dBackwardFilterDense(input.getDenseBlockValues(), dout.getDenseBlockValues(),
					outputBlock.getDenseBlockValues(), params.N, params.C, params.H, params.W, 
					params.K, params.R, params.S, params.stride_h, params.stride_w, params.pad_h, params.pad_w, 
					params.P, params.Q, params.numThreads);
			if(nnz != -1) {
				if(DMLScript.STATISTICS) {
					Statistics.nativeConv2dBwdFilterTime += System.nanoTime() - start;
					Statistics.numNativeConv2dBwdFilterCalls.increment();
				}
				// post-processing: maintain nnz
				outputBlock.setNonZeros(nnz);
				return;
			}
			else {
				// Fall back to Java when failures
				Statistics.incrementNativeFailuresCounter();
			}
		}
		// Fall back to Java when failures or sparse
		LibMatrixDNN.conv2dBackwardFilter(input, dout, outputBlock, params);
	}
	
	/**
	 * This method computes the backpropagation errors for previous layer of convolution operation
	 * 
	 * @param filter filter used in conv2d 
	 * @param dout errors from next layer
	 * @param outputBlock  output errors
	 * @param params convolution parameters
	 */
	public static void conv2dBackwardData(MatrixBlock filter, MatrixBlock dout, MatrixBlock outputBlock, DnnParameters params) {
		LibMatrixDNN.checkInputsConv2dBackwardData(filter, dout, outputBlock, params);
		params.numThreads = params.numThreads <= 0 ? NativeHelper.getMaxNumThreads() : params.numThreads;
		if(NativeHelper.isNativeLibraryLoaded() && !dout.isInSparseFormat() && !filter.isInSparseFormat()) {
			setNumThreads(params);
			long start = DMLScript.STATISTICS ? System.nanoTime() : 0;
			int nnz = NativeHelper.conv2dBackwardDataDense(filter.getDenseBlockValues(), dout.getDenseBlockValues(),
					outputBlock.getDenseBlockValues(), params.N, params.C, params.H, params.W, 
					params.K, params.R, params.S, params.stride_h, params.stride_w, params.pad_h, params.pad_w, 
					params.P, params.Q, params.numThreads);
			if(nnz != -1) {
				if(DMLScript.STATISTICS) {
					Statistics.nativeConv2dBwdDataTime += System.nanoTime() - start;
					Statistics.numNativeConv2dBwdDataCalls.increment();
				}
				// post-processing: maintain nnz
				outputBlock.setNonZeros(nnz);
				return;
			}
			else {
				// Fall back to Java when failures
				Statistics.incrementNativeFailuresCounter();
			}
		}
		// Fall back to Java when failures or sparse
		LibMatrixDNN.conv2dBackwardData(filter, dout, outputBlock, params);
	}
	
	private static boolean isSinglePrecision() {
		return ConfigurationManager.getDMLConfig()
			.getTextValue(DMLConfig.FLOATING_POINT_PRECISION).equals("single");
	}
	
	private static FloatBuffer toFloatBuffer(double[] input, ThreadLocal<FloatBuffer> buff, boolean copy) {
		//maintain thread-local buffer (resized on demand)
		FloatBuffer ret = buff.get();
		if( ret == null || ret.capacity() < input.length ) {
			ret = ByteBuffer.allocateDirect(4*input.length)
				.order(ByteOrder.nativeOrder()).asFloatBuffer();
			buff.set(ret);
		}
		//copy to direct byte buffer
		final FloatBuffer ret2 = ret;
		if( copy ) {
			IntStream.range(0, input.length).parallel()
				.forEach(i -> ret2.put(i, (float)input[i]));
		}
		return ret2;
	}
	
	public static void fromFloatBuffer(FloatBuffer buff, double[] output) {
		Arrays.parallelSetAll(output, i -> (double)buff.get(i) );
	}
}
