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

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.stream.IntStream;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.DenseBlockFactory;
import org.apache.sysds.utils.NativeHelper;
import org.apache.sysds.utils.stats.NativeStatistics;

public class LibMatrixNativeFFM
{
	private static final Log LOG = LogFactory.getLog(LibMatrixNativeFFM.class.getName());

	public static Arena arena;
	public static MemorySegment m1Segment, m2Segment, retSegment;
	public static Linker linker = Linker.nativeLinker();
	public static SymbolLookup symbolLookup = SymbolLookup.loaderLookup();
	public static FunctionDescriptor dmatmultDescriptor =
			FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, // m1Ptr
										ValueLayout.ADDRESS, // m2Ptr
										ValueLayout.ADDRESS, // retPtr
										ValueLayout.JAVA_INT, // m1 rlen
										ValueLayout.JAVA_INT, // m1 clen
										ValueLayout.JAVA_INT, // m2 clen
										ValueLayout.JAVA_INT); // k

	// active == true -> open arena. active == false -> close arena
	public static void controlArena(boolean active) {
		if(active)
			arena = Arena.ofConfined();
		else
			arena.close();
	}

	// ThreadLocal reuse of direct buffers for inputs/outputs (extended on demand).
	//   note: since we anyway have to convert from double to float, we use
	//   preallocated direct buffers (with thread-local reuse and resizing on demand)
	//   to ensure there are no additional copies created by the transfer over jni
	private static ThreadLocal<FloatBuffer> inBuff = new ThreadLocal<>();
	private static ThreadLocal<FloatBuffer> biasBuff = new ThreadLocal<>();
	private static ThreadLocal<FloatBuffer> filterBuff = new ThreadLocal<>();
	private static ThreadLocal<FloatBuffer> outBuff = new ThreadLocal<>();
	
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
	 * @return the ret matrixBlock if allocated otherwise a new matrixBlock.
	 */
	public static MatrixBlock matrixMult(MatrixBlock m1, MatrixBlock m2, MatrixBlock ret, int k) {
		if(NativeHelper.isNativeLibraryLoaded()){
			// Sanity check:
			k = k <= 0 ? NativeHelper.getMaxNumThreads() : k;
			
			// check inputs / outputs
			if (m1.isEmptyBlock(false) || m2.isEmptyBlock(false))
				return LibMatrixMult.emptyMatrixMult(m1,m2, ret);
			
			boolean isValidForNative = !isMatMultMemoryBound(m1.rlen, m1.clen, m2.clen) 
				&& !m1.isInSparseFormat() && !m2.isInSparseFormat()
				&& (m1.getDenseBlock().isContiguous() || !isSinglePrecision())
				&& m2.getDenseBlock().isContiguous() //contiguous but not allocated
				&& 8L * ret.getLength() < Integer.MAX_VALUE;
	
			if( isValidForNative ) 
			{
				// allocate output
				if(ret == null)
					ret = new MatrixBlock(m1.rlen, m2.clen, false);
				else 
					ret.reset(m1.rlen, m2.clen, false);
				ret.allocateBlock();
				
				long start = DMLScript.STATISTICS ? System.nanoTime() : 0;
				long nnz = 0;
				if( isSinglePrecision() ) {
					FloatBuffer fin1 = toFloatBuffer(m1.getDenseBlockValues(), inBuff, true);
					FloatBuffer fin2 = toFloatBuffer(m2.getDenseBlockValues(), filterBuff, true);
					FloatBuffer fout = toFloatBuffer(ret.getDenseBlockValues(), outBuff, false);
					nnz = NativeHelper.smmdd(fin1, fin2, fout, 
						m1.getNumRows(), m1.getNumColumns(), m2.getNumColumns(), k);
					fromFloatBuffer(outBuff.get(), ret.getDenseBlockValues());
				}
				else {
					DenseBlock a = m1.getDenseBlock();
					if( a.isContiguous() ) {
						int len = m1.rlen * m2.clen;

						// Look for dmatmult function
						MethodHandle methodHandle = linker
							.downcallHandle(symbolLookup.find("_Z8dmatmultPdS_S_iiii").get(), dmatmultDescriptor);

						// Invoke native function
						try {
							methodHandle.invokeExact(m1Segment, m2Segment, retSegment, m1.rlen, m1.clen, m2.clen, k);
						}
						catch(Throwable e) {
							throw new RuntimeException(e);
						}
						// Write values back into ret MatrixBlock and count nnz
						double[] tmp = ret.getDenseBlockValues();
						for(int i = 0; i < len; i++)
							nnz += (tmp[i] = retSegment.getAtIndex(ValueLayout.JAVA_DOUBLE, i)) != 0 ? 1 : 0;
					}
					else {
						//sequential processing of individual blocks to
						//avoid segementation faults with concurrent multi-threaded BLAS calls
						for(int bix = 0; bix < a.numBlocks(); bix++) {
							double[] tmp = new double[a.blockSize(bix)*m2.clen];
							nnz += NativeHelper.dmmdd(a.valuesAt(bix), m2.getDenseBlockValues(),
									tmp, a.blockSize(bix), m1.clen, m2.clen, k);
							int rl = bix * a.blockSize();
							ret.getDenseBlock().set(rl, rl+a.blockSize(bix), 0, m2.clen,
									DenseBlockFactory.createDenseBlock(tmp, new int[]{a.blockSize(bix),m2.clen}));
						}
					}
				}
				
				if(nnz > -1) {
					if(DMLScript.STATISTICS) {
						NativeStatistics.incrementLibMatrixMultTime(System.nanoTime() - start);
						NativeStatistics.incrementNumLibMatrixMultCalls();
					}
					ret.setNonZeros(nnz);
					ret.examSparsity();
					return ret;
				}
				//else record failure and fallback to java
				NativeStatistics.incrementFailuresCounter();
				LOG.warn("matrixMult: Native mat mult failed. Falling back to java version ("
					+ "loaded=" + NativeHelper.isNativeLibraryLoaded()
					+ ", sparse=" + (m1.isInSparseFormat() | m2.isInSparseFormat()) + ")");
			}
		}
		else
			LOG.warn("Was valid for native MM but native lib was not loaded");
		
		return LibMatrixMult.matrixMult(m1, m2, ret, k);
	}
	
	public static void tsmm(MatrixBlock m1, MatrixBlock ret, boolean leftTrans, int k) {
		if( m1.isEmptyBlock(false) )
			return;
		
		if( NativeHelper.isNativeLibraryLoaded() && (ret.clen > 1 || ret.getLength()==1)
			&& !LibMatrixMult.isOuterProductTSMM(m1.rlen, m1.clen, leftTrans)
			&& !m1.sparse && (m1.getDenseBlock().isContiguous() | leftTrans) )
		{
			ret.sparse = false;
			ret.allocateDenseBlock();
			long start = DMLScript.STATISTICS ? System.nanoTime() : 0;
			
			DenseBlock a = m1.getDenseBlock();
			double[] cvals = ret.getDenseBlockValues();
			long nnz = 0;
			if( a.isContiguous() ) {
				nnz = NativeHelper.tsmm(a.valuesAt(0),
					cvals, m1.rlen, m1.clen, leftTrans, k);
			}
			else { //large blocks (but only leftTrans)
				//sequential processing of individual blocks to 
				//avoid segementation faults with concurrent multi-threaded BLAS calls
				IntStream.range(0, a.numBlocks()).forEach(bix -> {
					double[] tmp = new double[m1.clen*m1.clen];
					NativeHelper.tsmm(a.valuesAt(bix),
						tmp, a.blockSize(bix), m1.clen, leftTrans, k);
					LibMatrixMult.vectAdd(tmp, cvals, 0, 0, m1.clen*m1.clen);
				});
				nnz = ret.recomputeNonZeros();
			}
			//TODO flip upper triangular matrix down for consistent
			//representations with the default java implementation?
			
			if(nnz > -1) {
				if(DMLScript.STATISTICS) {
					NativeStatistics.incrementLibMatrixMultTime(System.nanoTime() - start);
					NativeStatistics.incrementNumLibMatrixMultCalls();
				}
				ret.setNonZeros(nnz);
				ret.examSparsity();
				return;
			}
			//fallback to default java implementation
			LOG.warn("Native TSMM failed. Falling back to java version.");
			NativeStatistics.incrementFailuresCounter();
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
			long nnz;
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
					NativeStatistics.incrementConv2dTime(System.nanoTime() - start);
					NativeStatistics.incrementNumConv2dCalls();
				}
				outputBlock.setNonZeros(nnz);
				return;
			}
			else {
				// Fall back to Java in case of failures, reset output to ensure correctness
				LOG.warn("Native conv2d call returned with error - falling back to java operator.");
				if( !(isSinglePrecision() && params.bias!=null) )
					outputBlock.reset();
				NativeStatistics.incrementFailuresCounter();
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
			long nnz = NativeHelper.conv2dBackwardFilterDense(input.getDenseBlockValues(), dout.getDenseBlockValues(),
					outputBlock.getDenseBlockValues(), params.N, params.C, params.H, params.W, 
					params.K, params.R, params.S, params.stride_h, params.stride_w, params.pad_h, params.pad_w, 
					params.P, params.Q, params.numThreads);
			if(nnz != -1) {
				if(DMLScript.STATISTICS) {
					NativeStatistics.incrementConv2dBwdFilterTime(System.nanoTime() - start);
					NativeStatistics.incrementNumConv2dBwdFilterCalls();
				}
				// post-processing: maintain nnz
				outputBlock.setNonZeros(nnz);
				return;
			}
			else {
				// Fall back to Java when failures
				NativeStatistics.incrementFailuresCounter();
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
			long nnz = NativeHelper.conv2dBackwardDataDense(filter.getDenseBlockValues(), dout.getDenseBlockValues(),
					outputBlock.getDenseBlockValues(), params.N, params.C, params.H, params.W, 
					params.K, params.R, params.S, params.stride_h, params.stride_w, params.pad_h, params.pad_w, 
					params.P, params.Q, params.numThreads);
			if(nnz != -1) {
				if(DMLScript.STATISTICS) {
					NativeStatistics.incrementConv2dBwdDataTime(System.nanoTime() - start);
					NativeStatistics.incrementNumConv2dBwdDataCalls();
				}
				// post-processing: maintain nnz
				outputBlock.setNonZeros(nnz);
				return;
			}
			else {
				// Fall back to Java when failures
				NativeStatistics.incrementFailuresCounter();
			}
		}
		// Fall back to Java when failures or sparse
		LibMatrixDNN.conv2dBackwardData(filter, dout, outputBlock, params);
	}
	
	public static boolean isSinglePrecision() {
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
