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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.Callable;

import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.LibMatrixDNNRotate180.Rotate180Worker;
import org.apache.sysds.utils.NativeHelper;
import org.apache.sysds.utils.Statistics;

/**
 * This class contains the set of operators used for performing conv2d
 */
public class LibMatrixDNNConv2d 
{
	/**
	 * Factory method that returns list of callable tasks for performing conv2d
	 * 
	 * @param params convolution parameters
	 * @return list of callable tasks for performing conv2d
	 */
	public static ArrayList<Callable<Long>> getConv2dWorkers(DnnParameters params) {
		ArrayList<Callable<Long>> ret = new ArrayList<>();
		
		// Try to create twice as many tasks as threads for improved load balance
		// (due to constant-sized intermediates, GC works well, so the overhead per task is small)
		int k = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		int taskSize = (int)(Math.ceil((double)params.N / k / 2));
		
		MatrixBlock in1 = params.input1;
		boolean isEmptyDenseInput = !in1.isInSparseFormat() && in1.denseBlock == null;
		boolean isTransPref = in1.sparse && !params.input2.sparse && !params.output.sparse &&
			MatrixBlock.evalSparseFormatInMemory(in1.clen, in1.rlen, in1.nonZeros);
		boolean applyNative = isEligibleForConv2dSparse(params)
			&& !(!isEmptyDenseInput && isTransPref);
		if( applyNative )
			Statistics.numNativeSparseConv2dCalls.increment();
		
		//transpose filter once for efficient sparse-dense multiplies in LoopedIm2ColConv2dTransAllChan
		//in order to share the temporary object and its creation costs across threads
		if( !applyNative && !isEmptyDenseInput && isTransPref ) {
			params.input2 = LibMatrixReorg.transpose(params.input2, 
				new MatrixBlock(params.input2.clen, params.input2.rlen, false), k);
		}
		
		for(int i = 0; i*taskSize < params.N; i++) {
			//note: we prefer the java backend for sparse inputs because the native 
			//implementation simply converts the sparse input into dense rows
			if( applyNative ) 
				ret.add(new SparseNativeConv2d(i*taskSize, Math.min((i+1)*taskSize, params.N), params));
			else if(!isEmptyDenseInput && isTransPref)
				ret.add(new LoopedIm2ColConv2dTransAllChan(i*taskSize, Math.min((i+1)*taskSize, params.N), params));
			else if(!isEmptyDenseInput)
				ret.add(new LoopedIm2ColConv2dAllChan(i*taskSize, Math.min((i+1)*taskSize, params.N), params));
			else
				throw new DMLRuntimeException("Unsupported operator");
		}
		return ret;
	}
	
	/**
	 * Factory method that returns list of callable tasks for performing conv2d backward filter
	 * 
	 * @param params convolution parameters
	 * @return list of callable tasks for performing conv2d backward filter
	 */
	public static ArrayList<Callable<Long>> getConv2dBackwardFilterWorkers(DnnParameters params) {
		ArrayList<Callable<Long>> ret = new ArrayList<>();
		// Try to create as many tasks as threads. 
		// Creating more tasks will help in tail, but would have additional overhead of maintaining the intermediate
		// data structures such as im2col blocks.
		int k = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		int taskSize = (int)(Math.ceil((double)params.N / k));
		
		boolean isEmptyDenseInput = (!params.input1.isInSparseFormat() && params.input1.denseBlock == null) || 
			(!params.input2.isInSparseFormat() && params.input2.denseBlock == null);
		boolean applyNative = isEligibleForConv2dBackwardFilterSparseDense(params)
			&& !params.input2.isInSparseFormat();
		if( applyNative )
			Statistics.numNativeSparseConv2dBwdFilterCalls.increment();
		
		for(int i = 0; i*taskSize < params.N; i++) {
			//note: we prefer the java backend for sparse filters because the native 
			//implementation simply rotates the sparse filters into dense rows
			if( applyNative ) 
				ret.add(new SparseNativeConv2dBackwardFilterDense(i*taskSize, Math.min((i+1)*taskSize, params.N), params));
			else if( params.input2.sparse && params.input1.getSparsity() > params.input2.getSparsity() )
				ret.add(new Conv2dBackwardFilterTrans(i*taskSize, Math.min((i+1)*taskSize, params.N), params));
			else if(!isEmptyDenseInput)
				ret.add(new Conv2dBackwardFilter(i*taskSize, Math.min((i+1)*taskSize, params.N), params));
			else
				throw new DMLRuntimeException("Unsupported operator");
		}
		return ret;
	}
	
	/**
	 * Factory method that returns list of callable tasks for performing conv2d backward data
	 * 
	 * @param params convolution parameters
	 * @return list of callable tasks for performing conv2d backward data
	 */
	public static ArrayList<Callable<Long>> getConv2dBackwardDataWorkers(DnnParameters params) {
		ArrayList<Callable<Long>> ret = new ArrayList<>();
		
		// Try to create as many tasks as threads. 
		// Creating more tasks will help in tail, but would have additional overhead of maintaining the intermediate
		// data structures such as im2col blocks.
		int k = OptimizerUtils.getConstrainedNumThreads(params.numThreads);
		int taskSize = (int)(Math.ceil((double)params.N / k));
		
		boolean isEmptyDenseInput = (!params.input1.isInSparseFormat() && params.input1.denseBlock == null) || 
			(!params.input2.isInSparseFormat() && params.input2.denseBlock == null);
		boolean applyNative = isEligibleForConv2dBackwardDataDense(params)
			&& !params.input2.isInSparseFormat();
		if( applyNative )
			Statistics.numNativeSparseConv2dBwdDataCalls.increment();
		
		for(int i = 0; i*taskSize < params.N; i++) {
			//note: we prefer the java backend for sparse filters because the native 
			//implementation simply converts the sparse filters into dense rows
			if( applyNative ) 
				ret.add(new SparseNativeConv2dBackwardDataDense(i*taskSize, Math.min((i+1)*taskSize, params.N), params));
			else if(!isEmptyDenseInput)
				ret.add(new Conv2dBackwardData(i*taskSize, Math.min((i+1)*taskSize, params.N), params));
			else
				throw new DMLRuntimeException("Unsupported operator");
		}
		
		return ret;
	}
	
	/**
	 * Performs convolution via: partialCopy1(filter %*% im2col(input)) = output
	 */
	private static class LoopedIm2ColConv2dAllChan implements Callable<Long> 
	{
		protected final int _rl, _ru; 
		protected final DnnParameters _params;
		
		public LoopedIm2ColConv2dAllChan(int rl, int ru, DnnParameters params) {
			_rl = rl; _ru = ru;
			_params = params;
		}

		@Override
		public Long call() throws Exception {
			final int PQ = _params.P*_params.Q, K = _params.K, CRS = _params.C*_params.R*_params.S;
			MatrixBlock outIm2col = new MatrixBlock(CRS, PQ, _params.input1.sparse).allocateBlock();
			LibMatrixDNNIm2Col.preallocateSparseOutput(_params.input1, outIm2col);
			MatrixBlock outMM = new MatrixBlock(K, PQ, _params.output.sparse);
			for(int n = _rl; n < _ru; n++)  {
				LibMatrixDNNIm2Col.im2col(_params.input1, outIm2col, n, _params, false);
				
				// filter %*% _im2ColOutBlock => matMultOutBlock
				outMM.reset(outMM.rlen, outMM.clen, _params.output.sparse);
				LibMatrixDNNHelper.singleThreadedMatMult(_params.input2, outIm2col, outMM, false, true, _params);
				
				// Copy the outMM of shape [K x PQ] to a row in params.output 
				partialCopy1(outMM, _params.output, n, K, PQ);
				
				// Add bias to current row if necessary, always dense
				if(_params.bias != null)
					addBias(n, _params.output.getDenseBlockValues(),
						_params.bias.getDenseBlockValues(), K, PQ);
			}
			
			//multi-threaded nnz maintenance of current working set
			return _params.output.recomputeNonZeros(_rl, _ru-1);
		}
		
		// Copy the matrix src of shape [K X PQ] to the r-th row in params.output 
		private static void partialCopy1(MatrixBlock src, MatrixBlock dest, int r, int K, int PQ) {
			// Copying is required as LibMatrixMult.matrixMult (and/or Java) is not pointer aware.
			if( src.isEmptyBlock() )
				return;
			if( src.sparse ) { //* <- SPARSE
				SparseBlock srcBlock = src.sparseBlock;
				SparseBlock sdestBlock = dest.sparseBlock;
				double[] ddestBlock = dest.getDenseBlockValues();
				
				for(int k = 0; k < src.getNumRows(); k++) {
					if( srcBlock.isEmpty(k) ) continue;
					int apos = srcBlock.pos(k);
					int alen = srcBlock.size(k);
					int[] aix = srcBlock.indexes(k);
					double[] avals = srcBlock.values(k);
					if( dest.sparse ) {
						sdestBlock.setIndexRange(r,
							0, K*PQ, avals, aix, apos, alen);
					}
					else {
						int desPosK = r + k*PQ;
						for(int j = apos; j < apos+alen; j++)
							ddestBlock[desPosK+aix[j]] = avals[j];
					}
				}
			}
			else { //* <- DENSE
				if( dest.sparse ) {
					dest.getSparseBlock().setIndexRange(r, 0, K*PQ,
						src.getDenseBlockValues(), 0, K*PQ);
				}
				else {
					System.arraycopy(src.getDenseBlockValues(), 0,
						dest.getDenseBlockValues(), r*K*PQ, K*PQ);
				}
			}
		}
	}
	
	/**
	 * This implementation is similar to LoopedIm2ColConv2dAllChan, except for using a 
	 * sparse-dense matrix multiplication with t(t(Xi) %*% t(F)) instead of a 
	 * dense-sparse matrix multiplication with Xi %*% F.
	 * 
	 * NOTE: this implementation assumes that the filter is passed in transposed form
	 * in order to share this temporary matrix (and its creation cost) across threads.
	 */
	private static class LoopedIm2ColConv2dTransAllChan extends LoopedIm2ColConv2dAllChan
	{
		public LoopedIm2ColConv2dTransAllChan(int rl, int ru, DnnParameters params) {
			super(rl, ru, params);
		}

		@Override
		public Long call() throws Exception {
			final int PQ = _params.P*_params.Q, K = _params.K, CRS = _params.C*_params.R*_params.S;
			MatrixBlock outIm2col = new MatrixBlock(PQ, CRS, _params.input1.sparse).allocateBlock();
			LibMatrixDNNIm2Col.preallocateSparseOutput(_params.input1, outIm2col);
			MatrixBlock outMM = new MatrixBlock(PQ, K, false);
			
			for(int n = _rl; n < _ru; n++)  {
				LibMatrixDNNIm2Col.im2col(_params.input1, outIm2col, n, _params, true);
				
				// t(_im2ColOutBlock) %*% t(filter) => t(matMultOutBlock)
				outMM.reset(outMM.rlen, outMM.clen, false);
				LibMatrixDNNHelper.singleThreadedMatMult(outIm2col, _params.input2, outMM, false, false, _params);
				
				// Copy the matrix matMultOutBlock of shape [K X PQ] to params.output.denseBlock + destPos
				partialCopyTrans(outMM, _params.output, n*K*PQ, K, PQ);
				
				// Add bias to current row if necessary, always dense
				if(_params.bias != null)
					addBias(n, _params.output.getDenseBlockValues(),
						_params.bias.getDenseBlockValues(), K, PQ);
			}
			
			//multi-threaded nnz maintenance of current working set
			return _params.output.recomputeNonZeros(_rl, _ru-1);
		}
		
		private static void partialCopyTrans(MatrixBlock src, MatrixBlock dest, int destPos, int K, int PQ) {
			if( src.isEmptyBlock() )
				return;
			//copy src into its destination row w/ piggybacked transpose
			//src is [PQ x K] -> [K x PQ] -> [1 x KPQ]
			if(src.isInSparseFormat()) {
				SparseBlock sblock = src.sparseBlock;
				double[] c = dest.getDenseBlockValues();
				for(int i = 0; i < src.getNumRows(); i++) {
					if( sblock.isEmpty(i) ) continue;
					int apos = sblock.pos(i);
					int alen = sblock.size(i);
					int[] aix = sblock.indexes(i);
					double[] avals = sblock.values(i);
					int desPosK = destPos + i;
					for(int j = apos; j < apos+alen; j++)
						c[desPosK+aix[j]*PQ] = avals[j];
				}
			}
			else {
				double[] a = src.getDenseBlockValues();
				double[] c = dest.getDenseBlockValues();
				final int blocksizeIJ = 128; //128KB for L2
				//cache-conscious blocked execution
				for( int bi = 0; bi < PQ; bi+=blocksizeIJ )
					for( int bj = 0; bj < K; bj+=blocksizeIJ ) {
						int bimin = Math.min(bi+blocksizeIJ, PQ);
						int bjmin = Math.min(bj+blocksizeIJ, K);
						//core transpose operation
						for(int i=bi, aix=bi*K+bj, cix=bj*PQ+bi; i<bimin; i++, aix+=K, cix++)
							LibMatrixReorg.transposeRow(a, c, aix, destPos+cix, PQ, bjmin-bj);
					}
			}
		}
	}
	
	/**
	 * This operator is used only if native is enabled, filter is dense and input is sparse
	 */
	private static class SparseNativeConv2d implements Callable<Long> 
	{
		public final int _rl, _ru; 
		private final DnnParameters _params;
		public SparseNativeConv2d(int rl, int ru, DnnParameters params) {
			_rl = rl; _ru = ru;
			_params = params;
		}

		@Override
		public Long call() throws Exception {
			int KPQ = _params.K*_params.P*_params.Q;
			double[] temp = new double[KPQ];
			for(int n = _rl; n < _ru; n++)  {
				if( !_params.input1.getSparseBlock().isEmpty(n) ) {
					int apos = _params.input1.getSparseBlock().pos(n);
					int alen = _params.input1.getSparseBlock().size(n);
					int[] aix = _params.input1.getSparseBlock().indexes(n);
					double[] avals = _params.input1.getSparseBlock().values(n);
					NativeHelper.conv2dSparse(apos, alen, aix, avals, _params.input2.getDenseBlockValues(), temp, 
							1, _params.C, _params.H, _params.W, _params.K, _params.R, _params.S, 
							_params.stride_h, _params.stride_w, _params.pad_h, _params.pad_w, _params.P, _params.Q, 1);
					System.arraycopy(temp, 0, _params.output.getDenseBlockValues(), n*KPQ, KPQ);
				}
			}
			//multi-threaded nnz maintenance of current working set
			return _params.output.recomputeNonZeros(_rl, _ru-1);
		}
	}
	
	// BACKWARD DATA
	
	/**
	 * This operator is used only if native is enabled and filter is sparse. 
	 * dout is converted into dense if sparse.
	 */
	private static class SparseNativeConv2dBackwardDataDense implements Callable<Long> 
	{
		public final int _rl, _ru; 
		private final DnnParameters _params; 
		public SparseNativeConv2dBackwardDataDense(int rl, int ru, DnnParameters params) {
			_rl = rl; _ru = ru;
			_params = params;
		}

		@Override
		public Long call() throws Exception {
			int CHW = _params.C*_params.H*_params.W;
			double [] ret = new double[CHW];
			double [] filterArr = _params.input1.getDenseBlockValues();
			double [] dout_n = new double[_params.P*_params.Q*_params.K];
			for(int n = _rl; n < _ru; n++) {
				getRowInDenseFormat(_params.input2, n, dout_n);
				if(n > _rl)
					Arrays.fill(ret, 0);
				NativeHelper.conv2dBackwardDataDense(filterArr, dout_n, ret, 1, 
						_params.C, _params.H, _params.W, _params.K, 
						_params.R, _params.S, _params.stride_h, _params.stride_w, _params.pad_h, _params.pad_w, _params.P, _params.Q, 1);
				System.arraycopy(ret, 0, _params.output.getDenseBlockValues(), n*CHW, CHW);
			}
			//multi-threaded nnz maintenance of current working set
			return _params.output.recomputeNonZeros(_rl, _ru-1);
		}
	}
	
	/**
	 * General conv2d backward data operator
	 */
	private static class Conv2dBackwardData implements Callable<Long> {

		public final int _rl, _ru; 
		private final DnnParameters _params; 
		public Conv2dBackwardData(int rl, int ru, DnnParameters params) {
			_rl = rl; _ru = ru;
			_params = params;
		}
		
		@Override
		public Long call() throws Exception {
			int PQ = _params.P*_params.Q; int K = _params.K; int CRS = _params.C*_params.R*_params.S;
			MatrixBlock filter = _params.input1;
			MatrixBlock dout = _params.input2;
			MatrixBlock outRotate = new MatrixBlock(PQ, K, dout.sparse);
			MatrixBlock outMM = new MatrixBlock(PQ, CRS, false);
			outRotate.allocateBlock();
			LibMatrixDNNRotate180.Rotate180Worker rotate180Worker = 
				LibMatrixDNNRotate180.Rotate180Worker.getWorker( dout, outRotate, _params, true, false);
			for(int n = _rl; n < _ru; n++)  {
				// rotate180(dout[n,]) => dout_reshaped
				rotate180Worker.execute(n, 0);
				// dout_reshaped %*% filter => temp
				outMM.reset(PQ, CRS, false);
				LibMatrixDNNHelper.singleThreadedMatMult(outRotate, filter, outMM, !outRotate.sparse, false, _params);
				// col2im(temp) => output[n,] 
				LibMatrixDNNIm2Col.col2imOverSingleImage(n, outMM, _params);
			}
			
			//multi-threaded nnz maintenance of current working set
			return _params.output.recomputeNonZeros(_rl, _ru-1);
		}
	}
	
	//BACKWARD FILTER
	
	/**
	 * This operator is used only if native is enabled and input is sparse. 
	 * dout is converted into dense if sparse.
	 */
	private static class SparseNativeConv2dBackwardFilterDense implements Callable<Long> 
	{
		public final int _rl, _ru;
		private final DnnParameters _params;
		public SparseNativeConv2dBackwardFilterDense(int rl, int ru, DnnParameters params) {
			_rl = rl; _ru = ru;
			_params = params;
		}
		
		@Override
		public Long call() throws Exception {
			int CRS = _params.C*_params.R*_params.S, PQ = _params.P*_params.Q, K = _params.K;
			MatrixBlock dout_n = new MatrixBlock(PQ, K, false);
			dout_n.allocateBlock();
			LibMatrixDNNRotate180.Rotate180Worker rotate180Worker = 
					LibMatrixDNNRotate180.Rotate180Worker.getWorker( _params.input2, dout_n, _params, true, false);
			double [] ldout_n = dout_n.getDenseBlockValues();
			double [] partRet = new double[CRS*_params.K]; //CRS x K
			for(int n = _rl; n < _ru; n++) {
				if( !_params.input1.getSparseBlock().isEmpty(n) ) {
					// rotate180(dout[n,]) => dout_n
					rotate180Worker.execute(n, 0);
					
					int apos = _params.input1.getSparseBlock().pos(n);
					int alen = _params.input1.getSparseBlock().size(n);
					int[] aix = _params.input1.getSparseBlock().indexes(n);
					double[] avals = _params.input1.getSparseBlock().values(n);
					NativeHelper.conv2dBackwardFilterSparseDense(apos, alen, aix, avals, 
							ldout_n, partRet, 1, _params.C, _params.H, _params.W, _params.K, 
							_params.R, _params.S, _params.stride_h, _params.stride_w, _params.pad_h, _params.pad_w, _params.P, _params.Q, 1);
				}
			}
			inplaceTransAdd(partRet, _params);
			return 0L;
		}
	}
	
	/**
	 * General conv2d backward data operator
	 */
	private static class Conv2dBackwardFilter implements Callable<Long> {
		private final int _rl, _ru; 
		private final DnnParameters _params; 
		
		public Conv2dBackwardFilter(int rl, int ru, DnnParameters params) {
			_rl = rl; _ru = ru;
			_params = params;
		}
		
		@Override
		public Long call() throws Exception {
			int PQ = _params.P*_params.Q, K = _params.K, CRS = _params.C*_params.R*_params.S;
			MatrixBlock dout = _params.input2;
			MatrixBlock im2ColOutBlock = new MatrixBlock(CRS, PQ, _params.input1.sparse).allocateBlock();
			LibMatrixDNNIm2Col.preallocateSparseOutput(_params.input1, im2ColOutBlock);
			MatrixBlock outRotate = new MatrixBlock(PQ, K, dout.sparse);
			MatrixBlock outMM = new MatrixBlock(CRS, K, false);
			outRotate.allocateBlock();
			
			Rotate180Worker rotate180Worker = Rotate180Worker.getWorker( dout, outRotate, _params, true, false);
			double [] partRet = new double[CRS*_params.K];
			for(int n = _rl; n < _ru; n++) {
				// rotate180(dout[n,]) => dout_reshaped
				rotate180Worker.execute(n, 0);
				
				// im2col(input) => _im2ColOutBlock
				LibMatrixDNNIm2Col.im2col(_params.input1, im2ColOutBlock, n, _params, false);
				
				outMM.reset(CRS, K, false);
				LibMatrixDNNHelper.singleThreadedMatMult(im2ColOutBlock, outRotate, outMM, !im2ColOutBlock.sparse, !outRotate.sparse, _params);
				
				if( !outMM.isEmptyBlock() ) //accumulate row results
					LibMatrixMult.vectAdd(outMM.getDenseBlockValues(), partRet, 0, 0, K*CRS);
			}
			inplaceTransAdd(partRet, _params);
			return 0L;
		}
	}
	
	private static class Conv2dBackwardFilterTrans implements Callable<Long> {
		private final int _rl, _ru; 
		private final DnnParameters _params;
		
		public Conv2dBackwardFilterTrans(int rl, int ru, DnnParameters params) {
			_rl = rl; _ru = ru;
			_params = params;
		}
		
		@Override
		public Long call() throws Exception {
			int PQ = _params.P*_params.Q, K = _params.K, CRS = _params.C*_params.R*_params.S;
			MatrixBlock dout = _params.input2;
			MatrixBlock im2ColOutBlock = new MatrixBlock(PQ, CRS, _params.input1.sparse).allocateBlock();
			LibMatrixDNNIm2Col.preallocateSparseOutput(_params.input1, im2ColOutBlock);
			MatrixBlock outRotate = new MatrixBlock(K, PQ, dout.sparse).allocateBlock();
			MatrixBlock outMM = new MatrixBlock(K, CRS, false).allocateBlock();
			Rotate180Worker rotate180Worker = Rotate180Worker.getWorker( dout, outRotate, _params, true, true);
			double [] partRet = new double[CRS*_params.K];
			for(int n = _rl; n < _ru; n++) {
				// rotate180(dout[n,]) => dout_reshaped
				rotate180Worker.execute(n, 0);
				
				// im2col(input) => _im2ColOutBlock
				LibMatrixDNNIm2Col.im2col(_params.input1, im2ColOutBlock, n, _params, true);
				
				outMM.reset(K, CRS, false);
				//Timing time = new Timing(true);
				LibMatrixDNNHelper.singleThreadedMatMult(outRotate, im2ColOutBlock, 
					outMM, !outRotate.sparse, !im2ColOutBlock.sparse, _params);
				
				if( !outMM.isEmptyBlock() ) //accumulate row results
					LibMatrixMult.vectAdd(outMM.getDenseBlockValues(), partRet, 0, 0, K*CRS);
			}
			//no need to transpose because t(t(out)) cancel out
			inplaceAdd(partRet, _params);
			return 0L;
		}
	}
	
	private static void inplaceAdd(double[] a, DnnParameters params) {
		synchronized (params.output.denseBlock) {
			LibMatrixMult.vectAdd(a, params.output.getDenseBlockValues(), 0, 0, a.length);
		}
	}
	
	private static void inplaceTransAdd(double[] a, DnnParameters params) {
		synchronized (params.output.denseBlock) {
			// Perform transposed addition: output of size [K, CRS] += input of size [CRS,K]
			double [] c = params.output.getDenseBlockValues();
			final int CRS = params.C*params.R*params.S, K = params.K;
			final int blocksizeIJ = 128; //L2 cache
			
			//cache-conscious blocked execution
			for( int bi=0; bi<CRS; bi+=blocksizeIJ )
				for( int bj=0; bj<K; bj+=blocksizeIJ ) {
					int bimin = Math.min(bi+blocksizeIJ, CRS);
					int bjmin = Math.min(bj+blocksizeIJ, K);
					//core transpose add operation
					for(int i=bi, aix=bi*K; i<bimin; i++, aix+=K)
						for(int j=bj, cix=i+bj*CRS; j<bjmin; j++, cix+=CRS)
							c[cix] += a[aix+j];
				}
		}
	}
	
	private static void getRowInDenseFormat(MatrixBlock input, int n, double []  ret) {
		if(input.getNumColumns() != ret.length) {
			throw new DMLRuntimeException("Invalid parameters");
		}
		// Use temporary array to avoid binary search
		if(input.isInSparseFormat()) {
			Arrays.fill(ret, 0);
			if( !input.sparseBlock.isEmpty(n) ) {
				int apos = input.sparseBlock.pos(n);
				int alen = input.sparseBlock.size(n);
				int[] aix = input.sparseBlock.indexes(n);
				double[] avals = input.sparseBlock.values(n);
				for(int j=apos; j<apos+alen; j++)
					ret[ aix[j] ] = avals[j];
			}
		}
		else {
			System.arraycopy(input.getDenseBlockValues(),
				n*input.getNumColumns(), ret, 0, input.getNumColumns());
		}
	}
	
	private static void addBias(int r, double [] out, double [] bias, int K, int PQ) {
		for(int k=0, cix=r*K*PQ; k<K; k++, cix+=PQ)
			LibMatrixMult.vectAddInPlace(bias[k], out, cix, PQ);
	}
	
	// ----------------------------------------------------------------------------------------------
	// TODO: Support sparse native convolution operations without dense intermediates + dense matmult
	// Currently, it will fall back to more optimized sparse Java-based operators.
	private static boolean isEligibleForConv2dBackwardFilterSparseDense(DnnParameters params) {
		// NativeHelper.conv2dBackwardFilterSparseDense only if input is sparse. 
		// dout converted to dense if sparse.
		// return params.enableNative && params.input1.isInSparseFormat();
		return false;
	}
	
	private static boolean isEligibleForConv2dSparse(DnnParameters params) {
		// NativeHelper.conv2dSparse only if filter is dense and input is sparse
		// return params.enableNative && params.input1.isInSparseFormat() && !params.input2.isInSparseFormat();
		return false;
	}
	
	private static boolean isEligibleForConv2dBackwardDataDense(DnnParameters params) {
		// NativeHelper.conv2dBackwardDataDense only if filter is dense. 
		// dout converted to dense if sparse.
		// return params.enableNative && !params.input1.isInSparseFormat();
		return false;
	}
	// ----------------------------------------------------------------------------------------------
}
