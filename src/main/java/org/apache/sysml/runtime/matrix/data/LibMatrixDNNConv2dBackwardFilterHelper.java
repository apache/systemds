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

import java.util.concurrent.Callable;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.runtime.matrix.data.LibMatrixDNNIm2ColHelper.Im2colWorker;
import org.apache.sysml.runtime.matrix.data.LibMatrixDNNRotate180Helper.Rotate180Worker;
import org.apache.sysml.utils.NativeHelper;

public class LibMatrixDNNConv2dBackwardFilterHelper {

	/**
	 * This operator is used only if native is enabled and input is sparse. 
	 * dout is converted into dense if sparse.
	 */
	public static class SparseNativeConv2dBackwardFilterDense implements Callable<Long> 
	{

		public int _rl; public int _ru; 
		private final ConvolutionParameters _params; 
		public SparseNativeConv2dBackwardFilterDense(int rl, int ru, ConvolutionParameters params) {
			_rl = rl; _ru = ru;
			_params = params;
		}
		
		@Override
		public Long call() throws Exception {
			int CRS = _params.C*_params.R*_params.S, PQ = _params.P*_params.Q, K = _params.K;
			MatrixBlock dout_n = new MatrixBlock(PQ, K, false);
			dout_n.allocateBlock();
			LibMatrixDNNRotate180Helper.Rotate180Worker rotate180Worker = 
					LibMatrixDNNRotate180Helper.Rotate180Worker.getWorker( _params.input2, dout_n, _params, true, false);
			double [] ldout_n = dout_n.getDenseBlock();
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
	public static class Conv2dBackwardFilter implements Callable<Long> {
		private final int _rl, _ru; 
		private final ConvolutionParameters _params; 
		
		public Conv2dBackwardFilter(int rl, int ru, ConvolutionParameters params) {
			_rl = rl; _ru = ru;
			_params = params;
		}
		
		@Override
		public Long call() throws Exception {
			int PQ = _params.P*_params.Q, K = _params.K, CRS = _params.C*_params.R*_params.S;
			MatrixBlock dout = _params.input2;
			MatrixBlock im2ColOutBlock = new MatrixBlock(CRS, PQ, false);
			MatrixBlock outRotate = new MatrixBlock(PQ, K, dout.sparse);
			MatrixBlock outMM = new MatrixBlock(CRS, K, false);
			outRotate.allocateBlock();
			
			Im2colWorker im2ColWorker = Im2colWorker.getWorker( _params.input1, im2ColOutBlock, _params, true, false);
			Rotate180Worker rotate180Worker = Rotate180Worker.getWorker( dout, outRotate, _params, true, false);
			double [] partRet = new double[CRS*_params.K];
			long time1 = 0; long time2 = 0;
			for(int n = _rl; n < _ru; n++) {
				// rotate180(dout[n,]) => dout_reshaped
				rotate180Worker.execute(n, 0);
				
				// im2col(input) => _im2ColOutBlock
				long t1 = DMLScript.FINEGRAINED_STATISTICS ? System.nanoTime() : 0;
				im2ColWorker.execute(n);
				long t2 = DMLScript.FINEGRAINED_STATISTICS ? System.nanoTime() : 0;
				
				outMM.reset(CRS, K, false);
				LibMatrixDNNHelper.singleThreadedMatMult(im2ColOutBlock, outRotate, outMM, !im2ColOutBlock.sparse, !outRotate.sparse, _params);
				long t3 = DMLScript.FINEGRAINED_STATISTICS ? System.nanoTime() : 0;
				
				if( !outMM.isEmptyBlock() ) //accumulate row results
					LibMatrixMult.vectAdd(outMM.getDenseBlock(), partRet, 0, 0, K*CRS);
				
				if(DMLScript.FINEGRAINED_STATISTICS) {
					time1 += t2 - t1;
					time2 += t3 - t2;
				}
			}
			inplaceTransAdd(partRet, _params);
			if(DMLScript.FINEGRAINED_STATISTICS) {
				LibMatrixDNN.loopedConvBwdFilterIm2ColTime.addAndGet(time1);
				LibMatrixDNN.loopedConvBwdFilterMatMultTime.addAndGet(time2);
			}
			return 0L;
		}
	}
	
	public static class Conv2dBackwardFilterTrans implements Callable<Long> {
		private final int _rl, _ru; 
		private final ConvolutionParameters _params; 
		
		public Conv2dBackwardFilterTrans(int rl, int ru, ConvolutionParameters params) {
			_rl = rl; _ru = ru;
			_params = params;
		}
		
		@Override
		public Long call() throws Exception {
			int PQ = _params.P*_params.Q, K = _params.K, CRS = _params.C*_params.R*_params.S;
			MatrixBlock dout = _params.input2;
			MatrixBlock im2ColOutBlock = new MatrixBlock(PQ, CRS, false).allocateBlock();
			MatrixBlock outRotate = new MatrixBlock(K, PQ, dout.sparse).allocateBlock();
			MatrixBlock outMM = new MatrixBlock(K, CRS, false).allocateBlock();
			
			Im2colWorker im2ColWorker = Im2colWorker.getWorker( _params.input1, im2ColOutBlock, _params, true, true);
			Rotate180Worker rotate180Worker = Rotate180Worker.getWorker( dout, outRotate, _params, true, true);
			double [] partRet = new double[CRS*_params.K];
			long time1 = 0; long time2 = 0;
			for(int n = _rl; n < _ru; n++) {
				// rotate180(dout[n,]) => dout_reshaped
				rotate180Worker.execute(n, 0);
				
				// im2col(input) => _im2ColOutBlock
				long t1 = DMLScript.FINEGRAINED_STATISTICS ? System.nanoTime() : 0;
				im2ColWorker.execute(n);
				long t2 = DMLScript.FINEGRAINED_STATISTICS ? System.nanoTime() : 0;
				
				outMM.reset(K, CRS, false);
				//Timing time = new Timing(true);
				LibMatrixDNNHelper.singleThreadedMatMult(outRotate, im2ColOutBlock, 
					outMM, !outRotate.sparse, !im2ColOutBlock.sparse, _params);
				long t3 = DMLScript.FINEGRAINED_STATISTICS ? System.nanoTime() : 0;
				
				if( !outMM.isEmptyBlock() ) //accumulate row results
					LibMatrixMult.vectAdd(outMM.getDenseBlock(), partRet, 0, 0, K*CRS);
				
				if(DMLScript.FINEGRAINED_STATISTICS) {
					time1 += t2 - t1;
					time2 += t3 - t2;
				}
			}
			//no need to transpose because t(t(out)) cancel out
			inplaceAdd(partRet, _params);
			if(DMLScript.FINEGRAINED_STATISTICS) {
				LibMatrixDNN.loopedConvBwdFilterIm2ColTime.addAndGet(time1);
				LibMatrixDNN.loopedConvBwdFilterMatMultTime.addAndGet(time2);
			}
			return 0L;
		}
	}
	
	private static void inplaceAdd(double[] a, ConvolutionParameters params) {
		synchronized (params.output.denseBlock) {
			LibMatrixMult.vectAdd(a, params.output.denseBlock, 0, 0, a.length);
		}
	}
	
	private static void inplaceTransAdd(double[] a, ConvolutionParameters params) {
		synchronized (params.output.denseBlock) {
			// Perform transposed addition: output of size [K, CRS] += input of size [CRS,K]
			double [] c = params.output.denseBlock;
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
}
