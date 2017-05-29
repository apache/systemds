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
import org.apache.sysml.runtime.util.ConvolutionUtils;
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
			int CRS = _params.C*_params.R*_params.S; 
			double [] dout_n = new double[_params.P*_params.Q*_params.K];
			LibMatrixDNNRotate180Helper.Rotate180Worker rotate180Worker = 
					LibMatrixDNNRotate180Helper.Rotate180Worker.getWorker( _params.input2, dout_n, _params, true);
			// partialRetBlock is size: [params.C*params.R*params.S, params.K]
			double [] partialRetBlock = new double[CRS*_params.K];
			for(int n = _rl; n < _ru; n++) {
				if( !_params.input1.getSparseBlock().isEmpty(n) ) {
					// rotate180(dout[n,]) => dout_n
					rotate180Worker.execute(n, 0);
					
					int apos = _params.input1.getSparseBlock().pos(n);
					int alen = _params.input1.getSparseBlock().size(n);
					int[] aix = _params.input1.getSparseBlock().indexes(n);
					double[] avals = _params.input1.getSparseBlock().values(n);
					NativeHelper.conv2dBackwardFilterSparseDense(apos, alen, aix, avals, 
							dout_n, partialRetBlock, 1, _params.C, _params.H, _params.W, _params.K, 
							_params.R, _params.S, _params.stride_h, _params.stride_w, _params.pad_h, _params.pad_w, _params.P, _params.Q, 1);
				}
			}
			inplaceTransposedAddition(partialRetBlock, _params);
			return 0L;
		}
	}
	
	/**
	 * General conv2d backward data operator
	 */
	public static class Conv2dBackwardFilter implements Callable<Long> {

		public int _rl; public int _ru; 
		private final ConvolutionParameters _params; 
		public Conv2dBackwardFilter(int rl, int ru, ConvolutionParameters params) {
			_rl = rl; _ru = ru;
			_params = params;
		}
		
		@Override
		public Long call() throws Exception {
			int PQ = _params.P*_params.Q; int K = _params.K; int CRS = _params.C*_params.R*_params.S;
			MatrixBlock dout = _params.input2;
			MatrixBlock im2ColOutBlock = new MatrixBlock(CRS, PQ, false);
			im2ColOutBlock.allocateDenseBlock();
			MatrixBlock dout_reshaped = new MatrixBlock(PQ, K, false);
			dout_reshaped.allocateDenseBlock();
			LibMatrixDNNIm2ColHelper.Im2colWorker im2ColWorker = LibMatrixDNNIm2ColHelper.Im2colWorker.getWorker( _params.input1, im2ColOutBlock, _params, true);
			LibMatrixDNNRotate180Helper.Rotate180Worker rotate180Worker = 
					LibMatrixDNNRotate180Helper.Rotate180Worker.getWorker( dout, dout_reshaped.getDenseBlock(), _params, true);
			double [] partialRetBlock = new double[CRS*_params.K];
			long time1 = 0; long time2 = 0;
			for(int n = _rl; n < _ru; n++) {
				// rotate180(dout[n,]) => dout_reshaped
				rotate180Worker.execute(n, 0);
				
				// im2col(input) => _im2ColOutBlock
				long t1 = DMLScript.STATISTICS && LibMatrixDNN.DISPLAY_STATISTICS ? System.nanoTime() : 0;
				im2ColWorker.execute(n);
				long t2 = DMLScript.STATISTICS && LibMatrixDNN.DISPLAY_STATISTICS ? System.nanoTime() : 0;
				
				MatrixBlock temp = new MatrixBlock(CRS, K, false);
				LibMatrixDNNHelper.singleThreadedMatMult(im2ColOutBlock, dout_reshaped, temp, true, true, _params);
				long t3 = DMLScript.STATISTICS && LibMatrixDNN.DISPLAY_STATISTICS ? System.nanoTime() : 0;
				
				if(!temp.isEmptyBlock()) {
					// partialRetBlock is size: [params.C*params.R*params.S, params.K]
					ConvolutionUtils.binaryOperationInPlace(temp, partialRetBlock, 0, K, 0, CRS, 
							LibMatrixDNN._binaryElementWiseAddition);
				}
				
				if(DMLScript.STATISTICS && LibMatrixDNN.DISPLAY_STATISTICS) {
					time1 += t2 - t1;
					time2 += t3 - t2;
				}
			}
			inplaceTransposedAddition(partialRetBlock, _params);
			if(DMLScript.STATISTICS && LibMatrixDNN.DISPLAY_STATISTICS) {
				LibMatrixDNN.loopedConvBwdFilterIm2ColTime.addAndGet(time1);
				LibMatrixDNN.loopedConvBwdFilterMatMultTime.addAndGet(time2);
			}
			return 0L;
		}
	}
	private static synchronized void inplaceTransposedAddition(double [] partialRetBlock, ConvolutionParameters params) {
		// Perform transposed addition: output of size [K, CRS] += partialRetBlock of size [CRS,K]
		int iter = 0; int CRS = params.C*params.R*params.S; int K = params.K;
		double [] outputArr = params.output.denseBlock;
		for(int i = 0; i < CRS; i++) {
			for(int j = 0; j < K; j++, iter++) {
				int index = j*CRS+i;
				outputArr[index] += partialRetBlock[iter];
			}
		}
	}
}
