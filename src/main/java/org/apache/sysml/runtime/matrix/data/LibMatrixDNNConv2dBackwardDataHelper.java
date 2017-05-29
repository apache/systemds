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

import java.util.Arrays;
import java.util.concurrent.Callable;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.utils.NativeHelper;

/**
 * This class contains the set of operators used for performing conv2d backward data
 */
public class LibMatrixDNNConv2dBackwardDataHelper {

	/**
	 * This operator is used only if native is enabled and filter is sparse. 
	 * dout is converted into dense if sparse.
	 */
	public static class SparseNativeConv2dBackwardDataDense implements Callable<Long> 
	{
		public int _rl; public int _ru; 
		private final ConvolutionParameters _params; 
		public SparseNativeConv2dBackwardDataDense(int rl, int ru, ConvolutionParameters params) {
			_rl = rl; _ru = ru;
			_params = params;
		}

		@Override
		public Long call() throws Exception {
			int CHW = _params.C*_params.H*_params.W;
			double [] ret = new double[CHW];
			double [] filterArr = _params.input1.getDenseBlock();
			double [] dout_n = new double[_params.P*_params.Q*_params.K];
			for(int n = _rl; n < _ru; n++) {
				LibMatrixDNNHelper.getRowInDenseFormat(_params.input2, n, dout_n);
				if(n > _rl)
					Arrays.fill(ret, 0);
				NativeHelper.conv2dBackwardDataDense(filterArr, dout_n, ret, 1, 
						_params.C, _params.H, _params.W, _params.K, 
						_params.R, _params.S, _params.stride_h, _params.stride_w, _params.pad_h, _params.pad_w, _params.P, _params.Q, 1);
				System.arraycopy(ret, 0, _params.output.getDenseBlock(), n*CHW, CHW);
			}
			return 0L;
		}
	}
	
	/**
	 * General conv2d backward data operator
	 */
	public static class Conv2dBackwardData implements Callable<Long> {

		public int _rl; public int _ru; 
		private final ConvolutionParameters _params; 
		public Conv2dBackwardData(int rl, int ru, ConvolutionParameters params) {
			_rl = rl; _ru = ru;
			_params = params;
		}
		
		@Override
		public Long call() throws Exception {
			int PQ = _params.P*_params.Q; int K = _params.K; int CRS = _params.C*_params.R*_params.S;
			MatrixBlock filter = _params.input1;
			MatrixBlock dout = _params.input2;
			MatrixBlock dout_reshaped = new MatrixBlock(PQ, K, false);
			dout_reshaped.allocateDenseBlock();
			LibMatrixDNNRotate180Helper.Rotate180Worker rotate180Worker = 
					LibMatrixDNNRotate180Helper.Rotate180Worker.getWorker( dout, dout_reshaped.getDenseBlock(), _params, true);
			long time1 = 0; long time2 = 0;
			for(int n = _rl; n < _ru; n++)  {
				// rotate180(dout[n,]) => dout_reshaped
				rotate180Worker.execute(n, 0);
				
				// dout_reshaped %*% filter => temp
				MatrixBlock temp = new MatrixBlock(PQ, CRS, false);
				long t1 = DMLScript.STATISTICS && LibMatrixDNN.DISPLAY_STATISTICS ? System.nanoTime() : 0;
				LibMatrixDNNHelper.singleThreadedMatMult(dout_reshaped, filter, temp, true, false, _params);
				long t2 = DMLScript.STATISTICS && LibMatrixDNN.DISPLAY_STATISTICS ? System.nanoTime() : 0;
				// col2im(temp) => output[n,] 
				LibMatrixDNNHelper.doCol2imOverSingleImage(n, temp, _params);
				long t3 = DMLScript.STATISTICS && LibMatrixDNN.DISPLAY_STATISTICS ? System.nanoTime() : 0;
				
				if(DMLScript.STATISTICS && LibMatrixDNN.DISPLAY_STATISTICS) {
					time1 += t2 - t1;
					time2 += t3 - t2;
				}
			}
			if(DMLScript.STATISTICS && LibMatrixDNN.DISPLAY_STATISTICS) {
				LibMatrixDNN.loopedConvBwdDataMatMultTime.addAndGet(time1);
				LibMatrixDNN.loopedConvBwdDataCol2ImTime.addAndGet(time2);
			}
			return 0L;
		}
		
	}
}
