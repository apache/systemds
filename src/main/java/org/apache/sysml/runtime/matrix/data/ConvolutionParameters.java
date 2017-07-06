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

import java.io.Serializable;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.util.ConvolutionUtils;

/**
 * This class is container that stores parameters required for executing following operations:
 * conv2d, conv2d_backward_data, conv2d_backward_filter, maxpooling, maxpooling_backward 
 */
public class ConvolutionParameters implements Serializable {
	private static final long serialVersionUID = -212362627205772829L;
	public int N; public int C; public int H; public int W;
	public int K; public int R; public int S; public int stride_h; public int stride_w; public int pad_h; public int pad_w;
	public int P; public int Q; public int numThreads;
	
	public boolean enableNative = false;
	public MatrixBlock input1; public MatrixBlock input2; public MatrixBlock output;
	
	public MatrixBlock bias;
	public int [] start_indexes_h, end_indexes_h, start_indexes_w, end_indexes_w; 
	
	private int convertToInt(long val) throws DMLRuntimeException {
		if( val > Integer.MAX_VALUE ) {
			throw new DMLRuntimeException("The value for ConvolutionParameters is too large:" + val);
		}
		return (int) val;
	}
	
	public boolean compare(ConvolutionParameters that) {
		if(this.N == that.N && this.C == that.C && this.H == that.H && this.W == that.W
				&& this.K == that.K && this.R == that.R && this.S == that.S && this.stride_h == that.stride_h
				 && this.stride_w == that.stride_w  && this.pad_h == that.pad_h
				  && this.pad_w == that.pad_w   && this.numThreads == that.numThreads) {
			return true;
		}
		return false;
	}
	
	public String toString() {
		return "(NCHW=[" + N + " " + C + " " + H + " " + W + "], KCRS=[" + K + " " + R + " " + S + "], stride=[" + stride_h + "," + stride_w  + 
				"], pad=[" + pad_h + "," + pad_w + "])";  
	}
	
	public ConvolutionParameters(long N, long C, long H, long W,
			long K, long R, long S, long stride_h, long stride_w, long pad_h, long pad_w, int numThreads) throws DMLRuntimeException {
		this.N = convertToInt(N);
		this.C = convertToInt(C);
		this.H = convertToInt(H);
		this.W = convertToInt(W);
		this.K = convertToInt(K);
		this.R = convertToInt(R);
		this.S = convertToInt(S);
		this.stride_h = convertToInt(stride_h);
		this.stride_w = convertToInt(stride_w);
		this.pad_h = convertToInt(pad_h);
		this.pad_w = convertToInt(pad_w);
		if(H >= 0 && pad_h >= 0 && R >= 0 && stride_h >= 0)
			P = (int) ((H + 2 * pad_h - R) / stride_h + 1);
		else
			P = -1;
		// P = convertToInt(ConvolutionUtils.getP(H, R, stride_h, pad_h));
		
		if(W >= 0 && pad_w >= 0 && S >= 0 && stride_w >= 0)
			Q = (int) ((W + 2 * pad_w - S) / stride_w + 1);
		else
			Q = -1;
		// Q = convertToInt(ConvolutionUtils.getQ(W, S, stride_w, pad_w));
		
		this.numThreads = numThreads;
	}
	
	public ConvolutionParameters(int N, int C, int H, int W,
		int K, int R, int S, int stride_h, int stride_w, int pad_h, int pad_w, int numThreads) {
		this.N = N;
		this.C = C;
		this.H = H;
		this.W = W;
		this.K = K;
		this.R = R;
		this.S = S;
		this.stride_h = stride_h;
		this.stride_w = stride_w;
		this.pad_h = pad_h;
		this.pad_w = pad_w;
		P = (int) ConvolutionUtils.getP(H, R, stride_h, pad_h);
		Q = (int) ConvolutionUtils.getQ(W, S, stride_w, pad_w);
		this.numThreads = numThreads;
	}
	
	public boolean isOutputThreadSafe() {
		return output.isThreadSafe();
	}
}
