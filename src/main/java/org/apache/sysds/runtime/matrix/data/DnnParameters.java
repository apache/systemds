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

import java.io.Serializable;

import org.apache.sysds.hops.Hop;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.util.DnnUtils;

/**
 * This class is container that stores parameters required for executing following operations:
 * conv2d, conv2d_backward_data, conv2d_backward_filter, maxpooling, maxpooling_backward, lstm, lstm_backward
 */
public class DnnParameters implements Serializable 
{
	private static final long serialVersionUID = -212362627205772829L;
	
	public int N, C, H, W, K, R, S, P, Q, D, T, M;
	public int stride_h, stride_w, pad_h, pad_w;
	public int numThreads;
	
	// Optional variables used by ConvolutionCPInstruction
	public boolean enableNative = false;
	public boolean return_sequences;

	public MatrixBlock input1; public MatrixBlock input2; public MatrixBlock output;
	public MatrixBlock input3, input4, input5, input6, input7, input8, input9, output2, output3, output4, output5;
	
	public MatrixBlock bias;
	public int [] start_indexes_h, end_indexes_h, start_indexes_w, end_indexes_w;
	
	public double minValForMaxPoolOperations = -Double.MAX_VALUE; 
	
	public DnnParameters(long N, long C, long H, long W,
			long K, long R, long S, long stride_h, long stride_w, 
			long pad_h, long pad_w, int numThreads) {
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
		
		if(W >= 0 && pad_w >= 0 && S >= 0 && stride_w >= 0)
			Q = (int) ((W + 2 * pad_w - S) / stride_w + 1);
		else
			Q = -1;
		
		this.numThreads = numThreads;
	}
	
	public DnnParameters(int N, int C, int H, int W,
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
		if(H <= 0 || R <= 0 || stride_h < 0 || pad_h < 0)
			P = -1;
		else
			P = (int) DnnUtils.getP(H, R, stride_h, pad_h);
		if(W <= 0 || S <= 0 || stride_w < 0 || pad_w < 0)
			Q = -1;
		else
			Q = (int) DnnUtils.getQ(W, S, stride_w, pad_w);
		this.numThreads = numThreads;
	}

	public DnnParameters(int N, int D, int T, int M, MatrixBlock x, MatrixBlock w, MatrixBlock bias, MatrixBlock out0,
						 MatrixBlock c0, boolean return_sequences, int numThreads){
		this.N = N;
		this.D = D;
		this.T = T;
		this.M = M;

		this.input1 = x;
		this.input2 = w;
		this.bias = bias;
		this.input3 = out0;
		this.input4 = c0;

		this.return_sequences = return_sequences;
		this.numThreads = numThreads;
	}

	public DnnParameters(int n, int d, int t, int m, MatrixBlock x, MatrixBlock w, MatrixBlock bias, MatrixBlock out0, MatrixBlock c0, MatrixBlock cache_out, MatrixBlock cache_c, MatrixBlock cache_ifog, boolean return_sequences, MatrixBlock dout, MatrixBlock dc, MatrixBlock dx, MatrixBlock dw, MatrixBlock db, MatrixBlock dout0, MatrixBlock dc0, int numThreads) {
		this(n, d, t, m, x, w, bias, out0, c0,  return_sequences, numThreads);
		this.input5 = dout;
		this.input6 = dc;
		this.input7 = cache_out;
		this.input8 = cache_c;
		this.input9 = cache_ifog;
		this.output = dx;
		this.output2 = dw;
		this.output3 = db;
		this.output4 = dout0;
		this.output5 = dc0;
	}


	private static int convertToInt(long val) {
		if( val > Integer.MAX_VALUE )
			throw new DMLRuntimeException("The value for DnnParameters is too large:" + val);
		return (int) val;
	}
	
	public boolean compare(DnnParameters that) {
		if(this.N == that.N && this.C == that.C && this.H == that.H && this.W == that.W
				&& this.K == that.K && this.R == that.R && this.S == that.S && this.stride_h == that.stride_h
				 && this.stride_w == that.stride_w  && this.pad_h == that.pad_h
				  && this.pad_w == that.pad_w   && this.numThreads == that.numThreads) {
			return true;
		}
		return false;
	}
	
	@Override
	public String toString() {
		return "(NCHW=[" + N + " " + C + " " + H + " " + W + "], KCRS=[" + K + " " + R + " " + S + "], stride=[" + stride_h + "," + stride_w  + 
				"], pad=[" + pad_h + "," + pad_w + "])";  
	}
	
	public void setIfUnknown(Hop N, Hop C, Hop H, Hop W,
			Hop K, Hop R, Hop S, Hop stride_h, Hop stride_w, Hop pad_h, Hop pad_w, int numThreads) {
		if(this.N < 0) this.N = convertToInt(Hop.computeSizeInformation(N));
		if(this.C < 0) this.C = convertToInt(Hop.computeSizeInformation(C));
		if(this.H < 0) this.H = convertToInt(Hop.computeSizeInformation(H));
		if(this.W < 0) this.W = convertToInt(Hop.computeSizeInformation(W));
		if(this.K < 0) this.K = convertToInt(Hop.computeSizeInformation(K));
		if(this.R < 0) this.R = convertToInt(Hop.computeSizeInformation(R));
		if(this.S < 0) this.S = convertToInt(Hop.computeSizeInformation(S));
		if(this.stride_h < 0) this.stride_h = convertToInt(Hop.computeSizeInformation(stride_h));
		if(this.stride_w < 0) this.stride_w = convertToInt(Hop.computeSizeInformation(stride_w));
		if(this.pad_h < 0) this.pad_h = convertToInt(Hop.computeSizeInformation(pad_h));
		if(this.pad_w < 0) this.pad_w = convertToInt(Hop.computeSizeInformation(pad_w));
		if(this.P < 0 && this.H >= 0 && this.R >= 0 && this.stride_h >= 0 && this.pad_h >= 0) {
			this.P = (int) DnnUtils.getP(this.H, this.R, this.stride_h, this.pad_h);
		}
		if(this.Q < 0 && this.W >= 0 && this.S >= 0 && this.stride_w >= 0 && this.pad_w >= 0) {
			this.Q = (int) DnnUtils.getQ(this.W, this.S, this.stride_w, this.pad_w);
		}
		this.numThreads = numThreads;
	}
	
	public boolean isOutputThreadSafe() {
		return output.isThreadSafe();
	}
	
	public boolean isStride1Pad0() {
		return (stride_h==1 && stride_w==1
			&& pad_h==0 && pad_w==0);
	}
	
	public boolean isAllOnes(Integer...params) {
		boolean ret = true;
		for(int param : params)
			ret &= (param == 1);
		return ret;
	}
}
