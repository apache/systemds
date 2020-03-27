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

import java.util.Arrays;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.LibMatrixDNNHelper.CellIndex3;

/**
 * This class contains the different implementation of im2col operation
 */
public class LibMatrixDNNIm2Col 
{
	public static void im2col(MatrixBlock in, MatrixBlock out, int r, DnnParameters params, boolean trans) {
		im2col(in, out, r, params.C, params.R, params.S, params.H, params.W, params.P, params.Q,
			params.stride_h, params.stride_w, params.pad_h, params.pad_w, trans);
	}
	
	public static void im2col(MatrixBlock in, MatrixBlock out, int r, int C, int R, int S, int H, int W, int P, int Q,
			int stride_h, int stride_w, int pad_h, int pad_w, boolean trans) {
		boolean stride1Pad0 = stride_h == 1 
			&& stride_w == 1 && pad_h == 0 && pad_w == 0;
		
		//dense and sparse operation dispatch
		if( !in.sparse && stride1Pad0 && !trans )
			im2colDenseStride1Pad0(in.getDenseBlockValues(),
				out.getDenseBlockValues(), r*C*H*W, C, R, S, H, W, P, Q);
		else if( !in.sparse )
			im2colDense(in.getDenseBlockValues(), out.getDenseBlockValues(),
				r, C, R, S, H, W, P, Q, stride_h, stride_w, pad_h, pad_w, trans);
		else
			im2colSparse(in, out, r, C, R, S, H, W, P, Q,
				stride_h, stride_w, pad_h, pad_w, trans);
	}
	
	public static void im2colDenseStride1Pad0(double[] in, double[] out, int ai, int C, int R, int S, int H, int W, int P, int Q) {
		int CRS = C * R * S;
		for (int c = 0; c < CRS; ++c) {
			int wOffset = c % S;
			int hOffset = (c / S) % R;
			int cInput = c / R / S;
			for (int h = 0; h < P; ++h) {
				int hPadded = h + hOffset;
				int outOffset = (c * P + h) * Q;
				int inputOffset = ai + (cInput * H + hPadded) * W;
				System.arraycopy(in, inputOffset + wOffset, out, outOffset, Q);
				int w = Q - 1;
				int wPadded = w + wOffset;
				boolean assign = (hPadded < H && wPadded < W);
				out[outOffset + w] = assign ? in[inputOffset + wPadded] : 0;
			}
		}
	}
	
	public static void im2colDense(double[] in, double[] out, int r, int C, int R, int S, int H, int W, int P, int Q,
			int stride_h, int stride_w, int pad_h, int pad_w, boolean trans) {
		Arrays.fill(out, 0); //reset for selective copy
		int CHW = C * H * W;
		int CRS = C * R * S;
		int nOffset = r * CHW;
		for (int c = 0; c < CRS; ++c) {
			int wOffset = c % S;
			int hOffset = (c / S) % R;
			int cInput = c / R / S;
			for (int h = 0; h < P; ++h) {
				int outOffset = trans ? c+(h*Q*CRS) : (c*P+h)*Q;
				int hPadded = h * stride_h - pad_h + hOffset;
				int inputOffset = nOffset + (cInput * H + hPadded) * W;
				if (hPadded < 0 || hPadded >= H ) continue;
				for (int w = 0; w < Q; ++w) {
					int wPadded = w * stride_w - pad_w + wOffset;
					if( wPadded >= 0 && wPadded < W )
						out[outOffset + (trans?w*CRS:w)] 
							= in[inputOffset + wPadded];
				}
			}
		}
	}
	
	public static void im2colSparse(MatrixBlock in, MatrixBlock out, int r, int C, int R, int S, int H, int W, int P, int Q,
			int stride_h, int stride_w, int pad_h, int pad_w, boolean trans) {
		out.reset();
		SparseBlock sblock = in.sparseBlock;
		if( sblock.isEmpty(r) )
			return;
		int apos = sblock.pos(r);
		int alen = sblock.size(r);
		int[] aix = sblock.indexes(r);
		double[] avals = sblock.values(r);
		boolean simple = (stride_h==1 && stride_w==1
			&& pad_h==0 && pad_w==0 && W == S && Q == 1);
		int RS = R * S;
		
		// Iterate over the sparse block
		CellIndex3 ix = new CellIndex3();
		for(int j=apos; j<apos+alen; j++) {
			// Note: the input is of shape [N, CHW]
			int chw = aix[j];
			
			// Get individual zero-based c,h,w indexes from zero-based 'chw'
			ix = LibMatrixDNNHelper.computeTensorIndexes(chw, H, W, ix);
			
			if( simple )
				appendInputValueToIm2colOutputSimple(out, ix.ix1, ix.ix2, ix.ix3, 
					avals[j], R, S, RS, P, trans);
			else
				appendInputValueToIm2colOutput(out, ix.ix1, ix.ix2, ix.ix3, avals[j], 
					R, S, RS, P, Q, stride_h, stride_w, pad_h, pad_w, trans);
		}
		
		out.sortSparseRows();
	}
	
	/**
	 * Appends the value corresponding to the given [, cInput, hInput, wInput] to the appropriate im2col location of the output
	 * 
	 * @param output output matrix block
	 * @param c input channel index (zero-based)
	 * @param h input height index (zero-based)
	 * @param w input width index (zero-based)
	 * @param value input value
	 * @param R filter height
	 * @param S filter width
	 * @param RS R*S
	 * @param P output height
	 * @param Q output width
	 * @param stride_h stride height
	 * @param stride_w stride width
	 * @param pad_h pad height
	 * @param pad_w pad width
	 * @param trans transposed output
	 */
	private static void appendInputValueToIm2colOutput(MatrixBlock output, int c, int h, int w, double value, 
		int R, int S, int RS, int P, int Q, int stride_h, int stride_w, int pad_h, int pad_w, boolean trans)
	{
		// For the given h,w index, insert avals[j] into respective r,s,p,q locations
		// Constraints: for(int r = 0; r < R; r++) { if(0 <= p && p < P && (hInput - r + pad_h) % stride_h == 0) { ... } }
		// Constraint 1: p >= 0 and p = (hInput - r + pad_h)  / stride_h
		// Therefore,  r <= hInput + pad_h 
		// Constraint 2: p < P and p = (hInput - r + pad_h)  / stride_h
		// Therefore,  hInput + pad_h - P*stride_h < r
		// Math.max(0, hInput + pad_h - P*stride_h + 1) <= r <= Math.min(R-1, hInput + pad_h)
		int rMin = Math.max(0, h + pad_h - P*stride_h + 1);
		int rMax = Math.min(R-1, h + pad_h);
		int sMin = Math.max(0, w + pad_w - Q*stride_w + 1);
		int sMax = Math.min(S-1, w + pad_w);
		// Constraint 3: (hInput - r + pad_h) % stride_h == 0
		rMin += Math.min((h-rMin+pad_h) % stride_h, rMax-rMin+1);
		sMin += Math.min((w-sMin+pad_w) % stride_w, sMax-sMin+1);
		
		for( int r=rMin, ix=c*RS+rMin*S; r<=rMax; r+=stride_h, ix+=stride_h*S ) {
			// Only append value if h == hInput, where h = (r - pad_h) + p*stride_h and 0 <= p < P
			// Therefore, p = (hInput - r + pad_h)  / stride_h. Use the same logic for q.
			final int pQ = (h - r + pad_h) / stride_h * Q;
			for(int s=sMin, ws=w-sMin+pad_w; s<=sMax; s+=stride_w, ws-=stride_w) {
				int q = ws / stride_w; // chw -> [crs, pq]
				output.appendValue(trans ? pQ+q : ix+s,
					trans ? ix+s : pQ+q, value);
			}
		}
	}
	
	private static void appendInputValueToIm2colOutputSimple(MatrixBlock output, int c, int h, int w,
		double value, int R, int S, int RS, int P, boolean trans) {
		int rMin = Math.max(0, h - P + 1);
		int rMax = Math.min(R-1, h);
		int cix = c*RS+w+rMin*S;
		for(int p=h-rMin; p >= h-rMax; p--, cix+=S)
			output.appendValue(trans?p:cix, trans?cix:p, value);
	}
	

	// ------------------------------------------------------------------------------------------------------
	// Since col2im always operates on intermediate generated as part of matmult, it is not clear which operator to select apriori.
	// Therefore, it is provided as utility function rather than an operator (like im2col or rotate180)
	
	//Converts input: PQ X CRS matrix and writes to 1 X CHW
	public static void col2imOverSingleImage(int outputN, MatrixBlock input, DnnParameters params) {
		if(input.rlen != params.P*params.Q || input.clen != params.C*params.R*params.S) {
			throw new DMLRuntimeException("Incorrect input dimensions");
		}
		
		double [] outputArray = null;
		if (!params.output.isInSparseFormat())
			outputArray = params.output.getDenseBlockValues();
		else {
			throw new DMLRuntimeException("Only dense output is implemented");
		}
		
		if(!input.isInSparseFormat()) {
			double [] inputArray = input.getDenseBlockValues();
			col2IMDenseInput(0, outputN, inputArray, outputArray, params);
		}
		else {
			if(!input.isEmptyBlock()) {
				int outOffset = outputN*params.C*params.H*params.W;
				int HW = params.H*params.W;
				CellIndex3 ix = new CellIndex3();
				SparseBlock sblock = input.sparseBlock;
				for(int i = 0; i < input.getNumRows(); i++) {
					if( sblock.isEmpty(i) ) continue;
					ix = LibMatrixDNNHelper.computeTensorIndexes(i, params.P, params.Q, ix);
					int tmpP = ix.ix2*params.stride_h - params.pad_h;
					int tmpQ = ix.ix3*params.stride_w - params.pad_w;
					if(ix.ix1 != 0) 
						throw new DMLRuntimeException("Incorrect tensor indexes: "+ ix + ", " + params.P + " " + params.Q);
					int apos = sblock.pos(i);
					int alen = sblock.size(i);
					int[] aix = sblock.indexes(i);
					double[] avals = sblock.values(i);
					for(int j = apos; j < apos+alen; j++) {
						ix = LibMatrixDNNHelper.computeTensorIndexes(aix[j], params.R, params.S, ix);
						int h = tmpP + ix.ix2;
						int w = tmpQ + ix.ix3;
						if(h >= 0 && h < params.H && w >= 0 && w < params.W) {
							int outIndex = outOffset + ix.ix1*HW + h*params.W + w;
							outputArray[outIndex] += avals[j];
						}
					}
				}
			}
		}
	}
	
	// Converts input: PQ X CRS matrix and writes to 1 X CHW if inputN == 0
	// Or converts input: NPQ X CRS matrix and writes to N X CHW 
	private static void col2IMDenseInput(int inputN, int outputN, double [] inputArray, double [] outputArray, DnnParameters params) {
		final int outputNOffset = outputN*params.C*params.H*params.W;
		final int HW = params.H*params.W;
		final int inputNPQ = inputN*params.P*params.Q;
		final int CRS = params.C*params.R*params.S;
		final int RS = params.R*params.S;
		for (int p = 0; p < params.P; p++) {
			// h = p*params.stride_h + r - params.pad_h
			//   = r + hOffset
			// Based on restrictions: h >= 0 and r >= 0 and h < params.H and r < params.R, we get
			// max(0, - hOffset) <= r < min(params.R, params.H - hOffset)
			final int hOffset = p*params.stride_h - params.pad_h;
			final int rStart = Math.max(0, - hOffset);
			final int rEnd = Math.min(params.R, params.H - hOffset);
			for (int q = 0; q < params.Q; q++) {
				// Using the same logic as above on following:
				// w = q*params.stride_w + s - params.pad_w
				final int wOffset = q*params.stride_w - params.pad_w;
				final int sStart = Math.max(0, - wOffset);
				final int sEnd = Math.min(params.S, params.W - wOffset);
				final int tempOffset = (inputNPQ + p*params.Q + q)*CRS;
				for (int c = 0; c < params.C; c++) {
					final int outOffset = outputNOffset + c*HW;
					final int inputOffset = tempOffset + c*RS;
					for (int r = rStart; r < rEnd; r++) {
						for (int s = sStart; s < sEnd; s++) {
							int inputIndex = inputOffset + r*params.S + s;
							int outIndex = outOffset + (hOffset + r)*params.W + wOffset + s;
							outputArray[outIndex] += inputArray[inputIndex];
						}
					}
				}
			}
		}
	}
	
	public static void preallocateSparseOutput(MatrixBlock in, MatrixBlock out) {
		if( !in.sparse )
			return;
		//preallocate sparse-rows (larger than average sparsity to account for skew)
		int estnnz = (int)Math.ceil(4*in.getSparsity()*out.clen);
		for(int r = 0; r < out.rlen; r++)
			out.getSparseBlock().allocate(r, Math.max(Math.min(estnnz, out.clen),16));
	}
}
