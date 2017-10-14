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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

/**
 * This class contains the different implementation of im2col operation
 */
public class LibMatrixDNNIm2ColHelper {
	private static final Log LOG = LogFactory.getLog(LibMatrixDNNIm2ColHelper.class.getName());
	static interface Im2colWorker {
		public void execute(int n);
		public void execute(int n, int c);
		public static Im2colWorker getWorker(MatrixBlock input, MatrixBlock out, ConvolutionParameters params, boolean allChannels, boolean trans) {
			if(!input.isInSparseFormat()) {
				boolean stride1Pad0 = params.stride_h == 1 && params.stride_w == 1 
						&& params.pad_h == 0 && params.pad_w == 0;
				// Note: Only dense im2col operators require the im2ColOutBlock to be allocated in the dense format.
				out.reset(out.rlen, out.clen, false);
				out.allocateDenseBlock();
				if( LOG.isTraceEnabled() ) 
					LOG.trace("Using DenseIm2colWorkerAllChannels operator to perform "
						+ "im2col (stride1pad0="+stride1Pad0+", allChannels="+allChannels+").");
				if(allChannels && stride1Pad0 )
					return new DenseIm2colWorkerStride1Pad0AllChannels(input.getDenseBlock(), out.getDenseBlock(), params);
				else if( allChannels )
					return new DenseIm2colWorkerAllChannels(input.getDenseBlock(), out.getDenseBlock(), params);
				else if( stride1Pad0 )
					return new DenseIm2colWorkerStride1Pad0(input.getDenseBlock(), out.getDenseBlock(), params);
				else
					return new DenseIm2colWorker(input.getDenseBlock(), out.getDenseBlock(), params);
			}
			else {
				if(LOG.isTraceEnabled()) 
					LOG.trace("Using SparseIm2colWorker operator to perform im2col.");
				out.reset(out.rlen, out.clen, true);
				out.allocateSparseRowsBlock();
				//preallocate sparse-rows
				double sparsity = Math.min(MatrixBlock.SPARSITY_TURN_POINT, 
					(input.getNonZeros()*2.0) / (input.getNumRows()*input.getNumColumns()));
				int estnnz = (int)Math.ceil((trans ? params.C*params.R*params.S : params.P*params.Q)*sparsity);
				for(int r = 0; r < out.rlen; r++)
					out.getSparseBlock().allocate(r, estnnz);
				
				return new SparseSparseIm2colWorkerAllChan(input, out, params, trans);
			}
		}
	}
	
	/**
	 * Special case operator for performing dense im2col when stride = [1, 1] and pad = [0, 0] by using System.arraycopy
	 */
	static class DenseIm2colWorkerStride1Pad0 implements Im2colWorker {
		double [] inputArray; double [] outputArray; 
		int CRS; int S; int R; int P; int Q; int CHW; int H; int W;
		public DenseIm2colWorkerStride1Pad0(double [] inputArray, double [] outputArray, ConvolutionParameters params) {
			this.inputArray = inputArray;
			this.outputArray = outputArray;
			this.CRS = params.C * params.R * params.S;
			this.H = params.H; this.W = params.W; this.R = params.R; this.S = params.S; this.P = params.P; this.Q = params.Q;
			this.CHW = params.C*params.H*params.W;
		}
		
		@Override
		public void execute(int n) {
			throw new RuntimeException("Not supported");
		}

		@Override
		public void execute(int n, int cInput) {
			int nOffset = n * CHW;
			int RS = R*S;
			for (int rs = 0; rs < RS; ++rs) {
				int wOffset = rs % S;
				int hOffset = rs / S;
				for (int h = 0; h < P; ++h) {
					int hPadded = h + hOffset;
					int outOffset = (rs * P + h) * Q;
					int inputOffset = nOffset + (cInput * H + hPadded) * W;
					System.arraycopy(inputArray, inputOffset + wOffset, outputArray, outOffset, Q);
					int w = Q - 1;
					int wPadded = w + wOffset;
					if (hPadded < H && wPadded < W)
						outputArray[outOffset + w] = inputArray[inputOffset + wPadded];
					else
						outputArray[outOffset + w] = 0;
				}
			}
		}
	}

	
		
	/**
	 * Special case operator for performing dense im2col when stride = [1, 1] and pad = [0, 0] by using System.arraycopy
	 */
	static class DenseIm2colWorkerStride1Pad0AllChannels implements Im2colWorker {
		double [] inputArray; double [] outputArray; 
		int CRS; int S; int R; int P; int Q; int CHW; int H; int W;
		public DenseIm2colWorkerStride1Pad0AllChannels(double [] inputArray, double [] outputArray, ConvolutionParameters params) {
			this.inputArray = inputArray;
			this.outputArray = outputArray;
			this.CRS = params.C * params.R * params.S;
			this.H = params.H; this.W = params.W; this.R = params.R; this.S = params.S; this.P = params.P; this.Q = params.Q;
			this.CHW = params.C*params.H*params.W;
		}
		
		@Override
		public void execute(int n, int c) {
			throw new RuntimeException("Not supported");
		}

		@Override
		public void execute(int n) {
			int nOffset = n * CHW;
			for (int c = 0; c < CRS; ++c) {
				int wOffset = c % S;
				int hOffset = (c / S) % R;
				int cInput = c / R / S;
				for (int h = 0; h < P; ++h) {
					int hPadded = h + hOffset;
					int outOffset = (c * P + h) * Q;
					int inputOffset = nOffset + (cInput * H + hPadded) * W;
					System.arraycopy(inputArray, inputOffset + wOffset, outputArray, outOffset, Q);
					int w = Q - 1;
					int wPadded = w + wOffset;
					if (hPadded < H && wPadded < W)
						outputArray[outOffset + w] = inputArray[inputOffset + wPadded];
					else
						outputArray[outOffset + w] = 0;
				}
			}
		}
	}
	
	/**
	 * Performing dense im2col (general case)
	 */
	static class DenseIm2colWorker implements Im2colWorker {
		double [] inputArray; double [] outputArray; 
		int CRS; int S; int R; int P; int Q; int CHW; int H; int W; 
		int stride_h; int stride_w; int pad_h; int pad_w;
		public DenseIm2colWorker(double [] inputArray, double [] outputArray, ConvolutionParameters params) {
			this.inputArray = inputArray;
			this.outputArray = outputArray;
			this.CRS = params.C * params.R * params.S;
			this.H = params.H; this.W = params.W; this.R = params.R; this.S = params.S; this.P = params.P; this.Q = params.Q;
			this.CHW = params.C*params.H*params.W;
			this.stride_h = params.stride_h; this.stride_w = params.stride_w;
			this.pad_h = params.pad_h; this.pad_w = params.pad_w;
		}
		
		@Override
		public void execute(int n) {
			throw new RuntimeException("Not supported");
		}

		@Override
		public void execute(int n, int cInput) {
			int nOffset = n * CHW; int RS = R*S;
			for (int rs = 0; rs < RS; ++rs) {
				int wOffset = rs % S;
				int hOffset = rs / S;
				for (int h = 0; h < P; ++h) {
					int outOffset = (rs * P + h) * Q;
					int hPadded = h * stride_h - pad_h + hOffset;
					int inputOffset = nOffset + (cInput * H + hPadded) * W;
					if (hPadded < 0 || hPadded >= H) {
						Arrays.fill(outputArray, outOffset, outOffset+Q, 0);
					} else {
						for (int w = 0; w < Q; ++w) {
							int wPadded = w * stride_w - pad_w + wOffset;
							if (wPadded >= 0 && wPadded < W)
								outputArray[outOffset + w] = inputArray[inputOffset + wPadded];
							else
								outputArray[outOffset + w] = 0;
						}
					}
				}
			}
		}
	}
	
	/**
	 * Performing dense im2col (general case)
	 */
	private static class DenseIm2colWorkerAllChannels implements Im2colWorker {
		double [] inputArray; double [] outputArray; 
		int CRS; int S; int R; int P; int Q; int CHW; int H; int W; 
		int stride_h; int stride_w; int pad_h; int pad_w;
		public DenseIm2colWorkerAllChannels(double [] inputArray, double [] outputArray, ConvolutionParameters params) {
			this.inputArray = inputArray;
			this.outputArray = outputArray;
			this.CRS = params.C * params.R * params.S;
			this.H = params.H; this.W = params.W; this.R = params.R; this.S = params.S; this.P = params.P; this.Q = params.Q;
			this.CHW = params.C*params.H*params.W;
			this.stride_h = params.stride_h; this.stride_w = params.stride_w;
			this.pad_h = params.pad_h; this.pad_w = params.pad_w;
		}
		
		@Override
		public void execute(int n, int c) {
			throw new RuntimeException("Not supported");
		}

		@Override
		public void execute(int n) {
			int nOffset = n * CHW;
			for (int c = 0; c < CRS; ++c) {
				int wOffset = c % S;
				int hOffset = (c / S) % R;
				int cInput = c / R / S;
				for (int h = 0; h < P; ++h) {
					int outOffset = (c * P + h) * Q;
					int hPadded = h * stride_h - pad_h + hOffset;
					int inputOffset = nOffset + (cInput * H + hPadded) * W;
					if (hPadded < 0 || hPadded >= H) {
						Arrays.fill(outputArray, outOffset, outOffset+Q, 0);
					} else {
						for (int w = 0; w < Q; ++w) {
							int wPadded = w * stride_w - pad_w + wOffset;
							if (wPadded >= 0 && wPadded < W)
								outputArray[outOffset + w] = inputArray[inputOffset + wPadded];
							else
								outputArray[outOffset + w] = 0;
						}
					}
				}
			}
		}
	}
	
	/**
	 * Performing sparse im2col for all channels for a given row n of the input matrix.
	 */
	private static class SparseSparseIm2colWorkerAllChan implements Im2colWorker {
		private final MatrixBlock input, output;
		private final int S, R, P, Q, W, HW;
		private final int stride_h, stride_w, pad_h, pad_w;
		private final boolean trans;
		public SparseSparseIm2colWorkerAllChan(MatrixBlock input, MatrixBlock im2ColOutBlock, ConvolutionParameters params, boolean trans) {
			this.input = input;
			this.output = im2ColOutBlock;
			this.HW = params.H * params.W;
			this.W = params.W; this.R = params.R; this.S = params.S; this.P = params.P; this.Q = params.Q;
			this.stride_h = params.stride_h; this.stride_w = params.stride_w;
			this.pad_h = params.pad_h; this.pad_w = params.pad_w;
			this.trans = trans;
			if(!input.isInSparseFormat()) 
				throw new RuntimeException("Incorrect operator selection. Expected dense input for SparseIm2colWorkerAllChannels");
		}
		
		@Override
		public void execute(int n, int c) {
			throw new RuntimeException("Not supported");
		}

		@Override
		public void execute(int n) {
			output.reset();
			SparseBlock sblock = input.sparseBlock;
			if( sblock.isEmpty(n) )
				return;
			int apos = sblock.pos(n);
			int alen = sblock.size(n);
			int[] aix = sblock.indexes(n);
			double[] avals = sblock.values(n);
			
			// Iterate over the sparse block
			for(int j=apos; j<apos+alen; j++) {
				// Note: the input is of shape [N, CHW]
				int chw = aix[j];
				
				// Get individual zero-based c,h,w indexes from zero-based 'chw'
				int cInput = chw / HW;
				int hInput = (chw - cInput*HW)/W;
				int wInput = chw % W; 
				
				appendInputValueToIm2colOutput(output, cInput, hInput, wInput, avals[j], 
						R, S, P, Q, stride_h, stride_w, pad_h, pad_w, trans);
			}
			// Since the chw are appended in sorted order, no need to sort the output rows
			// unless in trans mode, then sorting is needed
			if( trans )
				output.sortSparseRows();
		}
	}
	
	/**
	 * Performing sparse im2col for a given channel c and for a given row n of the input matrix.
	 */
	@SuppressWarnings("unused")
	private static class SparseSparseIm2colWorker implements Im2colWorker {
		private final MatrixBlock input, output;
		private final int S, R, P, Q, W, HW;
		private final int stride_h, stride_w, pad_h, pad_w; 
		final boolean trans;
		
		public SparseSparseIm2colWorker(MatrixBlock input, MatrixBlock im2ColOutBlock, ConvolutionParameters params, boolean trans) {
			this.input = input;
			this.output = im2ColOutBlock;
			this.HW = params.H*params.W;
			this.W = params.W; this.R = params.R; this.S = params.S; this.P = params.P; this.Q = params.Q;
			this.stride_h = params.stride_h; this.stride_w = params.stride_w;
			this.pad_h = params.pad_h; this.pad_w = params.pad_w;
			this.trans = trans;
		}
		
		@Override
		public void execute(int n) {
			throw new RuntimeException("Not supported");
		}

		@Override
		public void execute(int n, int cInput) {
			output.reset();
			SparseBlock sblock = input.sparseBlock;
			if( sblock.isEmpty(n) )
				return;
			int apos = sblock.pos(n);
			int alen = sblock.size(n);
			int[] aix = sblock.indexes(n);
			double[] avals = sblock.values(n);
				
			// Iterate over the sparse block
			for(int j=apos; j<apos+alen; j++) {
				// Note: the input is of shape [N, CHW]
				int chw = aix[j];
				
				if(cInput == (chw / HW)) {
					// Get individual zero-based c,h,w indexes from zero-based 'chw'
					int hInput = (chw - cInput*HW)/W;
					int wInput = chw % W; 
					
					appendInputValueToIm2colOutput(output, cInput, hInput, wInput, avals[j], 
							R, S, P, Q, stride_h, stride_w, pad_h, pad_w, trans);
				}
			}
			// Since the chw are appended in sorted order, no need to sort the output rows
			// unless in trans mode, then sorting is needed
			if( trans )
				output.sortSparseRows();
		}
	}
	
	/**
	 * Appends the value corresponding to the given [, cInput, hInput, wInput] to the appropriate im2col location of the output
	 * 
	 * @param output output matrix block
	 * @param cInput input channel index (zero-based)
	 * @param hInput input height index (zero-based)
	 * @param wInput input width index (zero-based)
	 * @param value input value
	 * @param R filter height
	 * @param S filter width
	 * @param P output height
	 * @param Q output width
	 * @param stride_h stride height
	 * @param stride_w stride width
	 * @param pad_h pad height
	 * @param pad_w pad width
	 */
	private static void appendInputValueToIm2colOutput(MatrixBlock output, int cInput, int hInput, int wInput, double value, 
			int R, int S, int P, int Q, int stride_h, int stride_w, int pad_h, int pad_w, boolean trans) {
		if(value == 0) 
			return;
		int RS = R*S;
		// For the given h,w index, insert avals[j] into respective r,s,p,q locations
		
		// Constraints: for(int r = 0; r < R; r++) { if(0 <= p && p < P && (hInput - r + pad_h) % stride_h == 0) { ... } }
		// Constraint 1: p >= 0 and p = (hInput - r + pad_h)  / stride_h
		// Therefore,  r <= hInput + pad_h 
		// Constraint 2: p < P and p = (hInput - r + pad_h)  / stride_h
		// Therefore,  hInput + pad_h - P*stride_h < r
		// Math.max(0, hInput + pad_h - P*stride_h + 1) <= r <= Math.min(R-1, hInput + pad_h)
		int rMin = Math.max(0, hInput + pad_h - P*stride_h + 1);
		int rMax = Math.min(R-1, hInput + pad_h);
		int sMin = Math.max(0, wInput + pad_w - Q*stride_w + 1);
		int sMax = Math.min(S-1, wInput + pad_w);
		// Constraint 3: (hInput - r + pad_h) % stride_h == 0
		while((hInput - rMin + pad_h) % stride_h != 0 && rMin <= rMax) rMin++;
		while((wInput - sMin + pad_w) % stride_w != 0 && sMin <= sMax) sMin++;
		
		for(int r = rMin; r <= rMax; r += stride_h) {
			// Only append value if h == hInput, where h = (r - pad_h) + p*stride_h and 0 <= p < P
			// Therefore, p = (hInput - r + pad_h)  / stride_h. Use the same logic for q.
			final int p = (hInput - r + pad_h)  / stride_h;
			final int pQ = p*Q;
			final int outRowIndex = cInput*RS + r*S;
			for(int s = sMin; s <= sMax; s += stride_w) {
				int q = (wInput - s + pad_w)  / stride_w;
				// chw -> [crs, pq]
				if( trans )
					output.appendValue(pQ + q, outRowIndex + s, value);
				else
					output.appendValue(outRowIndex + s, pQ + q, value);
				// Since the chw are appended in sorted order, no need to sort the output rows
			}
		}
	}
}
