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
		public static Im2colWorker getWorker(MatrixBlock input, MatrixBlock im2ColOutBlock, ConvolutionParameters params, boolean allChannels) {
			if(allChannels) {
				if(!input.isInSparseFormat()) {
					// Note: Only dense im2col operators require the im2ColOutBlock to be allocated in the dense format.
					im2ColOutBlock.allocateDenseBlock();
					if (params.stride_h == 1 && params.stride_w == 1 && params.pad_h == 0 && params.pad_w == 0)  {
						if(LOG.isTraceEnabled()) LOG.trace("Using DenseIm2colWorkerStride1Pad0AllChannels operator to perform im2col.");
						return new DenseIm2colWorkerStride1Pad0AllChannels(input.getDenseBlock(), im2ColOutBlock.getDenseBlock(), params);
					}
					else {
						if(LOG.isTraceEnabled()) LOG.trace("Using DenseIm2colWorkerAllChannels operator to perform im2col.");
						return new DenseIm2colWorkerAllChannels(input.getDenseBlock(), im2ColOutBlock.getDenseBlock(), params);
					}
				}
				else {
					if(LOG.isTraceEnabled()) LOG.trace("Using SparseIm2colWorkerAllChannels operator to perform im2col.");
					long worstCaseNNZ = params.R*params.S*input.getNonZeros();
					im2ColOutBlock.reset(im2ColOutBlock.getNumRows(), im2ColOutBlock.getNumColumns(), true, worstCaseNNZ);
					im2ColOutBlock.allocateSparseRowsBlock();
					return new SparseIm2colWorkerAllChannels(input, im2ColOutBlock, params);
				}
			}
			else {
				if(!input.isInSparseFormat()) {
					// Note: Only dense im2col operators require the im2ColOutBlock to be allocated in the dense format.
					im2ColOutBlock.allocateDenseBlock();
					if (params.stride_h == 1 && params.stride_w == 1 && params.pad_h == 0 && params.pad_w == 0) {
						if(LOG.isTraceEnabled()) LOG.trace("Using DenseIm2colWorkerStride1Pad0 operator to perform im2col.");
						return new DenseIm2colWorkerStride1Pad0(input.getDenseBlock(), im2ColOutBlock.getDenseBlock(), params);
					}
					else {
						if(LOG.isTraceEnabled()) LOG.trace("Using DenseIm2colWorker operator to perform im2col.");
						return new DenseIm2colWorker(input.getDenseBlock(), im2ColOutBlock.getDenseBlock(), params);
					}
				}
				else {
					if(LOG.isTraceEnabled()) LOG.trace("Using SparseIm2colWorker operator to perform im2col.");
					long worstCaseNNZ = params.R*params.S*input.getNonZeros();
					im2ColOutBlock.reset(im2ColOutBlock.getNumRows(), im2ColOutBlock.getNumColumns(), true, worstCaseNNZ);
					im2ColOutBlock.allocateSparseRowsBlock();
					return new SparseIm2colWorker(input, im2ColOutBlock, params);
				}
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
	static class DenseIm2colWorkerAllChannels implements Im2colWorker {
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
	static class SparseIm2colWorkerAllChannels implements Im2colWorker {
		MatrixBlock input;  MatrixBlock output; Im2colSparseMetadata meta;
		int CRS; int S; int R; int P; int Q; int H; int W; int RS; int HW;
		int stride_h; int stride_w; int pad_h; int pad_w;
		public SparseIm2colWorkerAllChannels(MatrixBlock input, MatrixBlock im2ColOutBlock, ConvolutionParameters params) {
			this.input = input;
			this.output = im2ColOutBlock;
			this.CRS = params.C * params.R * params.S;
			this.RS = params.R * params.S;
			this.HW = params.H * params.W;
			this.H = params.H; this.W = params.W; this.R = params.R; this.S = params.S; this.P = params.P; this.Q = params.Q;
			this.stride_h = params.stride_h; this.stride_w = params.stride_w;
			this.pad_h = params.pad_h; this.pad_w = params.pad_w;
			meta = new Im2colSparseMetadata(CRS);
			if(!input.isInSparseFormat()) 
				throw new RuntimeException("Incorrect operator selection. Expected dense input for SparseIm2colWorkerAllChannels");
		}
		
		@Override
		public void execute(int n, int c) {
			throw new RuntimeException("Not supported");
		}

		@Override
		public void execute(int n) {
			if( !input.sparseBlock.isEmpty(n) ) {
				meta.reset();
				output.reset(output.getNumRows(), output.getNumColumns(), true, 0);
				int apos = input.sparseBlock.pos(n);
				int alen = input.sparseBlock.size(n);
				int[] aix = input.sparseBlock.indexes(n);
				double[] avals = input.sparseBlock.values(n);
				
				// Iterate over the sparse block
				for(int j=apos; j<apos+alen; j++) {
					// Note: the input is of shape [N, CHW]
					int chw = aix[j];
					
					// Get individual zero-based c,h,w indexes from zero-based 'chw'
					int cInput = chw / HW;
					int hInput = (chw - cInput*HW)/W;
					int wInput = chw % W; 
					
					appendInputValueToIm2colOutput(output, cInput, hInput, wInput, avals[j], 
							R, S, P, Q, stride_h, stride_w, pad_h, pad_w, meta);
				}
				if(meta.sortRows)
					output.sortSparseRows();
				output.setNonZeros(meta.nnz);
			}
			else {
				output.reset(output.getNumRows(), output.getNumColumns(), true, 0);
				output.setNonZeros(0);
			}
		}
	}
	
	/**
	 * This class stores metadata for maintaining sparse output of im2col.
	 */
	private static class Im2colSparseMetadata {
		public int [] lastIndexPerRow;
		public int nnz;
		public boolean sortRows;
		public Im2colSparseMetadata(int numRows) {
			lastIndexPerRow = new int[numRows];
		}
		public void reset() {
			Arrays.fill(lastIndexPerRow, 0);
			sortRows = false;
			nnz = 0;
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
	 * @param meta computes meta information for the given sparse block
	 */
	private static void appendInputValueToIm2colOutput(MatrixBlock output, int cInput, int hInput, int wInput, double value, 
			int R, int S, int P, int Q, int stride_h, int stride_w, int pad_h, int pad_w, Im2colSparseMetadata meta) {
		if(value == 0) 
			return;
		int RS = R*S;
		// For the given h,w index, insert avals[j] into respective r,s,p,q locations
		for(int r = 0; r < R; r++) {
			// Only append value if h == hInput, where h = (r - pad_h) + p*stride_h and 0 <= p < P
			// Therefore, p = (hInput - r + pad_h)  / stride_h. Use the same logic for q.
			int p = (hInput - r + pad_h)  / stride_h;
			if(0 <= p && p < P && (hInput - r + pad_h) % stride_h == 0) {
				for(int s = 0; s < S; s++) {
					int outRowIndex = cInput*RS + r*S + s;
					int q = (wInput - s + pad_w)  / stride_w;
					if(0 <= q && q < Q && (wInput - s + pad_w) % stride_w == 0) {
						// chw -> [crs, pq]
						output.appendValue(outRowIndex, p*Q + q, value);
						meta.nnz++;
						if(meta.lastIndexPerRow[outRowIndex] > p*Q + q)
							meta.sortRows = true;
						meta.lastIndexPerRow[outRowIndex] = p*Q + q;
					}
				}
			}
		}
	}
	
	/**
	 * Performing sparse im2col for a given channel c and for a given row n of the input matrix.
	 */
	static class SparseIm2colWorker implements Im2colWorker {
		MatrixBlock input; MatrixBlock output; Im2colSparseMetadata meta;
		int CRS; int S; int R; int P; int Q; int H; int W; int HW; int RS;
		int stride_h; int stride_w; int pad_h; int pad_w; double [] temp;
		public SparseIm2colWorker(MatrixBlock input, MatrixBlock im2ColOutBlock, ConvolutionParameters params) {
			this.input = input;
			this.output = im2ColOutBlock;
			this.CRS = params.C * params.R * params.S;
			this.HW = params.H*params.W;
			this.RS = params.R*params.S;
			this.H = params.H; this.W = params.W; this.R = params.R; this.S = params.S; this.P = params.P; this.Q = params.Q;
			this.stride_h = params.stride_h; this.stride_w = params.stride_w;
			this.pad_h = params.pad_h; this.pad_w = params.pad_w;
			temp = new double[input.getNumColumns()];
			meta = new Im2colSparseMetadata(CRS);
		}
		
		@Override
		public void execute(int n) {
			throw new RuntimeException("Not supported");
		}

		@Override
		public void execute(int n, int cInput) {
			if( !input.sparseBlock.isEmpty(n) ) {
				meta.reset();
				output.reset(output.getNumRows(), output.getNumColumns(), true, 0);
				int apos = input.sparseBlock.pos(n);
				int alen = input.sparseBlock.size(n);
				int[] aix = input.sparseBlock.indexes(n);
				double[] avals = input.sparseBlock.values(n);
				
				// Iterate over the sparse block
				for(int j=apos; j<apos+alen; j++) {
					// Note: the input is of shape [N, CHW]
					int chw = aix[j];
					
					if(cInput == (chw / HW)) {
						// Get individual zero-based c,h,w indexes from zero-based 'chw'
						int hInput = (chw - cInput*HW)/W;
						int wInput = chw % W; 
						
						appendInputValueToIm2colOutput(output, cInput, hInput, wInput, avals[j], 
								R, S, P, Q, stride_h, stride_w, pad_h, pad_w, meta);
					}
				}
				if(meta.sortRows)
					output.sortSparseRows();
				output.setNonZeros(meta.nnz);
			}
			else {
				output.reset(output.getNumRows(), output.getNumColumns(), true, 0);
				output.setNonZeros(0);
			}
		}
		
	}

}
