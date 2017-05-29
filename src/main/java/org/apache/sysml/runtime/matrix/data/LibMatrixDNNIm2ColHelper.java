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

/**
 * This class contains the different implementation of im2col operation
 */
public class LibMatrixDNNIm2ColHelper {
	
	static interface Im2colWorker {
		public void execute(int n);
		public void execute(int n, int c);
		public static Im2colWorker getWorker(MatrixBlock input, MatrixBlock im2ColOutBlock, ConvolutionParameters params, boolean allChannels) {
			if(im2ColOutBlock.isInSparseFormat() || im2ColOutBlock.getDenseBlock() == null)
				throw new RuntimeException("im2col output is always in dense format");
			if(allChannels) {
				if(!input.isInSparseFormat()) {
					if (params.stride_h == 1 && params.stride_w == 1 && params.pad_h == 0 && params.pad_w == 0) 
						return new DenseIm2colWorkerStride1Pad0AllChannels(input.getDenseBlock(), im2ColOutBlock.getDenseBlock(), params);
					else
						return new DenseIm2colWorkerAllChannels(input.getDenseBlock(), im2ColOutBlock.getDenseBlock(), params);
				}
				else 
					return new SparseIm2colWorkerAllChannels(input, im2ColOutBlock, params);
			}
			else {
				if(!input.isInSparseFormat()) {
					if (params.stride_h == 1 && params.stride_w == 1 && params.pad_h == 0 && params.pad_w == 0) 
						return new DenseIm2colWorkerStride1Pad0(input.getDenseBlock(), im2ColOutBlock.getDenseBlock(), params);
					else
						return new DenseIm2colWorker(input.getDenseBlock(), im2ColOutBlock.getDenseBlock(), params);
				}
				else 
					return new SparseIm2colWorker(input, im2ColOutBlock, params);
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
	 * Performing dense im2col (general case)
	 */
	static class SparseIm2colWorkerAllChannels implements Im2colWorker {
		MatrixBlock input; double [] outputArray; 
		int CRS; int S; int R; int P; int Q; int H; int W; 
		int stride_h; int stride_w; int pad_h; int pad_w; double [] temp;
		public SparseIm2colWorkerAllChannels(MatrixBlock input, MatrixBlock im2ColOutBlock, ConvolutionParameters params) {
			this.input = input;
			this.outputArray = im2ColOutBlock.getDenseBlock();
			this.CRS = params.C * params.R * params.S;
			this.H = params.H; this.W = params.W; this.R = params.R; this.S = params.S; this.P = params.P; this.Q = params.Q;
			this.stride_h = params.stride_h; this.stride_w = params.stride_w;
			this.pad_h = params.pad_h; this.pad_w = params.pad_w;
			temp = new double[input.getNumColumns()];
		}
		
		@Override
		public void execute(int n, int c) {
			throw new RuntimeException("Not supported");
		}

		@Override
		public void execute(int n) {
			// Using a temporary array improves performance by not requiring binary search for getValue
			// Since the access pattern depends on ConvolutionParameters, this serves as a temporary fix.
			fillTemp(input, n);
			// final int nOffset = n * params.C*params.H*params.W;
			for (int c = 0; c < CRS; ++c) {
				int wOffset = c % S;
				int hOffset = (c / S) % R;
				int cInput = c / R / S;
				for (int h = 0; h < P; ++h) {
					int outOffset = (c * P + h) * Q;
					int hPadded = h * stride_h - pad_h + hOffset;
					int tempOffset = (cInput * H + hPadded) * W;
					// int inputOffset = nOffset + tempOffset;
					if (hPadded < 0 || hPadded >= H) {
						Arrays.fill(outputArray, outOffset, outOffset+Q, 0);
					} else {
						for (int w = 0; w < Q; ++w) {
							int wPadded = w * stride_w - pad_w + wOffset;
							if (wPadded >= 0 && wPadded < W) 
								outputArray[outOffset + w] = temp[tempOffset + wPadded];
							else
								outputArray[outOffset + w] = 0;
						}
					}
				}
			}
		}
		// Returns the row of matrix in dense format
		private void fillTemp(MatrixBlock input, int n) {
			if(input.getNumColumns() != temp.length) {
				throw new RuntimeException("Invalid parameters");
			}
			// Use temporary array to avoid binary search
			if(input.isInSparseFormat()) {
				Arrays.fill(temp, 0);
				if( !input.sparseBlock.isEmpty(n) ) {
					int apos = input.sparseBlock.pos(n);
					int alen = input.sparseBlock.size(n);
					int[] aix = input.sparseBlock.indexes(n);
					double[] avals = input.sparseBlock.values(n);
					for(int j=apos; j<apos+alen; j++)
						temp[ aix[j] ] = avals[j];
				}
			}
			else {
				System.arraycopy(input.getDenseBlock(), n*input.getNumColumns(), temp, 0, input.getNumColumns());
			}
		}
	}
	
	/**
	 * Performing dense im2col (general case)
	 */
	static class SparseIm2colWorker implements Im2colWorker {
		MatrixBlock input; double [] outputArray; 
		int CRS; int S; int R; int P; int Q; int H; int W; 
		int stride_h; int stride_w; int pad_h; int pad_w; double [] temp;
		public SparseIm2colWorker(MatrixBlock input, MatrixBlock im2ColOutBlock, ConvolutionParameters params) {
			this.input = input;
			this.outputArray = im2ColOutBlock.getDenseBlock();
			this.CRS = params.C * params.R * params.S;
			this.H = params.H; this.W = params.W; this.R = params.R; this.S = params.S; this.P = params.P; this.Q = params.Q;
			this.stride_h = params.stride_h; this.stride_w = params.stride_w;
			this.pad_h = params.pad_h; this.pad_w = params.pad_w;
			temp = new double[input.getNumColumns()];
		}
		
		@Override
		public void execute(int n) {
			throw new RuntimeException("Not supported");
		}

		@Override
		public void execute(int n, int cInput) {
			// Using a temporary array improves performance by not requiring binary search for getValue
			// Since the access pattern depends on ConvolutionParameters, this serves as a temporary fix.
			fillTemp(input, n); int RS = R*S;
			for (int rs = 0; rs < RS; ++rs) {
				int wOffset = rs % S;
				int hOffset = rs / S;
				for (int h = 0; h < P; ++h) {
					int outOffset = (rs * P + h) * Q;
					int hPadded = h * stride_h - pad_h + hOffset;
					int tempOffset = (cInput * H + hPadded) * W;
					// int inputOffset = nOffset + tempOffset;
					if (hPadded < 0 || hPadded >= H) {
						Arrays.fill(outputArray, outOffset, outOffset+Q, 0);
					} else {
						for (int w = 0; w < Q; ++w) {
							int wPadded = w * stride_w - pad_w + wOffset;
							if (wPadded >= 0 && wPadded < W) 
								outputArray[outOffset + w] = temp[tempOffset + wPadded];
							else
								outputArray[outOffset + w] = 0;
						}
					}
				}
			}
		}
		// Returns the row of matrix in dense format
		private void fillTemp(MatrixBlock input, int n) {
			if(input.getNumColumns() != temp.length) {
				throw new RuntimeException("Invalid parameters");
			}
			// Use temporary array to avoid binary search
			if(input.isInSparseFormat()) {
				Arrays.fill(temp, 0);
				if( !input.sparseBlock.isEmpty(n) ) {
					int apos = input.sparseBlock.pos(n);
					int alen = input.sparseBlock.size(n);
					int[] aix = input.sparseBlock.indexes(n);
					double[] avals = input.sparseBlock.values(n);
					for(int j=apos; j<apos+alen; j++)
						temp[ aix[j] ] = avals[j];
				}
			}
			else {
				System.arraycopy(input.getDenseBlock(), n*input.getNumColumns(), temp, 0, input.getNumColumns());
			}
		}
	}

}
