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
 * This class contains the different implementation of rotate180 operation
 */
public class LibMatrixDNNRotate180Helper {

	static interface Rotate180Worker {
		public void execute(int inputN, int outputN);
		public static Rotate180Worker getWorker(MatrixBlock input, double [] outputArray, ConvolutionParameters params, boolean zeroOutSparseOutput) {
			if(!input.isInSparseFormat()) 
				return new DenseRotate180Worker(input, outputArray, params);
			else
				return new SparseRotate180Worker(input, outputArray, params, zeroOutSparseOutput);
		}
	}
	
	/**
	 * Performing dense rotate180 (general case)
	 */
	static class DenseRotate180Worker implements Rotate180Worker {

		double [] inputArray; double [] outputArray;  
		ConvolutionParameters params;
		public DenseRotate180Worker(MatrixBlock input, double [] outputArray,  ConvolutionParameters params) {
			this.outputArray = outputArray;
			this.params = params;
			inputArray = input.getDenseBlock();
			if(inputArray == null || outputArray == null)
				throw new RuntimeException("Incorrect usage: empty inputs");
		}
		
		@Override
		public void execute(int inputN, int outputN) {
			int outputOffset = outputN*params.K*params.P*params.Q;
			for (int k = 0; k < params.K; k++) {
				for (int p = 0; p < params.P; p++) {
					for (int q = 0; q < params.Q; q++) {
						outputArray[outputOffset + p*params.Q*params.K + q*params.K + k] = 
								inputArray[inputN*params.K*params.P*params.Q + k*params.P*params.Q + p*params.Q + q];
					}
				}
			}
		}
	}
	
	/**
	 * Performing rotate180 when input is sparse (general case)
	 */
	static class SparseRotate180Worker implements Rotate180Worker {

		double [] outputArray;  MatrixBlock input;
		ConvolutionParameters params; boolean zeroOutSparseOutput;
		public SparseRotate180Worker(MatrixBlock input, double [] outputArray,  ConvolutionParameters params, boolean zeroOutSparseOutput) {
			this.outputArray = outputArray;
			this.params = params;
			this.zeroOutSparseOutput = zeroOutSparseOutput;
			this.input = input;
			if(outputArray == null)
				throw new RuntimeException("Incorrect usage: empty inputs");
		}
		
		@Override
		public void execute(int inputN, int outputN) {
			if(zeroOutSparseOutput)
				Arrays.fill(outputArray, 0);
			
			int outputOffset = outputN*params.K*params.P*params.Q;
			if(!input.isEmptyBlock()) {
				if( !input.sparseBlock.isEmpty(inputN) ) {
					int [] tensorIndexes = new int[3];
					int apos = input.sparseBlock.pos(inputN);
					int alen = input.sparseBlock.size(inputN);
					int[] aix = input.sparseBlock.indexes(inputN);
					double[] avals = input.sparseBlock.values(inputN);
					for(int j = apos; j < apos+alen; j++) {
						LibMatrixDNNHelper.computeTensorIndexes(aix[j], tensorIndexes, params.P, params.Q);
						int k = tensorIndexes[0];
						int p = tensorIndexes[1];
						int q = tensorIndexes[2];
						outputArray[outputOffset + p*params.Q*params.K + q*params.K + k] = avals[j];
					}
				}
			}
		}
	}
}
