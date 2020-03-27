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

import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.LibMatrixDNNHelper.CellIndex3;

/**
 * This class contains the different implementation of rotate180 operation
 */
public class LibMatrixDNNRotate180
{
	public static interface Rotate180Worker {
		public void execute(int inputN, int outputN);
		public static Rotate180Worker getWorker(MatrixBlock in, MatrixBlock out,
			DnnParameters params, boolean zeroOutSparseOutput, boolean trans) {
			return in.isInSparseFormat() ?
				new SparseRotate180Worker(in, out, params, trans) :
				new DenseRotate180Worker(in, out, params);
		}
	}
	
	/**
	 * Performing dense rotate180 (general case)
	 */
	private static class DenseRotate180Worker implements Rotate180Worker {
		private final DenseBlock in, out;
		private final DnnParameters params;
		public DenseRotate180Worker(MatrixBlock input, MatrixBlock output, DnnParameters params) {
			this.in = input.getDenseBlock();
			this.out = output.getDenseBlock();
			this.params = params;
		}
		
		@Override
		public void execute(int inputN, int outputN) {
			//note: in (m x KPQ) -> out (m x KPQ)
			double[] avals = in.values(inputN), cvals = out.values(outputN);
			int aix = in.pos(inputN), cix = out.pos(outputN);
			int K = params.K, P = params.P, Q = params.Q;
			for (int k = 0; k < K; k++)
				for (int p = 0; p < P; p++)
					for (int q = 0; q < Q; q++)
						cvals[cix + p*Q*K + q*K + k] = avals[aix + k*P*Q + p*Q + q];
		}
	}
	
	/**
	 * Performing rotate180 when input is sparse (general case)
	 * 
	 * Why are we allocating the output of rotate180 in dense format ? 
	 * Because the number of rows of output (i.e. NPQ) is much larger than number of columns (i.e. K) 
	 */
	private static class SparseRotate180Worker implements Rotate180Worker {
		private final MatrixBlock in, out;
		private final DnnParameters params;
		private final boolean trans;
		
		public SparseRotate180Worker(MatrixBlock input, MatrixBlock output,
			DnnParameters params, boolean trans) {
			this.in = input;
			this.out = output;
			this.params = params;
			this.trans = trans;
		}
		
		@Override
		public void execute(int inputN, int outputN) {
			out.reset();
			
			SparseBlock sblock = in.sparseBlock;
			if( sblock==null || sblock.isEmpty(inputN) )
				return;
			
			CellIndex3 ix = new CellIndex3();
			int outputOffset = outputN*params.P*params.Q;
			int apos = sblock.pos(inputN);
			int alen = sblock.size(inputN);
			int[] aix = sblock.indexes(inputN);
			double[] avals = sblock.values(inputN);
			for(int j = apos; j < apos+alen; j++) {
				ix = LibMatrixDNNHelper.computeTensorIndexes(aix[j], params.P, params.Q, ix);
				if( trans )
					out.appendValue(ix.ix1, outputOffset + ix.ix2*params.Q + ix.ix3, avals[j]);
				else
					out.appendValue(outputOffset + ix.ix2*params.Q + ix.ix3, ix.ix1, avals[j]);
			}
		}
	}
}
