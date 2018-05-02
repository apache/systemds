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

package org.apache.sysml.hops.estim;

import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.DenseBlock;
import org.apache.sysml.runtime.matrix.data.LibMatrixAgg;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.SparseBlock;

/**
 * This estimator implements a remarkably simple yet effective
 * approach for incorporating structural properties into sparsity
 * estimation. The key idea is to maintain row and column nnz per
 * matrix, along with additional meta data.
 */
public class EstimatorMatrixHistogram extends SparsityEstimator
{
	@Override
	public double estim(MMNode root) {
		throw new DMLRuntimeException("Estimation of "
			+ "intermediate matrix histograms not supported yet.");
	}

	@Override
	public double estim(MatrixBlock m1, MatrixBlock m2) {
		MatrixHistogram h1 = new MatrixHistogram(m1);
		MatrixHistogram h2 = new MatrixHistogram(m2);
		return estimIntern(h1, h2);
	}

	@Override
	public double estim(MatrixCharacteristics mc1, MatrixCharacteristics mc2) {
		LOG.warn("Meta-data-only estimates not supported in "
			+ "EstimatorMatrixHistogram, falling back to EstimatorBasicAvg.");
		return new EstimatorBasicAvg().estim(mc1, mc2);
	}
	
	private double estimIntern(MatrixHistogram h1, MatrixHistogram h2) {
		long nnz = 0;
		//special case, with exact sparsity estimate, where the dot product
		//dot(h1.cNnz,h2rNnz) gives the exact number of non-zeros in the output
		if( h1.rMaxNnz <= 1 ) {
			for( int j=0; j<h1.getCols(); j++ )
				nnz += h1.cNnz[j] * h2.rNnz[j];
		}
		//general case with approximate output
		else {
			int mnOut = h1.getRows()*h2.getCols();
			double spOut = 0;
			for( int j=0; j<h1.getCols(); j++ ) {
				double lsp = (double) h1.cNnz[j] * h2.rNnz[j] / mnOut;
				spOut = spOut + lsp - spOut*lsp;
			}
			nnz = (long)(spOut * mnOut);
		}
		
		//compute final sparsity
		return OptimizerUtils.getSparsity(
			h1.getRows(), h2.getCols(), nnz);
	}
	
	private static class MatrixHistogram {
		private final int[] rNnz;
		private final int[] cNnz;
		private int rMaxNnz = 0;
		private int cMaxNnz = 0;
		
		public MatrixHistogram(MatrixBlock in) {
			rNnz = new int[in.getNumRows()];
			cNnz = new int[in.getNumColumns()];
			if( in.isEmptyBlock(false) )
				return;
			
			if( in.isInSparseFormat() ) {
				SparseBlock sblock = in.getSparseBlock();
				for( int i=0; i<in.getNumRows(); i++ ) {
					if( sblock.isEmpty(i) ) continue;
					int alen = sblock.size(i);
					rNnz[i] = alen;
					rMaxNnz = Math.max(rMaxNnz, alen);
					LibMatrixAgg.countAgg(sblock.values(i),
						cNnz, sblock.indexes(i), sblock.pos(i), alen);
				}
			}
			else {
				DenseBlock dblock = in.getDenseBlock();
				for( int i=0; i<in.getNumRows(); i++ ) {
					double[] avals = dblock.values(i);
					int lnnz = 0, aix = dblock.pos(i);
					for( int j=0; j<in.getNumColumns(); j++ ) {
						if( avals[aix+j] != 0 ) {
							cNnz[j] ++;
							lnnz ++;
						}
					}
					rNnz[i] = lnnz;
					rMaxNnz = Math.max(rMaxNnz, lnnz);
				}
			}
			
			for(int j=0; j<in.getNumColumns(); j++)
				cMaxNnz = Math.max(cMaxNnz, cNnz[j]);
		}
		
		public int getRows() {
			return rNnz.length;
		}
		
		public int getCols() {
			return cNnz.length;
		}
	}
}
