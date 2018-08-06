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

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.DenseBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.SparseBlock;
import org.apache.sysml.runtime.util.UtilFunctions;

/**
 * This estimator implements an approach called density maps, as introduced in
 * David Kernert, Frank Köhler, Wolfgang Lehner: SpMacho - Optimizing Sparse 
 * Linear Algebra Expressions with Probabilistic Density Estimation. EDBT 2015: 289-300.
 * 
 * The basic idea is to maintain a density matrix per input, i.e., non-zero counts per 
 * bxb cells of the matrix, and compute the output density map based on a matrix-multiply-like
 * aggregation of average estimates per cell. The output sparsity is then derived as an
 * aggregate of the output density map, but this density map can also be used for 
 * intermediates in chains of matrix multiplications.
 */
public class EstimatorDensityMap extends SparsityEstimator
{
	private static final int BLOCK_SIZE = 256;
	
	private final int _b;
	
	public EstimatorDensityMap() {
		this(BLOCK_SIZE);
	}
	
	public EstimatorDensityMap(int blocksize) {
		_b = blocksize;
	}
	
	@Override
	public MatrixCharacteristics estim(MMNode root) {
		//recursive density map computation of non-leaf nodes
		if( !root.getLeft().isLeaf() )
			estim(root.getLeft()); //obtain synopsis
		if( !root.getRight().isLeaf() )
			estim(root.getLeft()); //obtain synopsis
		MatrixBlock m1Map = !root.getLeft().isLeaf() ?
			(MatrixBlock)root.getLeft().getSynopsis() : computeDensityMap(root.getLeft().getData());
		MatrixBlock m2Map = !root.getRight().isLeaf() ?
			(MatrixBlock)root.getRight().getSynopsis() : computeDensityMap(root.getRight().getData());
		
		//estimate output density map and sparsity
		MatrixBlock outMap = estimIntern(m1Map, m2Map,
			false, root.getRows(), root.getLeft().getCols(), root.getCols());
		root.setSynopsis(outMap); //memoize density map
		return root.setMatrixCharacteristics(new MatrixCharacteristics(
			root.getLeft().getRows(), root.getRight().getCols(), (long)outMap.sum()));
	}

	@Override
	public double estim(MatrixBlock m1, MatrixBlock m2) {
		MatrixBlock m1Map = computeDensityMap(m1);
		MatrixBlock m2Map = (m1 == m2) ? //self product
			m1Map : computeDensityMap(m2);
		MatrixBlock outMap = estimIntern(m1Map, m2Map,
			true, m1.getNumRows(), m1.getNumColumns(), m2.getNumColumns());
		return OptimizerUtils.getSparsity( //aggregate output histogram
			m1.getNumRows(), m2.getNumColumns(), (long)outMap.sum());
	}
	
	@Override
	public double estim(MatrixBlock m1, MatrixBlock m2, OpCode op) {
		throw new NotImplementedException();
	}
	
	@Override
	public double estim(MatrixBlock m, OpCode op) {
		throw new NotImplementedException();
	}

	private MatrixBlock computeDensityMap(MatrixBlock in) {
		int rlen = (int)Math.ceil((double)in.getNumRows()/_b);
		int clen = (int)Math.ceil((double)in.getNumColumns()/_b);
		MatrixBlock out = new MatrixBlock(rlen, clen, false);
		
		//fast-path empty input
		if( in.isEmptyBlock(false) )
			return out;
		
		//allocate dense output block
		DenseBlock c = out.allocateBlock().getDenseBlock();
		
		//fast-path fully dense input
		if( in.getLength() == in.getNonZeros() ) {
			c.set(1); //set sparsity 1.0 into all cells
			out.setNonZeros(in.getLength());
			return out;
		}
		
		//compute nnz histogram
		if( in.isInSparseFormat() ) {
			SparseBlock sblock = in.getSparseBlock();
			for(int i=0; i<in.getNumRows(); i++) {
				if( sblock.isEmpty(i) ) continue;
				int alen = sblock.size(i);
				int apos = sblock.pos(i);
				int[] aix = sblock.indexes(i);
				for(int k=apos; k<apos+alen; k++)
					c.incr(i/_b, aix[k]/_b);
			}
		}
		else {
			for(int i=0; i<in.getNumRows(); i++) {
				for(int j=0; j<in.getNumColumns(); j++) {
					double aval = in.quickGetValue(i, j);
					if( aval != 0 )
						c.incr(i/_b, j/_b);
				}
			}
		}
		
		//scale histogram by block size, w/ awareness of boundary blocks
		for(int i=0; i<rlen; i++){
			int lrlen = UtilFunctions.computeBlockSize(in.getNumRows(), i+1, _b);
			for(int j=0; j<clen; j++) {
				double cval = c.get(i, j);
				if( cval == 0 ) continue;
				int lclen = UtilFunctions.computeBlockSize(in.getNumColumns(), j+1, _b);
				c.set(i, j, cval/lrlen/lclen);
			}
		}
		out.recomputeNonZeros();
		return out;
	}
	
	/**
	 * Computes the output density map given the density maps of the input operands.
	 * 
	 * @param m1Map density map left-hand-side operand
	 * @param m2Map density map right-hand-side operand
	 * @param retNnz return number of non-zeros instead of sparsity per cell
	 * @param mOrig number of rows of output matrix, required for returning nnz
	 * @param cdOrig common dimension of original matrix multiply
	 * @param nOrig number of columns of output matrix, required for returning nnz
	 * @return density map
	 */
	private MatrixBlock estimIntern(MatrixBlock m1Map, MatrixBlock m2Map, boolean retNnz, int mOrig, int cdOrig, int nOrig) {
		final int m = m1Map.getNumRows();
		final int cd = m1Map.getNumColumns();
		final int n = m2Map.getNumColumns();
		MatrixBlock out = new MatrixBlock(m, n, false);
		if( m1Map.isEmptyBlock(false) || m2Map.isEmptyBlock(false) )
			return out;
		
		//compute output density map with IKJ schedule
		DenseBlock c = out.allocateBlock().getDenseBlock();
		for(int i=0; i<m; i++) {
			for(int k=0; k<cd; k++) {
				int lbk = UtilFunctions.computeBlockSize(cdOrig, k+1, _b);
				double sp1 = m1Map.quickGetValue(i, k);
				if( sp1 == 0 ) continue;
				for(int j=0; j<n; j++) {
					double sp2 = m2Map.quickGetValue(k, j);
					if( sp2 == 0 ) continue;
					//custom multiply for scalar sparsity
					double tmp1 = 1 - Math.pow(1-sp1*sp2, lbk);
					//custom add for scalar sparsity
					double tmp2 = c.get(i, j);
					c.set(i, j, tmp1+tmp2 - tmp1*tmp2);
				}
			}
			//scale to non-zeros instead of sparsity if needed
			if( retNnz ) {
				int lbm = UtilFunctions.computeBlockSize(mOrig, i+1, _b);
				for( int j=0; j<n; j++ ) {
					int lbn = UtilFunctions.computeBlockSize(nOrig, j+1, _b);
					c.set(i, j, c.get(i, j) * lbm * lbn);
				}
			}
		}
		out.recomputeNonZeros();
		return out;
	}
}
