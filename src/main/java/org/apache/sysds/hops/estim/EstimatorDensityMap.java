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

package org.apache.sysds.hops.estim;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.UtilFunctions;

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
	public DataCharacteristics estim(MMNode root) {
		DensityMap m1Map = getCachedSynopsis(root.getLeft());
		DensityMap m2Map = getCachedSynopsis(root.getRight());
		
		//estimate output density map and sparsity
		DensityMap outMap = estimIntern(m1Map, m2Map, root.getOp());
		root.setSynopsis(outMap); //memoize density map
		return root.setDataCharacteristics(new MatrixCharacteristics(
			outMap.getNumRowsOrig(), outMap.getNumColumnsOrig(), outMap.getNonZeros()));
	}

	@Override
	public double estim(MatrixBlock m1, MatrixBlock m2) {
		return estim(m1, m2, OpCode.MM);
	}
	
	@Override
	public double estim(MatrixBlock m1, MatrixBlock m2, OpCode op) {
		if( isExactMetadataOp(op) )
			return estimExactMetaData(m1.getDataCharacteristics(), m2 != null ?
				m2.getDataCharacteristics() : null, op).getSparsity();
		DensityMap m1Map = new DensityMap(m1, _b);
		DensityMap m2Map = (m1 == m2 || m2 == null) ? //self product
			m1Map : new DensityMap(m2, _b);
		DensityMap outMap = estimIntern(m1Map, m2Map, op);
		return OptimizerUtils.getSparsity( //aggregate output histogram
			outMap.getNumRowsOrig(), outMap.getNumColumnsOrig(), outMap.getNonZeros());
	}
	
	@Override
	public double estim(MatrixBlock m, OpCode op) {
		return estim(m, null, op);
	}
	
	private DensityMap getCachedSynopsis(MMNode node) {
		if( node == null )
			return null;
		//ensure synopsis is properly cached and reused
		if( node.isLeaf() && node.getSynopsis() == null )
			node.setSynopsis(new DensityMap(node.getData(), _b));
		else if( !node.isLeaf() )
			estim(node); //recursively obtain synopsis
		return (DensityMap) node.getSynopsis();
	}
	
	/**
	 * Computes the output density map given the density maps of the input operands.
	 * 
	 * @param m1Map density map left-hand-side operand
	 * @param m2Map density map right-hand-side operand
	 * @param op operator code
	 * @return density map
	 */
	public DensityMap estimIntern(DensityMap m1Map, DensityMap m2Map, OpCode op) {
		switch(op) {
			case MM:      return estimInternMM(m1Map, m2Map);
			case MULT:    return estimInternMult(m1Map, m2Map);
			case PLUS:    return estimInternPlus(m1Map, m2Map);
			case NEQZERO: return m1Map;
			case EQZERO:  return estimInternEqZero(m1Map);
			
			case RBIND:
			case CBIND:
				//TODO simple append not possible due to partial blocks at end of m1Map
			
			case TRANS:   return estimInternTrans(m1Map);
			case DIAG:    return estimInternDiag(m1Map);
			case RESHAPE: return estimInternReshape(m1Map);
			default:
				throw new NotImplementedException();
		}
	}
	
	private DensityMap estimInternMM(DensityMap m1Map, DensityMap m2Map) {
		final int m = m1Map.getNumRows();
		final int cd = m1Map.getNumColumns();
		final int n = m2Map.getNumColumns();
		MatrixBlock out = new MatrixBlock(m1Map.getNumRows(), m2Map.getNumColumns(), false);
		DenseBlock c = out.allocateBlock().getDenseBlock();
		m1Map.toSparsity();
		m2Map.toSparsity();
		for(int i=0; i<m; i++) {
			for(int k=0; k<cd; k++) {
				int lbk = m1Map.getColBlockize(k);
				double sp1 = m1Map.get(i, k);
				if( sp1 == 0 ) continue;
				for(int j=0; j<n; j++) {
					double sp2 = m2Map.get(k, j);
					if( sp2 == 0 ) continue;
					//custom multiply for scalar sparsity
					double tmp1 = 1 - Math.pow(1-sp1*sp2, lbk);
					//custom add for scalar sparsity
					double tmp2 = c.get(i, j);
					c.set(i, j, tmp1+tmp2 - tmp1*tmp2);
				}
			}
		}
		out.recomputeNonZeros();
		return new DensityMap(out, m1Map.getNumRowsOrig(),
			m2Map.getNumColumnsOrig(), _b, true);
	}
	
	private DensityMap estimInternMult(DensityMap m1Map, DensityMap m2Map) {
		MatrixBlock out = new MatrixBlock(m1Map.getNumRows(), m1Map.getNumColumns(), false);
		DenseBlock c = out.allocateBlock().getDenseBlock();
		m1Map.toSparsity();
		m2Map.toSparsity();
		for(int i=0; i<m1Map.getNumRows(); i++)
			for(int j=0; j<m1Map.getNumColumns(); j++)
				c.set(i, j, m1Map.get(i, j) * m2Map.get(i, j));
		out.recomputeNonZeros();
		return new DensityMap(out, m1Map.getNumRowsOrig(),
			m1Map.getNumColumnsOrig(), _b, true);
	}
	
	private DensityMap estimInternPlus(DensityMap m1Map, DensityMap m2Map) {
		MatrixBlock out = new MatrixBlock(m1Map.getNumRows(), m1Map.getNumColumns(), false);
		DenseBlock c = out.allocateBlock().getDenseBlock();
		m1Map.toSparsity();
		m2Map.toSparsity();
		for(int i=0; i<m1Map.getNumRows(); i++)
			for(int j=0; j<m1Map.getNumColumns(); j++) {
				double sp1 = m1Map.get(i, j);
				double sp2 = m2Map.get(i, j);
				c.set(i, j, sp1 + sp2 - sp1 * sp2);
			}
		out.recomputeNonZeros();
		return new DensityMap(out, m1Map.getNumRowsOrig(),
			m1Map.getNumColumnsOrig(), _b, true);
	}
	
	private DensityMap estimInternTrans(DensityMap m1Map) {
		MatrixBlock out = LibMatrixReorg.transpose(m1Map.getMap(), 
			new MatrixBlock(m1Map.getNumColumns(), m1Map.getNumRows(), false));
		return new DensityMap(out, m1Map.getNumColumnsOrig(),
			m1Map.getNumRowsOrig(), _b, m1Map._scaled);
	}
	
	private DensityMap estimInternDiag(DensityMap m1Map) {
		if( m1Map.getNumColumnsOrig() > 1 )
			throw new NotImplementedException();
		m1Map.toNnz();
		MatrixBlock out = LibMatrixReorg.diag(m1Map.getMap(), 
			new MatrixBlock(m1Map.getNumRows(), m1Map.getNumRows(), false));
		return new DensityMap(out, m1Map.getNumRowsOrig(),
			m1Map.getNumRowsOrig(), _b, m1Map._scaled);
	}
	
	private static DensityMap estimInternReshape(DensityMap m1Map) {
		MatrixBlock out = new MatrixBlock(1,1,(double)m1Map.getNonZeros());
		int b = Math.max(m1Map.getNumRowsOrig(), m1Map.getNumColumnsOrig());
		return new DensityMap(out, m1Map.getNumRowsOrig(),
			m1Map.getNumColumnsOrig(), b, false);
	}
	
	private DensityMap estimInternEqZero(DensityMap m1Map) {
		MatrixBlock out = new MatrixBlock(m1Map.getNumRows(), m1Map.getNumColumns(), false);
		m1Map.toSparsity();
		for(int i=0; i<m1Map.getNumRows(); i++)
			for(int j=0; j<m1Map.getNumColumns(); j++)
				out.quickSetValue(i, j, 1-m1Map.get(i, j));
		return new DensityMap(out, m1Map.getNumRowsOrig(),
			m1Map.getNumColumnsOrig(), _b, m1Map._scaled);
	}
	
	public static class DensityMap {
		private final MatrixBlock _map;
		private final int _rlen;
		private final int _clen;
		private final int _b;
		private boolean _scaled; //false->nnz, true->sp
		
		public DensityMap(MatrixBlock in, int b) {
			_rlen = in.getNumRows();
			_clen = in.getNumColumns();
			_b = b;
			_map = init(in);
			_scaled = false;
			if( !isPow2(_b) )
				System.out.println("WARN: Invalid block size: "+_b);
		}
		
		public DensityMap(MatrixBlock map, int rlenOrig, int clenOrig, int b, boolean scaled) {
			_rlen = rlenOrig;
			_clen = clenOrig;
			_b = b;
			_map = map;
			_scaled = scaled;
			if( !isPow2(_b) )
				System.out.println("WARN: Invalid block size: "+_b);
		}
		
		public MatrixBlock getMap() {
			return _map;
		}
		
		public int getNumRows() {
			return _map.getNumRows();
		}
		
		public int getNumColumns() {
			return _map.getNumColumns();
		}
		
		public int getNumRowsOrig() {
			return _rlen;
		}
		
		public int getNumColumnsOrig() {
			return _clen;
		}
		
		public long getNonZeros() {
			if( _scaled ) toNnz();
			return Math.round(_map.sum());
		}
		
		public int getRowBlockize(int r) {
			return UtilFunctions.computeBlockSize(_rlen, r+1, _b);
		}
		
		public int getColBlockize(int c) {
			return UtilFunctions.computeBlockSize(_clen, c+1, _b);
		}
		
		public double get(int r, int c) {
			return _map.quickGetValue(r, c);
		}
		
		public void toSparsity() {
			if( _scaled ) return;
			//scale histogram by block size, w/ awareness of boundary blocks
			int rlen = _map.getNumRows();
			int clen = _map.getNumColumns();
			DenseBlock c = _map.getDenseBlock();
			for(int i=0; i<rlen; i++){
				int lrlen = getRowBlockize(i);
				for(int j=0; j<clen; j++) {
					double cval = c.get(i, j);
					if( cval == 0 ) continue;
					c.set(i, j, cval/lrlen/getColBlockize(j));
				}
			}
			_scaled = true;
		}
		
		public void toNnz() {
			if( !_scaled ) return;
			//scale histogram by block size, w/ awareness of boundary blocks
			int rlen = _map.getNumRows();
			int clen = _map.getNumColumns();
			DenseBlock c = _map.getDenseBlock();
			for(int i=0; i<rlen; i++){
				int lrlen = getRowBlockize(i);
				for(int j=0; j<clen; j++) {
					double cval = c.get(i, j);
					if( cval == 0 ) continue;
					c.set(i, j, cval * lrlen * getColBlockize(j));
				}
			}
			_scaled = false;
		}
		
		private MatrixBlock init(MatrixBlock in) {
			int rlen = (int)Math.ceil((double)_rlen/_b);
			int clen = (int)Math.ceil((double)_clen/_b);
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
				for(int i=0; i<_rlen; i++) {
					for(int j=0; j<_clen; j++) {
						double aval = in.quickGetValue(i, j);
						if( aval != 0 )
							c.incr(i/_b, j/_b);
					}
				}
			}
			out.recomputeNonZeros();
			return out;
		}
		
		private static boolean isPow2(int value) {
			double tmp = (Math.log(value) / Math.log(2));
			return Math.floor(tmp) == Math.ceil(tmp);
		}
	}
}
