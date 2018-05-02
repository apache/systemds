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

import java.util.BitSet;

import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.DenseBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.SparseBlock;

/**
 * This estimator implements naive but rather common approach of boolean matrix
 * multiplies which allows to infer the exact non-zero structure and thus is also
 * useful for sparse result preallocation.
 * 
 */
public class EstimatorBitsetMM extends SparsityEstimator
{
	@Override
	public double estim(MMNode root) {
		//recursive density map computation of non-leaf nodes
		if( !root.getLeft().isLeaf() )
			estim(root.getLeft()); //obtain synopsis
		if( !root.getRight().isLeaf() )
			estim(root.getLeft()); //obtain synopsis
		BitsetMatrix m1Map = !root.getLeft().isLeaf() ?
			(BitsetMatrix)root.getLeft().getSynopsis() : new BitsetMatrix(root.getLeft().getData());
		BitsetMatrix m2Map = !root.getRight().isLeaf() ?
			(BitsetMatrix)root.getRight().getSynopsis() : new BitsetMatrix(root.getRight().getData());
		
		//estimate output density map and sparsity via boolean matrix mult
		BitsetMatrix outMap = m1Map.matMult(m2Map);
		root.setSynopsis(outMap); //memoize boolean matrix
		return OptimizerUtils.getSparsity(
			outMap.getNumRows(), outMap.getNumColumns(), outMap.getNonZeros());
	}

	@Override
	public double estim(MatrixBlock m1, MatrixBlock m2) {
		BitsetMatrix m1Map = new BitsetMatrix(m1);
		BitsetMatrix m2Map = new BitsetMatrix(m2);
		BitsetMatrix outMap = m1Map.matMult(m2Map);
		return OptimizerUtils.getSparsity( //aggregate output histogram
			outMap.getNumRows(), outMap.getNumColumns(), outMap.getNonZeros());
	}

	@Override
	public double estim(MatrixCharacteristics mc1, MatrixCharacteristics mc2) {
		LOG.warn("Meta-data-only estimates not supported in EstimatorBitsetMM, falling back to EstimatorBasicAvg.");
		return new EstimatorBasicAvg().estim(mc1, mc2);
	}
	
	private static class BitsetMatrix {
		private final int _rlen;
		private final int _clen;
		private long _nonZeros;
		private BitSet[] _data;
		
		public BitsetMatrix(int rlen, int clen) {
			_rlen = rlen;
			_clen = clen;
			_data = new BitSet[_rlen];
			for(int i=0; i<_rlen; i++)
				_data[i] = new BitSet(_clen);
			_nonZeros = 0;
		}
		
		public BitsetMatrix(MatrixBlock in) {
			this(in.getNumRows(), in.getNumColumns());
			init(in);
		}
		
		public int getNumRows() {
			return _rlen;
		}
		
		public int getNumColumns() {
			return _clen;
		}
		
		public long getNonZeros() {
			return _nonZeros;
		}
		
		private void init(MatrixBlock in) {
			if( in.isInSparseFormat() ) {
				SparseBlock sblock = in.getSparseBlock();
				for(int i=0; i<in.getNumRows(); i++) {
					if(sblock.isEmpty(i)) continue;
					BitSet lbs = _data[i];
					int alen = sblock.size(i);
					int apos = sblock.pos(i);
					int[] aix = sblock.indexes(i);
					for(int k=apos; k<apos+alen; k++)
						lbs.set(aix[k]);
				}
			}
			else {
				DenseBlock dblock = in.getDenseBlock();
				for(int i=0; i<in.getNumRows(); i++) {
					BitSet lbs = _data[i];
					double[] avals = dblock.values(i);
					int aix = dblock.pos(i);
					for(int j=0; j<in.getNumColumns(); j++)
						if( avals[aix+j] != 0 )
							lbs.set(j);
				}
			}
			_nonZeros = in.getNonZeros();
		}
		
		public BitsetMatrix matMult(BitsetMatrix m2) {
			final int m = this._rlen;
			final int cd = this._clen;
			final int n = m2._clen;
			//matrix multiply with IKJ schedule and pure OR ops in inner loop
			BitsetMatrix out = new BitsetMatrix(m, n);
			for(int i=0; i<m; i++) {
				BitSet a = this._data[i], c = out._data[i];
				for(int k=0; k<cd; k++) {
					if( a.get(k) )
						c.or(m2._data[k]);
				}
				//maintain nnz 
				out._nonZeros += c.cardinality();
			}
			return out;
		}
	}
}
