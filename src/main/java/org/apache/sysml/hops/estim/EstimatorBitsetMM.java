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
import java.util.stream.IntStream;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysml.runtime.matrix.data.DenseBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.SparseBlock;

/**
 * This estimator implements a naive but rather common approach of boolean matrix
 * multiplies which allows to infer the exact non-zero structure and thus is
 * also useful for sparse result preallocation.
 * 
 * For example, the following paper indicates that this approach is used for sparse
 * spGEMM in NVIDIA cuSPARSE and Intel MKL:
 * Weifeng Liu and Brian Vinter. An Efficient GPU General Sparse Matrix-Matrix
 * Multiplication for Irregular Data. In IPDPS, pages 370–381, 2014.
 * 
 */
public class EstimatorBitsetMM extends SparsityEstimator {
	@Override
	public double estim(MMNode root) {
		// recursive density map computation of non-leaf nodes
		if (!root.getLeft().isLeaf())
			estim(root.getLeft()); // obtain synopsis
		if (!root.getRight().isLeaf())
			estim(root.getLeft()); // obtain synopsis
		BitsetMatrix m1Map = !root.getLeft().isLeaf() ? (BitsetMatrix) root.getLeft().getSynopsis()
				: new BitsetMatrix(root.getLeft().getData());
		BitsetMatrix m2Map = !root.getRight().isLeaf() ? (BitsetMatrix) root.getRight().getSynopsis()
				: new BitsetMatrix(root.getRight().getData());

		// estimate output density map and sparsity via boolean matrix mult
		BitsetMatrix outMap = m1Map.matMult(m2Map);
		root.setSynopsis(outMap); // memoize boolean matrix
		return OptimizerUtils.getSparsity(outMap.getNumRows(), outMap.getNumColumns(), outMap.getNonZeros());
	}

	@Override
	public double estim(MatrixBlock m1, MatrixBlock m2) {
		BitsetMatrix m1Map = new BitsetMatrix(m1);
		BitsetMatrix m2Map = (m1 == m2) ? //self product
			m1Map : new BitsetMatrix(m2);
		BitsetMatrix outMap = m1Map.matMult(m2Map);
		return OptimizerUtils.getSparsity( // aggregate output histogram
				outMap.getNumRows(), outMap.getNumColumns(), outMap.getNonZeros());
	}
	
	@Override
	public double estim(MatrixBlock m1, MatrixBlock m2, OpCode op) {
		throw new NotImplementedException();
	}
	
	@Override
	public double estim(MatrixBlock m, OpCode op) {
		throw new NotImplementedException();
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
			if (in.isEmptyBlock(false))
				return;
			if( MULTI_THREADED_BUILD && in.getNonZeros() > MIN_PAR_THRESHOLD ) {
				int k = 8 * InfrastructureAnalyzer.getLocalParallelism();
				int blklen = (int)Math.ceil((double)_rlen/k);
				IntStream.range(0, k).parallel().forEach(i ->
					buildIntern(this, in, i*blklen, Math.min((i+1)*blklen, _rlen)));
			}
			else {
				//single-threaded bitset construction
				buildIntern(this, in, 0, in.getNumRows());
			}
			_nonZeros = in.getNonZeros();
		}

		public BitsetMatrix matMult(BitsetMatrix m2) {
			BitsetMatrix out = new BitsetMatrix(_rlen, m2._clen);
			if( this.getNonZeros() == 0 || m2.getNonZeros() == 0 )
				return out;
			long size = (long)_rlen*_clen+(long)m2._rlen*m2._clen;
			if( MULTI_THREADED_ESTIM && size > MIN_PAR_THRESHOLD ) {
				int k = 8 * InfrastructureAnalyzer.getLocalParallelism();
				int blklen = (int)Math.ceil((double)_rlen/k);
				out._nonZeros = IntStream.range(0, k).parallel().mapToLong(i ->
					matMultIntern(this, m2, out, i*blklen, Math.min((i+1)*blklen, _rlen))).sum();
			}
			else {
				//single-threaded boolean matrix mult
				out._nonZeros = matMultIntern(this, m2, out, 0, _rlen);
			}
			return out;
		}
		
		private static void buildIntern(BitsetMatrix bitset, MatrixBlock in, int rl, int ru) {
			final int clen = in.getNumColumns();
			if (in.isInSparseFormat()) {
				SparseBlock sblock = in.getSparseBlock();
				for (int i = rl; i < ru; i++) {
					if (sblock.isEmpty(i))
						continue;
					BitSet lbs = bitset._data[i] = new BitSet(clen);
					int alen = sblock.size(i);
					int apos = sblock.pos(i);
					int[] aix = sblock.indexes(i);
					for (int k = apos; k < apos + alen; k++)
						lbs.set(aix[k]);
				}
			} else {
				DenseBlock dblock = in.getDenseBlock();
				for (int i = 0; i < in.getNumRows(); i++) {
					BitSet lbs = bitset._data[i] = new BitSet(clen);
					double[] avals = dblock.values(i);
					int aix = dblock.pos(i);
					for (int j = 0; j < in.getNumColumns(); j++)
						if (avals[aix + j] != 0)
							lbs.set(j);
				}
			}
		}
		
		private static long matMultIntern(BitsetMatrix bsa, BitsetMatrix bsb, BitsetMatrix bsc, int rl, int ru) {
			final int cd = bsa._clen;
			final int n = bsb._clen;
			// matrix multiply with IKJ schedule and pure OR ops in inner loop
			long lnnz = 0;
			for (int i = rl; i < ru; i++) {
				BitSet a = bsa._data[i];
				if( a != null ) {
					BitSet c = bsc._data[i] = new BitSet(n);
					for (int k = 0; k < cd; k++) {
						BitSet b = bsb._data[k];
						if (a.get(k) && b != null)
							c.or(b);
					}
					// maintain nnz
					lnnz += c.cardinality();
				}
			}
			return lnnz;
		}
	}
}
