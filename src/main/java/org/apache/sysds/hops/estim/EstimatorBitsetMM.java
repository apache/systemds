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

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.hops.HopsException;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;

import java.util.BitSet;
import java.util.stream.IntStream;

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
public class EstimatorBitsetMM extends SparsityEstimator
{
	private final int _type;
	
	public EstimatorBitsetMM() {
		this(-1);
	}
	
	public EstimatorBitsetMM(int type) {
		_type = type;
	}
	
	@Override
	public DataCharacteristics estim(MMNode root) {
		BitsetMatrix m1Map = getCachedSynopsis(root.getLeft());
		BitsetMatrix m2Map = getCachedSynopsis(root.getRight());
		
		BitsetMatrix outMap = estimInternal(m1Map, m2Map, root.getOp());
		root.setSynopsis(outMap); // memorize boolean matrix
		return root.setDataCharacteristics(new MatrixCharacteristics(
			outMap.getNumRows(), outMap.getNumColumns(), outMap.getNonZeros()));
	}

	@Override
	public double estim(MatrixBlock m1, MatrixBlock m2) {
		return estim(m1, m2, OpCode.MM);
	}
	
	@Override
	public double estim(MatrixBlock m1, MatrixBlock m2, OpCode op) {
		if( isExactMetadataOp(op) )
			return estimExactMetaData(m1.getDataCharacteristics(),
				m2.getDataCharacteristics(), op).getSparsity();
		BitsetMatrix m1Map = createBitset(m1);
		BitsetMatrix m2Map = (m1 == m2) ? //self product
			m1Map : createBitset(m2);
		BitsetMatrix outMap = estimInternal(m1Map, m2Map, op);
		return OptimizerUtils.getSparsity(outMap.getNumRows(),
			outMap.getNumColumns(), outMap.getNonZeros());
	}
	
	@Override
	public double estim(MatrixBlock m, OpCode op) {
		if( isExactMetadataOp(op) )
			return estimExactMetaData(m.getDataCharacteristics(), null, op).getSparsity();
		BitsetMatrix m1Map = createBitset(m);
		BitsetMatrix outMap = estimInternal(m1Map, null, op);
		return OptimizerUtils.getSparsity(outMap.getNumRows(),
			outMap.getNumColumns(), outMap.getNonZeros());
	}
	
	private BitsetMatrix getCachedSynopsis(MMNode node) {
		if( node == null )
			return null;
		//ensure synopsis is properly cached and reused
		if( node.isLeaf() && node.getSynopsis() == null )
			node.setSynopsis(createBitset(node.getData()));
		else if( !node.isLeaf() )
			estim(node); //recursively obtain synopsis
		return (BitsetMatrix) node.getSynopsis();
	}
	
	private static BitsetMatrix estimInternal(BitsetMatrix m1Map, BitsetMatrix m2Map, OpCode op) {
		switch(op) {
			case MM:      return m1Map.matMult(m2Map);
			case MULT:    return m1Map.and(m2Map);
			case PLUS:    return m1Map.or(m2Map);
			case RBIND:   return m1Map.rbind(m2Map);
			case CBIND:   return m1Map.cbind(m2Map);
			case NEQZERO: return m1Map;
			case EQZERO:  return m1Map.flip();
			case TRANS:   return m1Map.transpose();
			
			//TODO implement all as bitset operations in both BitsetMatrix1 and BitsetMatrix2
			case DIAG:
			case RESHAPE:
			default:
				throw new NotImplementedException();
		}
	}

	public abstract static class BitsetMatrix {
		protected final int _rlen;
		protected final int _clen;
		protected long _nonZeros;
	
		public BitsetMatrix(int rlen, int clen) {
			_rlen = rlen;
			_clen = clen;
			_nonZeros = 0;
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
		
		public abstract boolean get(int r, int c);
		
		public abstract void set(int r, int c);
		
		protected void init(MatrixBlock in) {
			if (in.isEmptyBlock(false))
				return;
			if( MULTI_THREADED_BUILD && in.getNonZeros() > MIN_PAR_THRESHOLD ) {
				int k = 4 * InfrastructureAnalyzer.getLocalParallelism();
				int blklen = (int)Math.ceil((double)_rlen/k);
				IntStream.range(0, k).parallel().forEach(i ->
					buildIntern(in, i*blklen, Math.min((i+1)*blklen, _rlen)));
			}
			else {
				//single-threaded bitset construction
				buildIntern(in, 0, in.getNumRows());
			}
			_nonZeros = in.getNonZeros();
		}

		public BitsetMatrix matMult(BitsetMatrix m2) {
			BitsetMatrix out = createBitSetMatrix(_rlen, m2._clen);
			if( this.getNonZeros() == 0 || m2.getNonZeros() == 0 )
				return out;
			long size = (long)_rlen*_clen+(long)m2._rlen*m2._clen;
			if( MULTI_THREADED_ESTIM && size > MIN_PAR_THRESHOLD ) {
				int k = 4 * InfrastructureAnalyzer.getLocalParallelism();
				int blklen = (int)Math.ceil((double)_rlen/k);
				out._nonZeros = IntStream.range(0, k).parallel().mapToLong(i ->
					matMultIntern(m2, out, i*blklen, Math.min((i+1)*blklen, _rlen))).sum();
			}
			else {
				//single-threaded boolean matrix mult
				out._nonZeros = matMultIntern(m2, out, 0, _rlen);
			}
			return out;
		}
		
		protected abstract BitsetMatrix createBitSetMatrix(int rlen, int clen);
		
		protected abstract void buildIntern(MatrixBlock in, int rl, int ru);
		
		protected abstract long matMultIntern(BitsetMatrix bsb, BitsetMatrix bsc, int rl, int ru);
		
		protected abstract BitsetMatrix and(BitsetMatrix bsb);
		
		protected abstract BitsetMatrix or(BitsetMatrix bsb);
		
		protected abstract BitsetMatrix rbind(BitsetMatrix bsb);
		
		protected abstract BitsetMatrix cbind(BitsetMatrix bsb);
		
		protected abstract BitsetMatrix flip();
		
		public BitsetMatrix transpose() {
			BitsetMatrix1 ret = new BitsetMatrix1(getNumRows(), getNumColumns());
			for(int i=0; i<getNumColumns(); i++)
				for(int k=0; k<getNumRows(); k++)
					if(get(i,k))
						ret.set(k, i);
			return ret;
		}
		
		//protected abstract BitsetMatrix diag();
		
		//protected abstract BitsetMatrix reshape(int rows, int cols, boolean byrow);
	}
	
	public BitsetMatrix createBitset(int m, int n) {
		return ((long)m*n < Integer.MAX_VALUE && _type != 2) ?
			new BitsetMatrix1(m, n) : //linearized long array
			new BitsetMatrix2(m, n);  //bitset per row
	}
	
	public BitsetMatrix createBitset(MatrixBlock in) {
		return (in.getLength() < Integer.MAX_VALUE && _type != 2) ?
			new BitsetMatrix1(in) : //linearized long array
			new BitsetMatrix2(in);  //bitset per row
	}
	
	/**
	 * This class represents a boolean matrix and provides key operations.
	 * In the interest of a cache-conscious matrix multiplication and reduced
	 * memory overhead, we use a linearized and padded array of longs instead
	 * of Java's BitSet per row (which causes memory overheads per row and does
	 * not allow for range ORs). However, this implies a maximum size of 16GB.
	 * 
	 */
	public static class BitsetMatrix1 extends BitsetMatrix {
		//linearized and padded data array in row-major order, where each long
		//represents 64 boolean values, all rows are aligned at 64 for simple access
		private final int _rowLen;
		private final long[] _data;

		public BitsetMatrix1(int rlen, int clen) {
			super(rlen, clen);
			_rowLen = (int) Math.ceil((double)clen / 64);
			_data = new long[rlen * _rowLen];
		}
		
		public BitsetMatrix1(MatrixBlock in) {
			this(in.getNumRows(), in.getNumColumns());
			init(in);
		}
		
		@Override
		protected BitsetMatrix createBitSetMatrix(int rlen, int clen) {
			return new BitsetMatrix1(rlen, clen);
		}
		
		@Override
		protected void buildIntern(MatrixBlock in, int rl, int ru) {
			if (in.isInSparseFormat()) {
				SparseBlock sblock = in.getSparseBlock();
				for (int i = rl; i < ru; i++) {
					if (sblock.isEmpty(i))
						continue;
					int alen = sblock.size(i);
					int apos = sblock.pos(i);
					int[] aix = sblock.indexes(i);
					for (int k = apos; k < apos + alen; k++)
						set(i, aix[k]);
				}
			} else {
				DenseBlock dblock = in.getDenseBlock();
				for (int i = rl; i < ru; i++) {
					double[] avals = dblock.values(i);
					int aix = dblock.pos(i);
					for (int j = 0; j < in.getNumColumns(); j++)
						if (avals[aix + j] != 0)
							set(i, j);
				}
			}
		}
		
		@Override
		protected long matMultIntern(BitsetMatrix bsb2, BitsetMatrix bsc2, int rl, int ru) {
			BitsetMatrix1 bsb = (BitsetMatrix1) bsb2;
			BitsetMatrix1 bsc = (BitsetMatrix1) bsc2;
			final long[] b = bsb._data;
			final long[] c = bsc._data;
			final int cd = _clen;
			final int n = bsb._clen;
			final int n64 = bsb._rowLen;
			
			final int blocksizeI = 32;
			final int blocksizeK = 24;
			final int blocksizeJ = 1024 * 64;
			
			long lnnz = 0;
			
			//blocked execution (cache-conscious)
			for( int bi = rl; bi < ru; bi+=blocksizeI ) {
				int bimin = Math.min(ru, bi+blocksizeI);
				for( int bk = 0; bk < cd; bk+=blocksizeK ) {
					int bkmin = Math.min(cd, bk+blocksizeK);
					for( int bj = 0; bj < n; bj+=blocksizeJ ) {
						//core sub block matrix multiplication
						int bjlen64 = (int)Math.ceil((double)(Math.min(n, bj+blocksizeJ)-bj)/64);
						int bj64 = bj/64;
						for( int i = bi, off=i*_rowLen; i < bimin; i++, off+=_rowLen) {
							for( int k = bk; k < bkmin; k++ ) {
								if( getCol(off, k) ) //implicit and
									or(b, c, k*n64+bj64, i*n64+bj64, bjlen64);
							}
						}
					}
				}
				// maintain nnz for entire output row block
				lnnz += card(c, bi*n64, (bimin-bi)*n64);
			}
			
			return lnnz;
		}
		
		@Override 
		public BitsetMatrix and(BitsetMatrix bsb) {
			if( !(bsb instanceof BitsetMatrix1) )
				throw new HopsException("Incompatible bitset types: "
					+ getClass().getSimpleName()+" and "+bsb.getClass().getSimpleName());
			BitsetMatrix1 b = (BitsetMatrix1) bsb;
			BitsetMatrix1 ret = new BitsetMatrix1(getNumRows(), getNumColumns());
			for(int i=0; i<_data.length; i++)
				ret._data[i] = _data[i] & b._data[i];
			ret._nonZeros = card(ret._data, 0, ret._data.length);
			return ret;
		}
		
		@Override 
		public BitsetMatrix or(BitsetMatrix bsb) {
			if( !(bsb instanceof BitsetMatrix1) )
				throw new HopsException("Incompatible bitset types: "
					+ getClass().getSimpleName()+" and "+bsb.getClass().getSimpleName());
			BitsetMatrix1 b = (BitsetMatrix1) bsb;
			BitsetMatrix1 ret = new BitsetMatrix1(getNumRows(), getNumColumns());
			for(int i=0; i<_data.length; i++)
				ret._data[i] = _data[i] | b._data[i];
			ret._nonZeros = card(ret._data, 0, ret._data.length);
			return ret;
		}
		
		@Override 
		public BitsetMatrix rbind(BitsetMatrix bsb) {
			if( !(bsb instanceof BitsetMatrix1) )
				throw new HopsException("Incompatible bitset types: "
					+ getClass().getSimpleName()+" and "+bsb.getClass().getSimpleName());
			BitsetMatrix1 b = (BitsetMatrix1) bsb;
			BitsetMatrix1 ret = new BitsetMatrix1(getNumRows()+bsb.getNumRows(), getNumColumns());
			System.arraycopy(_data, 0, ret._data, 0, _rlen*_rowLen);
			System.arraycopy(b._data, 0, ret._data, _rlen*_rowLen, b._rlen*_rowLen);
			ret._nonZeros = card(ret._data, 0, ret._data.length);
			return ret;
		}
		
		@Override 
		public BitsetMatrix cbind(BitsetMatrix bsb) {
			if( !(bsb instanceof BitsetMatrix1) )
				throw new HopsException("Incompatible bitset types: "
					+ getClass().getSimpleName()+" and "+bsb.getClass().getSimpleName());
			BitsetMatrix1 b = (BitsetMatrix1) bsb;
			BitsetMatrix1 ret = new BitsetMatrix1(getNumRows(), getNumColumns() + bsb.getNumColumns());
			//copy first matrix 
			for(int i=0; i<getNumRows(); i++)
				System.arraycopy(_data, i*_rowLen, ret._data, i*ret._rowLen, _rowLen);
			//copy second matrix
			for(int i=0; i<getNumRows(); i++)
				for(int j=0; j<b.getNumColumns(); j++)
					if( b.get(i, j) )
						ret.set(i, getNumColumns()+j);
			ret._nonZeros = card(ret._data, 0, ret._data.length);
			return ret;
		}
		
		@Override 
		public BitsetMatrix flip() {
			BitsetMatrix1 ret = new BitsetMatrix1(getNumRows(), getNumColumns());
			for(int i=0; i<_data.length; i++)
				ret._data[i] = ~_data[i];
			ret._nonZeros = (long)getNumRows() * getNumColumns() - getNonZeros();
			return ret;
		}
		
		@Override
		public void set(int r, int c) {
			int off = r * _rowLen;
			int wordIndex = wordIndex(c); //see BitSet.java
			_data[off + wordIndex] |= (1L << c); //see BitSet.java
		}
		
		@Override
		public boolean get(int r, int c) {
			int off = r * _rowLen;
			int wordIndex = wordIndex(c); //see Bitset.java
			return (_data[off + wordIndex] & (1L << c)) != 0; //see BitSet.java
		}
		
		private boolean getCol(int off, int c) {
			int wordIndex = wordIndex(c); //see Bitset.java
			return (_data[off + wordIndex] & (1L << c)) != 0; //see BitSet.java
		}
		
		private static int wordIndex(int bitIndex) {
			return bitIndex >> 6; //see BitSet.java
		}
		
		private static int card(long[] c, int ci, int len) {
			int sum = 0;
			for( int i = ci; i < ci+len; i++ )
				sum += Long.bitCount(c[i]);
			return sum;
		}
		
		private static void or(long[] b, long[] c, int bi, int ci, int len) {
			final int bn = len%8;
			//compute rest
			for( int i = 0; i < bn; i++, bi++, ci++ )
				c[ci] |= b[bi];
			//unrolled 8-block (for better instruction-level parallelism)
			for( int i = bn; i < len; i+=8, bi+=8, ci+=8 ) {
				c[ci+0] |= b[bi+0]; c[ci+1] |= b[bi+1];
				c[ci+2] |= b[bi+2]; c[ci+3] |= b[bi+3];
				c[ci+4] |= b[bi+4]; c[ci+5] |= b[bi+5];
				c[ci+6] |= b[bi+6]; c[ci+7] |= b[bi+7];
			}
		}
	}
	
	public static class BitsetMatrix2 extends BitsetMatrix {
		private BitSet[] _data;

		public BitsetMatrix2(int rlen, int clen) {
			super(rlen, clen);
			_data = new BitSet[_rlen];
		}

		public BitsetMatrix2(MatrixBlock in) {
			this(in.getNumRows(), in.getNumColumns());
			init(in);
		}
		
		@Override
		protected BitsetMatrix createBitSetMatrix(int rlen, int clen) {
			return new BitsetMatrix2(rlen, clen);
		}
		
		@Override
		protected void buildIntern(MatrixBlock in, int rl, int ru) {
			final int clen = in.getNumColumns();
			if (in.isInSparseFormat()) {
				SparseBlock sblock = in.getSparseBlock();
				for (int i = rl; i < ru; i++) {
					if (sblock.isEmpty(i))
						continue;
					BitSet lbs = _data[i] = new BitSet(clen);
					int alen = sblock.size(i);
					int apos = sblock.pos(i);
					int[] aix = sblock.indexes(i);
					for (int k = apos; k < apos + alen; k++)
						lbs.set(aix[k]);
				}
			} else {
				DenseBlock dblock = in.getDenseBlock();
				for (int i = rl; i < ru; i++) {
					BitSet lbs = _data[i] = new BitSet(clen);
					double[] avals = dblock.values(i);
					int aix = dblock.pos(i);
					for (int j = 0; j < in.getNumColumns(); j++)
						if (avals[aix + j] != 0)
							lbs.set(j);
				}
			}
		}
		
		@Override
		protected long matMultIntern(BitsetMatrix bsb2, BitsetMatrix bsc2, int rl, int ru) {
			BitsetMatrix2 bsb = (BitsetMatrix2) bsb2;
			BitsetMatrix2 bsc = (BitsetMatrix2) bsc2;
			final int cd = _clen;
			final int n = bsb._clen;
			// matrix multiply with IKJ schedule and pure OR ops in inner loop
			long lnnz = 0;
			for (int i = rl; i < ru; i++) {
				BitSet a = _data[i];
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
		
		@Override 
		public BitsetMatrix and(BitsetMatrix bsb) {
			if( !(bsb instanceof BitsetMatrix2) )
				throw new HopsException("Incompatible bitset types: "
					+ getClass().getSimpleName()+" and "+bsb.getClass().getSimpleName());
			BitsetMatrix2 b = (BitsetMatrix2) bsb;
			BitsetMatrix2 ret = new BitsetMatrix2(getNumRows(), getNumColumns());
			for(int i=0; i<_data.length; i++) {
				ret._data[i] = (BitSet)_data[i].clone();
				ret._data[i].and(b._data[i]);
				ret._nonZeros += ret._data[i].cardinality();
			}
			return ret;
		}
		
		@Override 
		public BitsetMatrix or(BitsetMatrix bsb) {
			if( !(bsb instanceof BitsetMatrix2) )
				throw new HopsException("Incompatible bitset types: "
					+ getClass().getSimpleName()+" and "+bsb.getClass().getSimpleName());
			BitsetMatrix2 b = (BitsetMatrix2) bsb;
			BitsetMatrix2 ret = new BitsetMatrix2(getNumRows(), getNumColumns());
			for(int i=0; i<_data.length; i++) {
				ret._data[i] = (BitSet)_data[i].clone();
				ret._data[i].or(b._data[i]);
				ret._nonZeros += ret._data[i].cardinality();
			}
			return ret;
		}
		
		@Override 
		public BitsetMatrix rbind(BitsetMatrix bsb) {
			if( !(bsb instanceof BitsetMatrix2) )
				throw new HopsException("Incompatible bitset types: "
					+ getClass().getSimpleName()+" and "+bsb.getClass().getSimpleName());
			BitsetMatrix2 b = (BitsetMatrix2) bsb;
			BitsetMatrix2 ret = new BitsetMatrix2(getNumRows()+bsb.getNumRows(), getNumColumns());
			System.arraycopy(_data, 0, ret._data, 0, _rlen); //shallow copy
			System.arraycopy(b._data, 0, ret._data, _rlen, b._rlen); //shallow copy
			return ret;
		}
		
		@Override
		protected BitsetMatrix cbind(BitsetMatrix bsb) {
			if( !(bsb instanceof BitsetMatrix2) )
				throw new HopsException("Incompatible bitset types: "
					+ getClass().getSimpleName()+" and "+bsb.getClass().getSimpleName());
			BitsetMatrix2 b = (BitsetMatrix2) bsb;
			BitsetMatrix2 ret = new BitsetMatrix2(getNumRows(), getNumColumns() + bsb.getNumColumns());
			//copy first matrix 
			for(int i=0; i<getNumRows(); i++)
				ret._data[i] = (BitSet)_data[i].clone();
			//copy second matrix (via append)
			for(int i=0; i<getNumRows(); i++) {
				for(int j=0; j<b.getNumColumns(); j++)
					if( b.get(i, j) )
						ret.set(i, getNumColumns()+j);
				ret._nonZeros += ret._data[i].cardinality();
			}
			return ret;
		}
		
		@Override 
		public BitsetMatrix flip() {
			BitsetMatrix2 ret = new BitsetMatrix2(getNumRows(), getNumColumns());
			for(int i=0; i<_data.length; i++) {
				ret._data[i] = (BitSet)_data[i].clone();
				ret._data[i].flip(0, _data[i].size());
				ret._nonZeros += ret._data[i].cardinality();
			}
			return ret;
		}
		
		@Override
		public boolean get(int r, int c) {
			return _data[r].get(c);
		}
		
		@Override
		public void set(int r, int c) {
			_data[r].set(c);
		}
	}
}
