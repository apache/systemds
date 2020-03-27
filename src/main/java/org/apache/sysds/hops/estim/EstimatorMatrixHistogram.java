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

import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.LibMatrixAgg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;

import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

/**
 * This estimator implements a remarkably simple yet effective
 * approach for incorporating structural properties into sparsity
 * estimation. The key idea is to maintain row and column nnz per
 * matrix, along with additional meta data.
 */
public class EstimatorMatrixHistogram extends SparsityEstimator
{
	//internal configurations
	private static final boolean DEFAULT_USE_EXTENDED = true;
	private static final boolean ADVANCED_SKETCH_PROP = false;
	
	private final boolean _useExtended;
	
	public EstimatorMatrixHistogram() {
		this(DEFAULT_USE_EXTENDED);
	}
	
	public EstimatorMatrixHistogram(boolean useExtended) {
		_useExtended = useExtended;
	}
	
	@Override
	public DataCharacteristics estim(MMNode root) {
		return estim(root, true);
	}
	
	public DataCharacteristics estim(MMNode root, boolean topLevel) {
		//NOTE: not estimateInputs due to handling of topLevel
		MatrixHistogram h1 = getCachedSynopsis(root.getLeft());
		MatrixHistogram h2 = getCachedSynopsis(root.getRight());
		
		//estimate output sparsity based on input histograms
		double ret = estimIntern(h1, h2, root.getOp(), root.getMisc());
		if( topLevel ) { //fast-path final result
			return MatrixHistogram.deriveOutputCharacteristics(
				h1, h2, ret, root.getOp(), root.getMisc());
		}
		
		//sketch propagation for intermediates other than final result
		if( h2 != null && root.getRight() != null )
			h2.setData(root.getRight().isLeaf() ? root.getRight().getData() : null);
		MatrixHistogram outMap = MatrixHistogram
			.deriveOutputHistogram(h1, h2, ret, root.getOp(), root.getMisc());
		root.setSynopsis(outMap);
		return root.setDataCharacteristics(new MatrixCharacteristics(
			outMap.getRows(), outMap.getCols(), outMap.getNonZeros()));
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
		MatrixHistogram h1 = new MatrixHistogram(m1, _useExtended);
		MatrixHistogram h2 = (m1 == m2) ? //self product
			h1 : new MatrixHistogram(m2, _useExtended);
		return estimIntern(h1, h2, op, null);
	}
	
	@Override
	public double estim(MatrixBlock m1, OpCode op) {
		if( isExactMetadataOp(op) )
			return estimExactMetaData(m1.getDataCharacteristics(), null, op).getSparsity();
		MatrixHistogram h1 = new MatrixHistogram(m1, _useExtended);
		return estimIntern(h1, null, op, null);
	}
	
	private MatrixHistogram getCachedSynopsis(MMNode node) {
		if( node == null )
			return null;
		//ensure synopsis is properly cached and reused
		if( node.isLeaf() && node.getSynopsis() == null )
			node.setSynopsis(new MatrixHistogram(node.getData(), _useExtended));
		else if( !node.isLeaf() )
			estim(node, false); //recursively obtain synopsis
		return (MatrixHistogram) node.getSynopsis();
	}
	
	public double estimIntern(MatrixHistogram h1, MatrixHistogram h2, OpCode op, long[] misc) {
		double msize = (double)h1.getRows()*h1.getCols();
		switch (op) {
			case MM:
				return estimInternMM(h1, h2);
			case MULT: {
				final double scale = IntStream.range(0, h1.getCols())
					.mapToDouble(j -> (double)h1.cNnz[j] * h2.cNnz[j]).sum()
					/ h1.getNonZeros() / h2.getNonZeros();
				return IntStream.range(0, h1.getRows())
					.mapToDouble(i -> (double)h1.rNnz[i] * h2.rNnz[i] * scale) //collisions
					.sum() / msize;
			}
			case PLUS: {
				final double scale = IntStream.range(0, h1.getCols())
					.mapToDouble(j -> (double)h1.cNnz[j] * h2.cNnz[j]).sum()
					/ h1.getNonZeros() / h2.getNonZeros();
				return IntStream.range(0, h1.getRows())
					.mapToDouble(i -> (double)h1.rNnz[i] + h2.rNnz[i] //all minus collisions
						- (double)h1.rNnz[i] * h2.rNnz[i] * scale)
					.sum() / msize;
			}
			case EQZERO:
				return OptimizerUtils.getSparsity(h1.getRows(), h1.getCols(),
					(long)h1.getRows() * h1.getCols() - h1.getNonZeros());
			case DIAG:
				return (h1.getCols()==1) ?
					OptimizerUtils.getSparsity(h1.getRows(), h1.getRows(), h1.getNonZeros()) :
					OptimizerUtils.getSparsity(h1.getRows(), 1, Math.min(h1.getRows(), h1.getNonZeros()));
			//binary operations that preserve sparsity exactly
			case CBIND:
				return OptimizerUtils.getSparsity(h1.getRows(),
					h1.getCols()+h2.getCols(), h1.getNonZeros() + h2.getNonZeros());
			case RBIND:
				return OptimizerUtils.getSparsity(h1.getRows()+h2.getRows(),
					h1.getCols(), h1.getNonZeros() + h2.getNonZeros());
			//unary operation that preserve sparsity exactly
			case NEQZERO:
			case TRANS:
			case RESHAPE:
				return OptimizerUtils.getSparsity(h1.getRows(), h1.getCols(), h1.getNonZeros());
			default:
				throw new NotImplementedException();
		}
	}
	
	private double estimInternMM(MatrixHistogram h1, MatrixHistogram h2) {
		long nnz = 0;
		//special case, with exact sparsity estimate, where the dot product
		//dot(h1.cNnz,h2rNnz) gives the exact number of non-zeros in the output
		if( h1.rMaxNnz <= 1 || h2.cMaxNnz <= 1 ) {
			for( int j=0; j<h1.getCols(); j++ )
				nnz += (long)h1.cNnz[j] * h2.rNnz[j];
		}
		//special case, with hybrid exact and approximate output
		else if(h1.cNnz1e!=null || h2.rNnz1e != null) {
			//NOTE: normally h1.getRows()*h2.getCols() would define mnOut
			//but by leveraging the knowledge of rows/cols w/ <=1 nnz, we account
			//that exact and approximate fractions touch different areas
			long mnOut = _useExtended ?
				(long)(h1.rNonEmpty-h1.rN1) * (h2.cNonEmpty-h2.cN1) :
				(long)(h1.getRows()-h1.rN1) * (h2.getCols()-h2.cN1);
			double spOutRest = 0;
			for( int j=0; j<h1.getCols(); j++ ) {
				//zero for non-existing extension vectors
				int h1c1ej = (h1.cNnz1e != null) ? h1.cNnz1e[j] : 0;
				int h2r1ej = (h2.rNnz1e != null) ? h2.rNnz1e[j] : 0;
				//exact fractions, w/o double counting
				nnz += (long)h1c1ej * h2.rNnz[j];
				nnz += (long)(h1.cNnz[j]-h1c1ej) * h2r1ej;
				//approximate fraction, w/o double counting
				double lsp = (double)(h1.cNnz[j]-h1c1ej) 
					* (h2.rNnz[j]-h2r1ej) / mnOut;
				spOutRest = spOutRest + lsp - spOutRest*lsp;
			}
			nnz += (long)(spOutRest * mnOut);
		}
		//general case with approximate output
		else {
			long mnOut = _useExtended ?
				(long)h1.rNonEmpty * h2.cNonEmpty :
				(long)h1.getRows() * h2.getCols();
			double spOut = 0;
			for( int j=0; j<h1.getCols(); j++ ) {
				double lsp = (double) h1.cNnz[j] * h2.rNnz[j] / mnOut;
				spOut = spOut + lsp - spOut*lsp;
			}
			nnz = (long)(spOut * mnOut);
		}
		
		if( _useExtended ) {
			//exploit lower bound on nnz based on half-full rows/cols
			//note: upper bound applied via modified output sizes
			nnz = (h1.rNdiv2 >= 0 && h2.cNdiv2 >= 0) ?
				Math.max((long)h1.rNdiv2 * h2.cNdiv2, nnz) : nnz;
		}
		
		//compute final sparsity
		return OptimizerUtils.getSparsity(
			h1.getRows(), h2.getCols(), nnz);
	}
	
	public static class MatrixHistogram {
		// count vectors (the histogram)
		private final int[] rNnz;    //nnz per row
		private int[] rNnz1e = null; //nnz per row for cols w/ <= 1 non-zeros
		private final int[] cNnz;    //nnz per col
		private int[] cNnz1e = null; //nnz per col for rows w/ <= 1 non-zeros
		// additional summary statistics
		private final int rMaxNnz, cMaxNnz;     //max nnz per row/row
		private final int rN1, cN1;             //number of rows/cols with nnz=1
		private final int rNonEmpty, cNonEmpty; //number of non-empty rows/cols (w/ empty is nnz=0)
		private final int rNdiv2, cNdiv2;       //number of rows/cols with nnz > #cols/2 and #rows/2
		private boolean fullDiag;               //true if there exists a full diagonal of nonzeros
		private MatrixBlock _data = null; //optional leaf data
		
		public MatrixHistogram(MatrixBlock in, boolean useExcepts) {
			// 1) allocate basic synopsis
			final int m = in.getNumRows();
			final int n = in.getNumColumns();
			rNnz = new int[in.getNumRows()];
			cNnz = new int[in.getNumColumns()];
			fullDiag = in.getNumRows() == in.getNonZeros()
				&& in.getNumRows() == in.getNumColumns();
			
			// 2) compute basic synopsis details
			if( in.getLength() == in.getNonZeros() ) {
				//fully dense: constant row/column counts
				Arrays.fill(rNnz, n);
				Arrays.fill(cNnz, m);
			}
			else if( !in.isEmpty() ) {
				if( in.isInSparseFormat() ) {
					SparseBlock a = in.getSparseBlock();
					for( int i=0; i<m; i++ ) {
						if( a.isEmpty(i) ) continue;
						int apos = a.pos(i);
						int alen = a.size(i);
						int[] aix = a.indexes(i);
						rNnz[i] = alen;
						LibMatrixAgg.countAgg(a.values(i), cNnz, aix, apos, alen);
						fullDiag &= aix[apos] == i;
					}
				}
				else {
					DenseBlock a = in.getDenseBlock();
					for( int i=0; i<m; i++ ) {
						rNnz[i] = a.countNonZeros(i);
						LibMatrixAgg.countAgg(a.values(i), cNnz, a.pos(i), n);
						fullDiag &= (rNnz[i]==1 && n>i && a.get(i, i)!=0);
					}
				}
			}
			
			// 3) compute meta data synopsis
			int[] rSummary = deriveSummaryStatistics(rNnz, getCols());
			int[] cSummary = deriveSummaryStatistics(cNnz, getRows());
			rMaxNnz = rSummary[0]; cMaxNnz = cSummary[0];
			rN1 = rSummary[1]; cN1 = cSummary[1];
			rNonEmpty = rSummary[2]; cNonEmpty = cSummary[2];
			rNdiv2 = rSummary[3]; cNdiv2 = cSummary[3];
			
			// 4) compute exception details if necessary (optional)
			if( useExcepts && !in.isEmpty() && (rMaxNnz > 1 || cMaxNnz > 1) 
				&& in.getLength() != in.getNonZeros() ) { //not fully dense
				rNnz1e = new int[in.getNumRows()];
				cNnz1e = new int[in.getNumColumns()];
				
				if( in.isInSparseFormat() ) {
					SparseBlock a = in.getSparseBlock();
					for( int i=0; i<m; i++ ) {
						if( a.isEmpty(i) ) continue;
						int alen = a.size(i);
						int apos = a.pos(i);
						int[] aix = a.indexes(i);
						for( int k=apos; k<apos+alen; k++ )
							if( cNnz[aix[k]] <= 1 )
								rNnz1e[i]++;
						if( alen == 1 )
							cNnz1e[aix[apos]]++;
					}
				}
				else {
					DenseBlock a = in.getDenseBlock();
					for( int i=0; i<m; i++ ) {
						double[] avals = a.values(i);
						int aix = a.pos(i);
						boolean rNnzlte1 = rNnz[i] <= 1;
						for( int j=0; j<n; j++ ) {
							if( avals[aix + j] != 0 ) {
								if( cNnz[j] <= 1 ) rNnz1e[i]++;
								if( rNnzlte1 ) cNnz1e[j]++;
							}
						}
					}
				}
			}
		}
		
		public MatrixHistogram(int[] r, int[] r1e, int[] c, int[] c1e, int rmax, int cmax) {
			rNnz = r;
			rNnz1e = r1e;
			cNnz = c;
			cNnz1e = c1e;
			rMaxNnz = rmax;
			cMaxNnz = cmax;
			rN1 = cN1 = -1;
			rNdiv2 = cNdiv2 = -1;
			
			//update non-zero rows/cols
			rNonEmpty = (int)Arrays.stream(rNnz).filter(i -> i!=0).count();
			cNonEmpty = (int)Arrays.stream(cNnz).filter(i -> i!=0).count();
		}
		
		public int getRows() {
			return rNnz.length;
		}
		
		public int getCols() {
			return cNnz.length;
		}
		
		public int[] getRowCounts() {
			return rNnz;
		}
		
		public int[] getColCounts() {
			return cNnz;
		}
		
		public long getNonZeros() {
			return getRows() < getCols() ?
				IntStream.range(0, getRows()).mapToLong(i-> rNnz[i]).sum() :
				IntStream.range(0, getCols()).mapToLong(i-> cNnz[i]).sum();
		}
		
		public void setData(MatrixBlock mb) {
			_data = mb;
		}
		
		public static MatrixHistogram deriveOutputHistogram(MatrixHistogram h1, MatrixHistogram h2, double spOut, OpCode op, long[] misc) {
			switch(op) {
				case MM:      return deriveMMHistogram(h1, h2, spOut);
				case MULT:    return deriveMultHistogram(h1, h2);
				case PLUS:    return derivePlusHistogram(h1, h2);
				case RBIND:   return deriveRbindHistogram(h1, h2);
				case CBIND:   return deriveCbindHistogram(h1, h2);
				case NEQZERO: return h1;
				case EQZERO:  return deriveEq0Histogram(h1);
				case DIAG:    return deriveDiagHistogram(h1);
				case TRANS:   return deriveTransHistogram(h1);
				case RESHAPE: return deriveReshapeHistogram(h1, (int)misc[0], (int)misc[1]);
				default:
					throw new NotImplementedException();
			}
		}
		
		public static DataCharacteristics deriveOutputCharacteristics(MatrixHistogram h1, MatrixHistogram h2, double spOut, OpCode op, long[] misc) {
			switch(op) {
				case MM:
					return new MatrixCharacteristics(h1.getRows(), h2.getCols(),
						OptimizerUtils.getNnz(h1.getRows(), h2.getCols(), spOut));
				case MULT:
				case PLUS:
				case NEQZERO:
				case EQZERO:
					return new MatrixCharacteristics(h1.getRows(), h1.getCols(),
						OptimizerUtils.getNnz(h1.getRows(), h1.getCols(), spOut));
				case RBIND:
					return new MatrixCharacteristics(h1.getRows()+h1.getRows(), h1.getCols(),
						OptimizerUtils.getNnz(h1.getRows()+h2.getRows(), h1.getCols(), spOut));
				case CBIND:
					return new MatrixCharacteristics(h1.getRows(), h1.getCols()+h2.getCols(),
						OptimizerUtils.getNnz(h1.getRows(), h1.getCols()+h2.getCols(), spOut));
				case DIAG:
					int ncol = h1.getCols()==1 ? h1.getRows() : 1;
					return new MatrixCharacteristics(h1.getRows(), ncol,
						OptimizerUtils.getNnz(h1.getRows(), ncol, spOut));
				case TRANS:
					return new MatrixCharacteristics(h1.getCols(), h1.getRows(), h1.getNonZeros());
				case RESHAPE:
					return new MatrixCharacteristics((int)misc[0], (int)misc[1],
						OptimizerUtils.getNnz((int)misc[0], (int)misc[1], spOut));
				default:
					throw new NotImplementedException();
			}
		}
		
		@SuppressWarnings("unused")
		private static MatrixHistogram deriveMMHistogram(MatrixHistogram h1, MatrixHistogram h2, double spOut) {
			//exact propagation if lhs or rhs full diag
			if( h1.fullDiag ) return h2;
			if( h2.fullDiag ) return h1;
			
			//get input/output nnz for scaling
			long nnz1 = h1.getNonZeros();
			long nnz2 = h2.getNonZeros();
			double nnzOut = spOut * h1.getRows() * h2.getCols();
			
			//propagate h1.r and h2.c to output via simple scaling
			//(this implies 0s propagate and distribution is preserved)
			int rMaxNnz = 0, cMaxNnz = 0;
			int[] rNnz = new int[h1.getRows()];
			Random rn = new Random();
			for( int i=0; i<h1.getRows(); i++ ) {
				rNnz[i] = probRound(nnzOut/nnz1 * h1.rNnz[i], rn);
				rMaxNnz = Math.max(rMaxNnz, rNnz[i]);
			}
			int[] cNnz = new int[h2.getCols()];
			if( ADVANCED_SKETCH_PROP && h1.rMaxNnz <= 1 
				&& h2._data != null && h2._data.isInSparseFormat() ) {
				SparseBlock rhs = h2._data.getSparseBlock();
				for( int j=0; j<h1.getCols(); j++ ) {
					if( h1.cNnz[j] == 0 || h2.rNnz[j] == 0 )
						continue;
					int apos = rhs.pos(j);
					int alen = rhs.size(j);
					int[] aix = rhs.indexes(j);
					int scale = h1.cNnz[j];
					for(int k=apos; k<apos+alen; k++ )
						cNnz[aix[k]] += scale;
				}
			}
			else {
				for( int i=0; i<h2.getCols(); i++ ) {
					cNnz[i] = probRound(nnzOut/nnz2 * h2.cNnz[i], rn);
					cMaxNnz = Math.max(cMaxNnz, cNnz[i]);
				}
			}
			
			//construct new histogram object
			return new MatrixHistogram(rNnz, null, cNnz, null, rMaxNnz, cMaxNnz);
		}
		
		private static MatrixHistogram deriveMultHistogram(MatrixHistogram h1, MatrixHistogram h2) {
			final double scaler = IntStream.range(0, h1.getCols())
				.mapToDouble(j -> (double)h1.cNnz[j] * h2.cNnz[j])
				.sum() / h1.getNonZeros() / h2.getNonZeros();
			final double scalec = IntStream.range(0, h1.getRows())
				.mapToDouble(j -> (double)h1.rNnz[j] * h2.rNnz[j])
				.sum() / h1.getNonZeros() / h2.getNonZeros();
			int rMaxNnz = 0, cMaxNnz = 0;
			Random rn = new Random();
			int[] rNnz = new int[h1.getRows()];
			for(int i=0; i<h1.getRows(); i++) {
				rNnz[i] = probRound((double)h1.rNnz[i] * h2.rNnz[i] * scaler, rn);
				rMaxNnz = Math.max(rMaxNnz, rNnz[i]);
			}
			int[] cNnz = new int[h1.getCols()];
			for(int i=0; i<h1.getCols(); i++) {
				cNnz[i] = probRound((double)h1.cNnz[i] * h2.cNnz[i] * scalec, rn);
				cMaxNnz = Math.max(cMaxNnz, cNnz[i]);
			}
			return new MatrixHistogram(rNnz, null, cNnz, null, rMaxNnz, cMaxNnz);
		}
		
		private static MatrixHistogram derivePlusHistogram(MatrixHistogram h1, MatrixHistogram h2) {
			final double scaler = IntStream.range(0, h1.getCols())
				.mapToDouble(j -> (double)h1.cNnz[j] * h2.cNnz[j])
				.sum() / h1.getNonZeros() / h2.getNonZeros();
			final double scalec = IntStream.range(0, h1.getRows())
				.mapToDouble(j -> (double)h1.rNnz[j] * h2.rNnz[j])
				.sum() / h1.getNonZeros() / h2.getNonZeros();
			int rMaxNnz = 0, cMaxNnz = 0;
			Random rn = new Random();
			int[] rNnz = new int[h1.getRows()];
			for(int i=0; i<h1.getRows(); i++) {
				rNnz[i] = probRound(h1.rNnz[i] + h2.rNnz[i] - (double)h1.rNnz[i] * h2.rNnz[i] * scaler, rn);
				rMaxNnz = Math.max(rMaxNnz, rNnz[i]);
			}
			int[] cNnz = new int[h1.getCols()];
			for(int i=0; i<h1.getCols(); i++) {
				cNnz[i] = probRound(h1.cNnz[i] + h2.cNnz[i] - (double)h1.cNnz[i] * h2.cNnz[i] * scalec, rn);
				cMaxNnz = Math.max(cMaxNnz, cNnz[i]);
			}
			return new MatrixHistogram(rNnz, null, cNnz, null, rMaxNnz, cMaxNnz);
		}
		
		private static MatrixHistogram deriveRbindHistogram(MatrixHistogram h1, MatrixHistogram h2) {
			int[] rNnz = ArrayUtils.addAll(h1.rNnz, h2.rNnz);
			int rMaxNnz = Math.max(h1.rMaxNnz, h2.rMaxNnz);
			int[] cNnz = new int[h1.getCols()];
			int cMaxNnz = 0;
			for(int i=0; i<h1.getCols(); i++) {
				cNnz[i] = h1.cNnz[i] + h2.cNnz[i];
				cMaxNnz = Math.max(cMaxNnz, cNnz[i]);
			}
			return new MatrixHistogram(rNnz, null, cNnz, null, rMaxNnz, cMaxNnz);
		}
		
		private static MatrixHistogram deriveCbindHistogram(MatrixHistogram h1, MatrixHistogram h2) {
			int[] rNnz = new int[h1.getRows()];
			int rMaxNnz = 0;
			for(int i=0; i<h1.getRows(); i++) {
				rNnz[i] = h1.rNnz[i] + h2.rNnz[i];
				rMaxNnz = Math.max(rMaxNnz, rNnz[i]);
			}
			int[] cNnz = ArrayUtils.addAll(h1.cNnz, h2.cNnz);
			int cMaxNnz = Math.max(h1.cMaxNnz, h2.cMaxNnz);
			return new MatrixHistogram(rNnz, null, cNnz, null, rMaxNnz, cMaxNnz);
		}
		
		private static MatrixHistogram deriveEq0Histogram(MatrixHistogram h1) {
			final int m = h1.getRows(), n = h1.getCols();
			int[] rNnz = new int[m], cNnz = new int[n];
			int rMaxNnz = 0, cMaxNnz = 0;
			for(int i=0; i<m; i++) {
				rNnz[i] = n - h1.rNnz[i];
				rMaxNnz = Math.max(rMaxNnz, rNnz[i]);
			}
			for(int j=0; j<n; j++) {
				cNnz[j] = m - h1.cNnz[j];
				cMaxNnz = Math.max(cMaxNnz, cNnz[j]);
			}
			return new MatrixHistogram(rNnz, null, cNnz, null, rMaxNnz, cMaxNnz);
		}
		
		private static MatrixHistogram deriveDiagHistogram(MatrixHistogram h1) {
			if( h1.getCols() == 1 ) { //vector-matrix
				//shallow copy as row count vector is preserved for rows/cols
				return new MatrixHistogram(h1.rNnz, null,
					h1.rNnz, null, h1.rMaxNnz, h1.rMaxNnz);
			}
			else { //matrix-vector
				final int m = h1.getRows(), n = h1.getCols();
				int[] rNnz = new int[m], cNnz = new int[1];
				int rMaxNnz = 0; Random rand = new Random();
				for(int i=0; i<m; i++) {
					rNnz[i] = probRound((double)h1.getNonZeros()/n, rand);
					rMaxNnz = Math.max(rMaxNnz, rNnz[i]);
					cNnz[0] += rNnz[i];
				}
				return new MatrixHistogram(rNnz, null, cNnz, null, rMaxNnz, cNnz[0]);
			}
		}
		
		private static MatrixHistogram deriveTransHistogram(MatrixHistogram h1) {
			return new MatrixHistogram(h1.cNnz, h1.cNnz1e, h1.rNnz, h1.rNnz1e, h1.cMaxNnz, h1.rMaxNnz);
		}
		
		private static MatrixHistogram deriveReshapeHistogram(MatrixHistogram h1, int rows, int cols) {
			if( h1.getRows() == rows )
				return h1;
			else if( h1.getCols() % cols != 0 
				&& h1.getRows() % rows != 0 )
				return null;
			
			//limitation: only applies to scenarios where each input row
			//maps to N output rows, or N input rows map to 1 output row.
			//TODO generalize implementation for partial fractions
			final int m = h1.getRows(), n = h1.getCols();
			int[] rNnz = new int[rows], cNnz = new int[cols];
			int rMaxNnz = 0, cMaxNnz = 0;
			if( h1.getCols() % cols == 0 ) { //1->N rows
				//scale and replicate row counts
				int scale = h1.getCols()/cols;
				for(int i=0, pos=0; i<m; i++, pos+=scale) {
					for(int j=0; j<scale; j++)
						rNnz[pos+j] = h1.rNnz[i]/scale;
					rMaxNnz = Math.max(rMaxNnz, h1.rNnz[i]/scale);
				}
				//aggregate column counts
				for(int j=0; j<n; j++)
					cNnz[j%cols] += h1.cNnz[j];
				for(int j=0; j<cols; j++)
					cMaxNnz = Math.max(cMaxNnz, cNnz[j]);
			}
			else if ( h1.getRows() % rows == 0 ) { //N->1 rows
				int scale = h1.getRows()/rows;
				//scale and replicate column counts
				for(int i=0, pos=0; i<n; i++, pos+=scale) {
					for(int j=0; j<scale; j++)
						cNnz[pos+j] = h1.cNnz[i]/scale;
					cMaxNnz = Math.max(cMaxNnz, h1.cNnz[i]/scale);
				}
				//aggregate row counts
				for(int i=0, pos=0; i<m; i+=scale, pos++)
					for(int i2=0; i2<scale; i2++)
						rNnz[pos] += h1.rNnz[i+i2];
				for(int i=0; i<rows; i++)
					rMaxNnz = Math.max(rMaxNnz, rNnz[i]);
			}
			return new MatrixHistogram(rNnz, null, cNnz, null, rMaxNnz, cMaxNnz);
		}
		
		private static int probRound(double inNnz, Random rand) {
			double temp = Math.floor(inNnz);
			double f = inNnz - temp; //non-int fraction [0,1)
			double randf = rand.nextDouble(); //uniform [0,1)
			return (int)((f > randf) ? temp+1 : temp);
		}
		
		private static int[] deriveSummaryStatistics(int[] counts, int N) {
			int max = Integer.MIN_VALUE, N2 = N/2;
			int cntN1 = 0, cntNeq0 = 0, cntNdiv2 = 0;
			for(int i=0; i<counts.length; i++) {
				final int cnti = counts[i];
				max = Math.max(max, cnti);
				cntN1 += (cnti == 1) ? 1 : 0;
				cntNeq0 += (cnti != 0) ? 1 : 0;
				cntNdiv2 += (cnti > N2) ? 1 : 0;
			}
			return new int[]{max, cntN1, cntNeq0, cntNdiv2};
		}
	}
}
