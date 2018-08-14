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

import java.util.Random;
import java.util.stream.IntStream;

import org.apache.commons.lang.ArrayUtils;
import org.apache.directory.api.util.exception.NotImplementedException;
import org.apache.sysml.hops.OptimizerUtils;
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
	//internal configurations
	private static final boolean DEFAULT_USE_EXCEPTS = true;
	
	private final boolean _useExcepts;
	
	public EstimatorMatrixHistogram() {
		this(DEFAULT_USE_EXCEPTS);
	}
	
	public EstimatorMatrixHistogram(boolean useExcepts) {
		_useExcepts = useExcepts;
	}
	
	@Override
	public MatrixCharacteristics estim(MMNode root) {
		//recursive histogram computation of non-leaf nodes
		if( !root.getLeft().isLeaf() )
			estim(root.getLeft()); //obtain synopsis
		if( !root.getRight().isLeaf() )
			estim(root.getRight()); //obtain synopsis
		MatrixHistogram h1 = !root.getLeft().isLeaf() ?
			(MatrixHistogram)root.getLeft().getSynopsis() :
			new MatrixHistogram(root.getLeft().getData(), _useExcepts);
		MatrixHistogram h2 = !root.getRight().isLeaf() ?
			(MatrixHistogram)root.getRight().getSynopsis() :
			new MatrixHistogram(root.getRight().getData(), _useExcepts);
		
		//estimate output sparsity based on input histograms
		double ret = estimIntern(h1, h2, root.getOp());
		MatrixHistogram outMap = MatrixHistogram.deriveOutputHistogram(h1, h2, ret, root.getOp());
		root.setSynopsis(outMap);
		return root.setMatrixCharacteristics(new MatrixCharacteristics(
			outMap.getRows(), outMap.getCols(), outMap.getNonZeros()));

	}
	
	@Override 
	public double estim(MatrixBlock m1, MatrixBlock m2) {
		return estim(m1, m2, OpCode.MM);
	}
	
	@Override
	public double estim(MatrixBlock m1, MatrixBlock m2, OpCode op) {
		if( isExactMetadataOp(op) )
			return estimExactMetaData(m1.getMatrixCharacteristics(),
				m2.getMatrixCharacteristics(), op).getSparsity();
		MatrixHistogram h1 = new MatrixHistogram(m1, _useExcepts);
		MatrixHistogram h2 = (m1 == m2) ? //self product
			h1 : new MatrixHistogram(m2, _useExcepts);
		return estimIntern(h1, h2, op);
	}
	
	@Override
	public double estim(MatrixBlock m1, OpCode op) {
		if( isExactMetadataOp(op) )
			return estimExactMetaData(m1.getMatrixCharacteristics(), null, op).getSparsity();
		MatrixHistogram h1 = new MatrixHistogram(m1, _useExcepts);
		return estimIntern(h1, null, op);
	}
	
	private double estimIntern(MatrixHistogram h1, MatrixHistogram h2, OpCode op) {
		double msize = (double)h1.getRows()*h1.getCols();
		switch (op) {
			case MM:
				return estimInternMM(h1, h2);
			case MULT: {
				final double N1 = h1.getNonZeros();
				final double N2 = h2.getNonZeros();
				final long scale = IntStream.range(0, h1.getCols())
					.mapToLong(j -> (long)h1.cNnz[j] * h2.cNnz[j]).sum();
				return IntStream.range(0, h1.getRows())
					.mapToDouble(i -> (long)h1.rNnz[i] * h2.rNnz[i] * scale / N1 / N2) //collisions
					.sum() / msize;
			}
			case PLUS: {
				final double N1 = h1.getNonZeros();
				final double N2 = h2.getNonZeros();
				final long scale = IntStream.range(0, h1.getCols())
					.mapToLong(j -> (long)h1.cNnz[j] * h2.cNnz[j]).sum();
				return IntStream.range(0, h1.getRows())
					.mapToDouble(i -> (long)h1.rNnz[i] + h2.rNnz[i] //all minus collisions
						- (long)h1.rNnz[i] * h2.rNnz[i] * scale / N1 / N2)
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
				nnz += h1.cNnz[j] * h2.rNnz[j];
		}
		//special case, with hybrid exact and approximate output
		else if(h1.cNnz1e!=null && h2.rNnz1e != null) {
			//note: normally h1.getRows()*h2.getCols() would define mnOut
			//but by leveraging the knowledge of rows/cols w/ <=1 nnz, we account
			//that exact and approximate fractions touch different areas
			long mnOut = (h1.rNonEmpty-h1.rN1) * (h2.cNonEmpty-h2.cN1);
			double spOutRest = 0;
			for( int j=0; j<h1.getCols(); j++ ) {
				//exact fractions, w/o double counting
				nnz += h1.cNnz1e[j] * h2.rNnz[j];
				nnz += (h1.cNnz[j]-h1.cNnz1e[j]) * h2.rNnz1e[j];
				//approximate fraction, w/o double counting
				double lsp = (double)(h1.cNnz[j]-h1.cNnz1e[j]) 
					* (h2.rNnz[j]-h2.rNnz1e[j]) / mnOut;
				spOutRest = spOutRest + lsp - spOutRest*lsp;
			}
			nnz += (long)(spOutRest * mnOut);
		}
		//general case with approximate output
		else {
			long mnOut = h1.getRows()*h2.getCols();
			double spOut = 0;
			for( int j=0; j<h1.getCols(); j++ ) {
				double lsp = (double) h1.cNnz[j] * h2.rNnz[j] / mnOut;
				spOut = spOut + lsp - spOut*lsp;
			}
			nnz = (long)(spOut * mnOut);
		}
		
		//exploit upper bound on nnz based on non-empty rows/cols
		nnz = (h1.rNonEmpty >= 0 && h2.cNonEmpty >= 0) ?
			Math.min((long)h1.rNonEmpty * h2.cNonEmpty, nnz) : nnz;
		
		//exploit lower bound on nnz based on half-full rows/cols
		nnz = (h1.rNdiv2 >= 0 && h2.cNdiv2 >= 0) ?
			Math.max((long)h1.rNdiv2 * h2.cNdiv2, nnz) : nnz;
		
		//compute final sparsity
		return OptimizerUtils.getSparsity(
			h1.getRows(), h2.getCols(), nnz);
	}
	
	private static class MatrixHistogram {
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
		
		public MatrixHistogram(MatrixBlock in, boolean useExcepts) {
			// 1) allocate basic synopsis
			final int m = in.getNumRows();
			final int n = in.getNumColumns();
			rNnz = new int[in.getNumRows()];
			cNnz = new int[in.getNumColumns()];
			fullDiag = in.getNumRows() == in.getNonZeros()
				&& in.getNumRows() == in.getNumColumns();
			
			// 2) compute basic synopsis details
			if( !in.isEmpty() ) {
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
			if( useExcepts & !in.isEmpty() && (rMaxNnz > 1 || cMaxNnz > 1) ) {
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
			rNonEmpty = cNonEmpty = -1;
			rNdiv2 = cNdiv2 = -1;
		}
		
		public int getRows() {
			return rNnz.length;
		}
		
		public int getCols() {
			return cNnz.length;
		}
		
		public long getNonZeros() {
			return getRows() < getCols() ?
				IntStream.range(0, getRows()).mapToLong(i-> rNnz[i]).sum() :
				IntStream.range(0, getCols()).mapToLong(i-> cNnz[i]).sum();
		}
		
		public static MatrixHistogram deriveOutputHistogram(MatrixHistogram h1, MatrixHistogram h2, double spOut, OpCode op) {
			switch(op) {
				case MM:    return deriveMMHistogram(h1, h2, spOut);
				case MULT:  return deriveMultHistogram(h1, h2);
				case PLUS:  return derivePlusHistogram(h1, h2);
				case RBIND: return deriveRbindHistogram(h1, h2);
				case CBIND: return deriveCbindHistogram(h1, h2);
				//TODO add missing unary operations
				default:
					throw new NotImplementedException();
			}
		}
		
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
			for( int i=0; i<h2.getCols(); i++ ) {
				cNnz[i] = probRound(nnzOut/nnz2 * h2.cNnz[i], rn);
				cMaxNnz = Math.max(cMaxNnz, cNnz[i]);
			}
			
			//construct new histogram object
			return new MatrixHistogram(rNnz, null, cNnz, null, rMaxNnz, cMaxNnz);
		}
		
		private static MatrixHistogram deriveMultHistogram(MatrixHistogram h1, MatrixHistogram h2) {
			final double N1 = h1.getNonZeros();
			final double N2 = h2.getNonZeros();
			final double scaler = IntStream.range(0, h1.getCols())
				.mapToDouble(j -> (long)h1.cNnz[j] * h2.cNnz[j]).sum();
			final double scalec = IntStream.range(0, h1.getRows())
				.mapToDouble(j -> (long)h1.rNnz[j] * h2.rNnz[j]).sum();
			int rMaxNnz = 0, cMaxNnz = 0;
			Random rn = new Random();
			int[] rNnz = new int[h1.getRows()];
			for(int i=0; i<h1.getRows(); i++) {
				rNnz[i] = probRound(h1.rNnz[i] * h2.rNnz[i] * scaler / N1 / N2, rn);
				rMaxNnz = Math.max(rMaxNnz, rNnz[i]);
			}
			int[] cNnz = new int[h1.getCols()];
			for(int i=0; i<h1.getCols(); i++) {
				cNnz[i] = probRound(h1.cNnz[i] * h2.cNnz[i] * scalec / N1 / N2, rn);
				cMaxNnz = Math.max(cMaxNnz, cNnz[i]);
			}
			return new MatrixHistogram(rNnz, null, cNnz, null, rMaxNnz, cMaxNnz);
		}
		
		private static MatrixHistogram derivePlusHistogram(MatrixHistogram h1, MatrixHistogram h2) {
			double msize = (double)h1.getRows()*h1.getCols();
			int rMaxNnz = 0, cMaxNnz = 0;
			Random rn = new Random();
			int[] rNnz = new int[h1.getRows()];
			for(int i=0; i<h1.getRows(); i++) {
				rNnz[i] = probRound(h1.rNnz[i]/msize + h2.rNnz[i]/msize - h1.rNnz[i]/msize * h2.rNnz[i]/msize, rn);
				rMaxNnz = Math.max(rMaxNnz, rNnz[i]);
			}
			int[] cNnz = new int[h1.getCols()];
			for(int i=0; i<h1.getCols(); i++) {
				cNnz[i] = probRound(h1.cNnz[i]/msize + h2.cNnz[i]/msize - h1.cNnz[i]/msize * h2.cNnz[i]/msize, rn);
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
