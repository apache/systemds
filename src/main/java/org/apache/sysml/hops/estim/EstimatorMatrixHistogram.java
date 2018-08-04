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

import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

import org.apache.directory.api.util.exception.NotImplementedException;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.matrix.data.DenseBlock;
import org.apache.sysml.runtime.matrix.data.LibMatrixAgg;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.SparseBlock;

/**
 * This estimator implements a remarkably simple yet effective approach for
 * incorporating structural properties into sparsity estimation. The key idea is
 * to maintain row and column nnz per matrix, along with additional meta data.
 */
public class EstimatorMatrixHistogram extends SparsityEstimator {
	// internal configurations
	private static final boolean DEFAULT_USE_EXCEPTS = true;

	private final boolean _useExcepts;

	public EstimatorMatrixHistogram() {
		this(DEFAULT_USE_EXCEPTS);
	}

	public EstimatorMatrixHistogram(boolean useExcepts) {
		_useExcepts = useExcepts;
	}

	@Override
	public double estim(MMNode root) {
		// recursive histogram computation of non-leaf nodes
		if (!root.getLeft().isLeaf())
			estim(root.getLeft()); // obtain synopsis
		if (!root.getRight().isLeaf())
			estim(root.getLeft()); // obtain synopsis
		MatrixHistogram h1 = !root.getLeft().isLeaf() ? (MatrixHistogram) root.getLeft().getSynopsis()
				: new MatrixHistogram(root.getLeft().getData(), _useExcepts);
		MatrixHistogram h2 = !root.getRight().isLeaf() ? (MatrixHistogram) root.getRight().getSynopsis()
				: new MatrixHistogram(root.getRight().getData(), _useExcepts);

		// estimate output sparsity based on input histograms
		double ret = estimIntern(h1, h2, OpCode.MM);

		// derive and memoize output histogram
		root.setSynopsis(MatrixHistogram.deriveOutputHistogram(h1, h2, ret));

		return ret;
	}

	@Override
	public double estim(MatrixBlock m1, MatrixBlock m2) {
		return estim(m1, m2, OpCode.MM);
	}

	@Override
	public double estim(MatrixBlock m1, MatrixBlock m2, OpCode op) {
		MatrixHistogram h1 = new MatrixHistogram(m1, _useExcepts);
		MatrixHistogram h2 = (m1 == m2) ? // self product
				h1 : new MatrixHistogram(m2, _useExcepts);
		return estimIntern(h1, h2, op);
	}

	@Override
	public double estim(MatrixBlock m1, OpCode op) {
		MatrixHistogram h1 = new MatrixHistogram(m1, _useExcepts);
		return estimIntern(h1, null, op);
	}

	private double estimIntern(MatrixHistogram h1, MatrixHistogram h2, OpCode op) {
		double msize = (double) h1.getRows() * h1.getCols();

		switch (op) {
		case MM:
			return estimInternMM(h1, h2);
		case MULT:
			return Math.min(
					IntStream.range(0, h1.getRows()).mapToDouble(i -> h1.rNnz[i] / msize * h2.rNnz[i] / msize).sum(),
					IntStream.range(0, h1.getCols()).mapToDouble(i -> h1.cNnz[i] / msize * h2.cNnz[i] / msize).sum());
		case PLUS:
			return Math.min(
					IntStream.range(0, h1.getRows())
							.mapToDouble(i -> h1.rNnz[i] / msize + h2.rNnz[i] / msize
									- h1.rNnz[i] / msize * h2.rNnz[i] / msize)
							.sum(),
					IntStream.range(0, h1.getCols()).mapToDouble(
							i -> h1.cNnz[i] / msize + h2.cNnz[i] / msize - h1.cNnz[i] / msize * h2.cNnz[i] / msize)
							.sum());
		case EQZERO:
			return OptimizerUtils.getSparsity(h1.getRows(), h1.getCols(),
					(long) h1.getRows() * h1.getCols() - h1.getNonZeros());
		case DIAG:
			return (h1.getCols() == 1) ? OptimizerUtils.getSparsity(h1.getRows(), h1.getRows(), h1.getNonZeros())
					: OptimizerUtils.getSparsity(h1.getRows(), 1, Math.min(h1.getRows(), h1.getNonZeros()));
		// binary operations that preserve sparsity exactly
		case CBIND:
			return OptimizerUtils.getSparsity(h1.getRows(), h1.getCols() + h2.getCols(),
					h1.getNonZeros() + h2.getNonZeros());
		case RBIND:
			return OptimizerUtils.getSparsity(h1.getRows() + h2.getRows(), h1.getCols(),
					h1.getNonZeros() + h2.getNonZeros());
		// unary operation that preserve sparsity exactly
		case NEQZERO:
			return OptimizerUtils.getSparsity(h1.getRows(), h1.getCols(), h1.getNonZeros());
		case TRANS:
			return OptimizerUtils.getSparsity(h1.getRows(), h1.getCols(), h1.getNonZeros());
		case RESHAPE:
			return OptimizerUtils.getSparsity(h1.getRows(), h1.getCols(), h1.getNonZeros());
		default:
			throw new NotImplementedException();
		}
	}

	private double estimInternMM(MatrixHistogram h1, MatrixHistogram h2) {
		long nnz = 0;
		// special case, with exact sparsity estimate, where the dot product
		// dot(h1.cNnz,h2rNnz) gives the exact number of non-zeros in the output
		if (h1.rMaxNnz <= 1 || h2.cMaxNnz <= 1) {
			for (int j = 0; j < h1.getCols(); j++)
				nnz += h1.cNnz[j] * h2.rNnz[j];
		}
		// special case, with hybrid exact and approximate output
		else if (h1.cNnz1e != null && h2.rNnz1e != null) {
			// note: normally h1.getRows()*h2.getCols() would define mnOut
			// but by leveraging the knowledge of rows/cols w/ <=1 nnz, we
			// account
			// that exact and approximate fractions touch different areas
			long mnOut = (h1.rNonEmpty - h1.rN1) * (h2.cNonEmpty - h2.cN1);
			double spOutRest = 0;
			for (int j = 0; j < h1.getCols(); j++) {
				// exact fractions, w/o double counting
				nnz += h1.cNnz1e[j] * h2.rNnz[j];
				nnz += (h1.cNnz[j] - h1.cNnz1e[j]) * h2.rNnz1e[j];
				// approximate fraction, w/o double counting
				double lsp = (double) (h1.cNnz[j] - h1.cNnz1e[j]) * (h2.rNnz[j] - h2.rNnz1e[j]) / mnOut;
				spOutRest = spOutRest + lsp - spOutRest * lsp;
			}
			nnz += (long) (spOutRest * mnOut);
		}
		// general case with approximate output
		else {
			long mnOut = h1.getRows() * h2.getCols();
			double spOut = 0;
			for (int j = 0; j < h1.getCols(); j++) {
				double lsp = (double) h1.cNnz[j] * h2.rNnz[j] / mnOut;
				spOut = spOut + lsp - spOut * lsp;
			}
			nnz = (long) (spOut * mnOut);
		}

		// exploit upper bound on nnz based on non-empty rows/cols
		nnz = (h1.rNonEmpty >= 0 && h2.cNonEmpty >= 0) ? Math.min((long) h1.rNonEmpty * h2.cNonEmpty, nnz) : nnz;

		// exploit lower bound on nnz based on half-full rows/cols
		nnz = (h1.rNdiv2 >= 0 && h2.cNdiv2 >= 0) ? Math.max((long) h1.rNdiv2 * h2.cNdiv2, nnz) : nnz;

		// compute final sparsity
		return OptimizerUtils.getSparsity(h1.getRows(), h2.getCols(), nnz);
	}

	private static class MatrixHistogram {
		// count vectors (the histogram)
		private final int[] rNnz; // nnz per row
		private int[] rNnz1e = null; // nnz per row for cols w/ <= 1 non-zeros
		private final int[] cNnz; // nnz per col
		private int[] cNnz1e = null; // nnz per col for rows w/ <= 1 non-zeros
		// additional summary statistics
		private final int rMaxNnz, cMaxNnz; // max nnz per row/row
		private final int rN1, cN1; // number of rows/cols with nnz=1
		private final int rNonEmpty, cNonEmpty; // number of non-empty rows/cols
												// (w/ empty is nnz=0)
		private final int rNdiv2, cNdiv2; // number of rows/cols with nnz >
											// #cols/2 and #rows/2
		private boolean fullDiag; // true if there exists a full diagonal of
									// nonzeros

		public MatrixHistogram(MatrixBlock in, boolean useExcepts) {
			// 1) allocate basic synopsis
			rNnz = new int[in.getNumRows()];
			cNnz = new int[in.getNumColumns()];
			fullDiag = in.getNumRows() == in.getNonZeros();

			// 2) compute basic synopsis details
			if (!in.isEmpty()) {
				if (in.isInSparseFormat()) {
					SparseBlock sblock = in.getSparseBlock();
					for (int i = 0; i < in.getNumRows(); i++) {
						if (sblock.isEmpty(i))
							continue;
						int apos = sblock.pos(i);
						int alen = sblock.size(i);
						int[] aix = sblock.indexes(i);
						rNnz[i] = alen;
						LibMatrixAgg.countAgg(sblock.values(i), cNnz, aix, apos, alen);
						fullDiag &= aix[apos] == i;
					}
				} else {
					DenseBlock dblock = in.getDenseBlock();
					for (int i = 0; i < in.getNumRows(); i++) {
						double[] avals = dblock.values(i);
						int lnnz = 0, aix = dblock.pos(i);
						for (int j = 0; j < in.getNumColumns(); j++) {
							if (avals[aix + j] != 0) {
								fullDiag &= (i == j);
								cNnz[j]++;
								lnnz++;
							}
						}
						rNnz[i] = lnnz;
					}
				}
			}

			// 3) compute meta data synopsis
			rMaxNnz = Arrays.stream(rNnz).max().orElse(0);
			cMaxNnz = Arrays.stream(cNnz).max().orElse(0);
			rN1 = (int) Arrays.stream(rNnz).filter(item -> item == 1).count();
			cN1 = (int) Arrays.stream(cNnz).filter(item -> item == 1).count();
			rNonEmpty = (int) Arrays.stream(rNnz).filter(v -> v != 0).count();
			cNonEmpty = (int) Arrays.stream(cNnz).filter(v -> v != 0).count();
			rNdiv2 = (int) Arrays.stream(rNnz).filter(item -> item > getCols() / 2).count();
			cNdiv2 = (int) Arrays.stream(cNnz).filter(item -> item > getRows() / 2).count();

			// 4) compute exception details if necessary (optional)
			if (useExcepts & !in.isEmpty() && (rMaxNnz > 1 || cMaxNnz > 1)) {
				rNnz1e = new int[in.getNumRows()];
				cNnz1e = new int[in.getNumColumns()];

				if (in.isInSparseFormat()) {
					SparseBlock sblock = in.getSparseBlock();
					for (int i = 0; i < in.getNumRows(); i++) {
						if (sblock.isEmpty(i))
							continue;
						int alen = sblock.size(i);
						int apos = sblock.pos(i);
						int[] aix = sblock.indexes(i);
						for (int k = apos; k < apos + alen; k++)
							rNnz1e[i] += cNnz[aix[k]] <= 1 ? 1 : 0;
						if (alen <= 1)
							for (int k = apos; k < apos + alen; k++)
								cNnz1e[aix[k]]++;
					}
				} else {
					DenseBlock dblock = in.getDenseBlock();
					for (int i = 0; i < in.getNumRows(); i++) {
						double[] avals = dblock.values(i);
						int aix = dblock.pos(i);
						boolean rNnzlte1 = rNnz[i] <= 1;
						for (int j = 0; j < in.getNumColumns(); j++) {
							if (avals[aix + j] != 0) {
								rNnz1e[i] += cNnz[j] <= 1 ? 1 : 0;
								cNnz1e[j] += rNnzlte1 ? 1 : 0;
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
			return getRows() < getCols() ? IntStream.range(0, getRows()).mapToLong(i -> rNnz[i]).sum()
					: IntStream.range(0, getRows()).mapToLong(i -> cNnz[i]).sum();
		}

		public static MatrixHistogram deriveOutputHistogram(MatrixHistogram h1, MatrixHistogram h2, double spOut) {
			// exact propagation if lhs or rhs full diag
			if (h1.fullDiag)
				return h2;
			if (h2.fullDiag)
				return h1;

			// get input/output nnz for scaling
			long nnz1 = Arrays.stream(h1.rNnz).sum();
			long nnz2 = Arrays.stream(h2.cNnz).sum();
			double nnzOut = spOut * h1.getRows() * h2.getCols();

			// propagate h1.r and h2.c to output via simple scaling
			// (this implies 0s propagate and distribution is preserved)
			int rMaxNnz = 0, cMaxNnz = 0;
			int[] rNnz = new int[h1.getRows()];
			Random rn = new Random();
			for (int i = 0; i < h1.getRows(); i++) {
				rNnz[i] = probRound(nnzOut / nnz1 * h1.rNnz[i], rn);
				rMaxNnz = Math.max(rMaxNnz, rNnz[i]);
			}
			int[] cNnz = new int[h2.getCols()];
			for (int i = 0; i < h2.getCols(); i++) {
				cNnz[i] = probRound(nnzOut / nnz2 * h2.cNnz[i], rn);
				cMaxNnz = Math.max(cMaxNnz, cNnz[i]);
			}

			// construct new histogram object
			return new MatrixHistogram(rNnz, null, cNnz, null, rMaxNnz, cMaxNnz);
		}

		private static int probRound(double inNnz, Random rand) {
			double temp = Math.floor(inNnz);
			double f = inNnz - temp; // non-int fraction [0,1)
			double randf = rand.nextDouble(); // uniform [0,1)
			return (int) ((f > randf) ? temp + 1 : temp);
		}
	}
}
