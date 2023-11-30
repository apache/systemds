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

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlockCSR;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.data.SparseRowVector;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.runtime.util.UtilFunctions;

public interface LibMatrixDenseToSparse {
	public static final Log LOG = LogFactory.getLog(LibMatrixDenseToSparse.class.getName());

	/**
	 * Convert the given matrix block to a sparse allocation.
	 * 
	 * @param r        The matrix block to modify, and return the sparse block in.
	 * @param allowCSR If CSR is allowed.
	 */
	public static void denseToSparse(MatrixBlock r, boolean allowCSR) {
		denseToSparse(r, allowCSR, 1);
	}

	public static void denseToSparse(MatrixBlock r, boolean allowCSR, int k) {
		final DenseBlock a = r.getDenseBlock();

		// set target representation, early abort on empty blocks
		r.sparse = true;
		if(a == null)
			return;

		if(k > 1 && r.getSparsity() > 0.01 && (r.rlen > 100 || ((long) r.rlen * r.clen > 100000)))
			denseToSparseParallel(r, k, allowCSR);
		else if(allowCSR && r.nonZeros <= Integer.MAX_VALUE)
			denseToSparseCSRSafe(r);
		else
			denseToSparseMCSR(r);

		// cleanup dense block
		r.denseBlock = null;
	}

	private static void denseToSparseCSRSafe(MatrixBlock r) {
		try {
			if(r.getNumRows() == 1)
				denseToSparseCSRSingleRow(r);
			else
				denseToSparseCSR(r);
		}
		catch(ArrayIndexOutOfBoundsException ioobe) {
			r.sparse = false;
			// this means something was wrong with the sparse count.
			final long nnzBefore = r.nonZeros;
			final long nnzNew = r.recomputeNonZeros();

			// try again.
			if(nnzBefore != nnzNew)
				denseToSparse(r, true);
			else
				denseToSparse(r, false);

		}
	}

	private static void denseToSparseCSRSingleRow(MatrixBlock r) {
		final DenseBlock a = r.getDenseBlock();
		final int n = r.clen;
		final int lnnz = (int) r.nonZeros;
		final int[] indexes = new int[lnnz];
		final double[] values = new double[lnnz];
		final double[] avals = a.values(0);
		addColCSR(avals, n, 0, indexes, values, 0);
		final int[] rptr = new int[2];
		rptr[1] = lnnz;
		r.sparseBlock = new SparseBlockCSR(rptr, indexes, values, lnnz);
	}

	private static void denseToSparseCSR(MatrixBlock r) {
		final DenseBlock a = r.getDenseBlock();
		final int m = r.rlen;
		final int n = r.clen;
		// allocate target in memory-efficient CSR format
		final int lnnz = (int) r.nonZeros;
		final int[] rptr = new int[m + 1];
		final int[] indexes = new int[lnnz];
		final double[] values = new double[lnnz];
		for(int i = 0, pos = 0; i < m; i++) {
			final double[] avals = a.values(i);
			final int aix = a.pos(i);
			pos = addColCSR(avals, n, aix, indexes, values, pos);
			rptr[i + 1] = pos;
		}
		r.sparseBlock = new SparseBlockCSR(rptr, indexes, values, lnnz);
	}

	private static int addColCSR(double[] avals, int n, int aix, int[] indexes, double[] values, int pos) {
		for(int j = 0; j < n; j++) {
			final double aval = avals[aix + j];
			if(aval != 0) {
				indexes[pos] = j;
				values[pos] = aval;
				pos++;
			}
		}
		return pos;
	}

	private static void denseToSparseMCSR(MatrixBlock r) {
		final DenseBlock a = r.getDenseBlock();

		final int m = r.rlen;
		final int n = r.clen;
		// remember number non zeros.
		long nnzTemp = r.getNonZeros();
		// fallback to less-memory efficient MCSR format,
		// which however allows much larger sparse matrices
		if(!r.allocateSparseRowsBlock())
			r.reset(); // reset if not allocated
		SparseBlockMCSR sblock = (SparseBlockMCSR) r.sparseBlock;
		toSparseMCSRRangeScan(a, sblock, n, 0, m);
		r.nonZeros = nnzTemp;
	}

	private static void toSparseMCSRRangeNoScan(DenseBlock a, SparseBlockMCSR b, int n, int rl, int ru, int est) {
		for(int i = rl; i < ru; i++)
			toSparseMCSRRowNoScan(a, b, n, i, est);
	}

	private static void toSparseMCSRRangeScan(DenseBlock a, SparseBlockMCSR b, int n, int rl, int ru) {
		for(int i = rl; i < ru; i++)
			toSparseMCSRRowScan(a, b, n, i);
	}

	private static void toSparseMCSRRowScan(DenseBlock a, SparseBlockMCSR b, int n, int i) {
		final double[] avals = a.values(i);
		final int aix = a.pos(i);
		// compute nnz per row (not via recomputeNonZeros as sparse allocated)
		final int lnnz = UtilFunctions.computeNnz(avals, aix, n);
		if(lnnz <= 0)
			return;
		SparseRowVector sb = toSparseMCSRRowKnownNNZ(avals, aix, n, i, lnnz);
		b.set(i, sb, false);
	}

	private static void toSparseMCSRRowNoScan(DenseBlock a, SparseBlockMCSR b, int n, int i, int est) {
		final double[] avals = a.values(i);
		final int aix = a.pos(i);
		final SparseRowVector sb = new SparseRowVector(est / 2);
		toSparseMCSRRowUnknownNNZ(avals, aix, n, i, sb);
		b.set(i, sb, false);
	}

	private static SparseRowVector toSparseMCSRRowKnownNNZ(double[] avals, int aix, int n, int i, int lnnz) {
		// allocate sparse row and append non-zero values
		final double[] vals = new double[lnnz];
		final int[] idx = new int[lnnz];
		for(int j = 0, o = 0; j < n; j++) {
			final double v = avals[aix + j];
			if(v != 0.0) {
				vals[o] = v;
				idx[o] = j;
				o++;
			}
		}
		return new SparseRowVector(vals, idx);
	}

	private static void toSparseMCSRRowUnknownNNZ(final double[] avals, int aix, final int n, final int i,
		final SparseRowVector sb) {
		for(int j = 0; j < n; j++)
			sb.append(j, avals[aix++]);
		sb.compact();
	}

	private static void denseToSparseParallel(MatrixBlock r, int k, boolean allowCSR) {
		final DenseBlock a = r.getDenseBlock();
		final ExecutorService pool = CommonThreadPool.get(k);
		try {
			r.denseBlock = null;
			r.sparseBlock = null;
			final int m = r.rlen;
			final int n = r.clen;
			// remember number non zeros.
			final long nnzTemp = r.getNonZeros();
			r.reset(r.getNumRows(), r.getNumColumns(), nnzTemp);
			// fallback to less-memory efficient MCSR format, for efficient parallel conversion.
			r.sparseBlock = new SparseBlockMCSR(r.getNumRows());
			r.sparse = true;
			final SparseBlockMCSR b = (SparseBlockMCSR) r.sparseBlock;
			final int blockSize = Math.max(1, m / k);
			double sp = r.getSparsity();
			final int est = Math.max(1, (int) (n * sp));

			List<Future<?>> tasks = new ArrayList<>();
			for(int i = 0; i < m; i += blockSize) {
				final int start = i;
				final int end = Math.min(m, i + blockSize);
				if(sp < 0.1)
					tasks.add(pool.submit(() -> toSparseMCSRRangeNoScan(a, b, n, start, end, est)));
				else
					tasks.add(pool.submit(() -> toSparseMCSRRangeScan(a, b, n, start, end)));
			}

			r.nonZeros = nnzTemp;
			for(Future<?> t : tasks)
				t.get();
		}
		catch(Exception e) {
			throw new RuntimeException(e);
		}
		finally {
			pool.shutdown();
		}
	}
}