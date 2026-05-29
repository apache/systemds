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
package org.apache.sysds.runtime.frame.data.lib;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;

public class MatrixBlockFromFrame {
	public static final Log LOG = LogFactory.getLog(MatrixBlockFromFrame.class.getName());

	public static final int blocksizeIJ = 32;

	public static Boolean WARNED_FOR_FAILED_CAST = false;

	private MatrixBlockFromFrame(){
		// private constructor for code coverage.
	}

	/**
	 * Converts a frame block with arbitrary schema into a matrix block. Since matrix block only supports value type
	 * double, we do a best effort conversion of non-double types which might result in errors for non-numerical data.
	 *
	 * @param frame Frame block to convert
	 * @param k     The parallelization degree
	 * @return MatrixBlock
	 */
	public static MatrixBlock convertToMatrixBlock(FrameBlock frame, int k) {
		return convertToMatrixBlock(frame, null, k);
	}

	/**
	 * Converts a frame block with arbitrary schema into a matrix block. Since matrix block only supports value type
	 * double, we do a best effort conversion of non-double types which might result in errors for non-numerical data.
	 *
	 * @param frame FrameBlock to convert
	 * @param ret   The returned MatrixBlock
	 * @param k     The parallelization degree
	 * @return MatrixBlock
	 */
	public static MatrixBlock convertToMatrixBlock(FrameBlock frame, MatrixBlock ret, int k) {
		try {

			final int m = frame.getNumRows();
			final int n = frame.getNumColumns();
			ret = allocateRet(ret, m, n);

			if(k == -1)
				k = InfrastructureAnalyzer.getLocalParallelism();

			long nnz = 0;
			if(k == 1)
				nnz = convert(frame, ret, n, 0, m);
			else
				nnz = convertParallel(frame, ret, m, n, k);

			ret.setNonZeros(nnz);
			ret.examSparsity();
			return ret;
		}
		catch(Exception e) {
			throw new DMLRuntimeException("Failed to convert FrameBlock to MatrixBlock", e);
		}
	}

	private static MatrixBlock allocateRet(MatrixBlock ret, final int m, final int n) {
		if(ret == null)
			ret = new MatrixBlock(m, n, false);
		else if(ret.getNumRows() != m || ret.getNumColumns() != n || ret.isInSparseFormat())
			ret.reset(m, n, false);
		if(!ret.isAllocated())
			ret.allocateDenseBlock();
		return ret;
	}

	private static long convert(FrameBlock frame, MatrixBlock mb, int n, int rl, int ru) {
		try {

			if(mb.getDenseBlock().isContiguous())
				return convertContiguous(frame, mb, n, rl, ru);
			else
				return convertGeneric(frame, mb, n, rl, ru);
		}
		catch(NumberFormatException | DMLRuntimeException e) {
			synchronized(WARNED_FOR_FAILED_CAST){
				if(!WARNED_FOR_FAILED_CAST) {
					LOG.error(
						"Failed to convert to Matrix because of number format errors, falling back to NaN on incompatible cells",
						e);
					WARNED_FOR_FAILED_CAST = true;
				}
			}
			return convertSafeCast(frame, mb, n, rl, ru);

		}
	}

	private static long convertParallel(FrameBlock frame, MatrixBlock mb, int m, int n, int k) throws Exception {
		ExecutorService pool = CommonThreadPool.get(k);
		try {
			List<Future<Long>> tasks = new ArrayList<>();
			final int blkz = Math.max(m / k, 1000);

			for(int i = 0; i < m; i += blkz) {
				final int start = i;
				final int end = Math.min(i + blkz, m);
				tasks.add(pool.submit(() -> convert(frame, mb, n, start, end)));
			}

			long nnz = 0;
			for(Future<Long> t : tasks)
				nnz += t.get();
			return nnz;
		}

		finally {
			pool.shutdown();
		}
	}

	private static long convertContiguous(final FrameBlock frame, final MatrixBlock mb, final int n, final int rl,
		final int ru) {
		long lnnz = 0;
		double[] c = mb.getDenseBlockValues();
		for(int bi = rl; bi < ru; bi += blocksizeIJ) {
			for(int bj = 0; bj < n; bj += blocksizeIJ) {
				int bimin = Math.min(bi + blocksizeIJ, ru);
				int bjmin = Math.min(bj + blocksizeIJ, n);
				lnnz = convertBlockContiguous(frame, n, lnnz, c, bi, bj, bimin, bjmin);
			}
		}
		return lnnz;
	}

	private static long convertBlockContiguous(final FrameBlock frame, final int n, long lnnz, double[] c, int rl,
		int cl, int ru, int cu) {
		for(int i = rl, aix = rl * n; i < ru; i++, aix += n)
			for(int j = cl; j < cu; j++)
				lnnz += (c[aix + j] = frame.getDoubleNaN(i, j)) != 0 ? 1 : 0;
		return lnnz;
	}

	private static long convertGeneric(final FrameBlock frame, final MatrixBlock mb, final int n, final int rl,
		final int ru) {
		long lnnz = 0;
		final DenseBlock c = mb.getDenseBlock();
		for(int bi = rl; bi < ru; bi += blocksizeIJ) {
			for(int bj = 0; bj < n; bj += blocksizeIJ) {
				int bimin = Math.min(bi + blocksizeIJ, ru);
				int bjmin = Math.min(bj + blocksizeIJ, n);
				lnnz = convertBlockGeneric(frame, lnnz, c, bi, bj, bimin, bjmin);
			}
		}
		return lnnz;
	}

	private static long convertBlockGeneric(final FrameBlock frame, long lnnz, final DenseBlock c, final int rl,
		final int cl, final int ru, final int cu) {
		for(int i = rl; i < ru; i++) {
			final double[] cvals = c.values(i);
			final int cpos = c.pos(i);
			for(int j = cl; j < cu; j++)
				lnnz += (cvals[cpos + j] = frame.getDoubleNaN(i, j)) != 0 ? 1 : 0;
		}
		return lnnz;
	}

	private static long convertSafeCast(final FrameBlock frame, final MatrixBlock mb, final int n, final int rl,
		final int ru) {
		final DenseBlock c = mb.getDenseBlock();
		long lnnz = 0;
		for(int bi = rl; bi < ru; bi += blocksizeIJ) {
			for(int bj = 0; bj < n; bj += blocksizeIJ) {
				int bimin = Math.min(bi + blocksizeIJ, ru);
				int bjmin = Math.min(bj + blocksizeIJ, n);
				lnnz = convertBlockSafeCast(frame, lnnz, c, bi, bj, bimin, bjmin);
			}
		}
		return lnnz;
	}

	private static long convertBlockSafeCast(final FrameBlock frame, long lnnz, final DenseBlock c, final int rl,
		final int cl, final int ru, final int cu) {
		for(int i = rl; i < ru; i++) {
			final double[] cvals = c.values(i);
			final int cpos = c.pos(i);
			for(int j = cl; j < cu; j++) {
				try {
					lnnz += (cvals[cpos + j] = frame.getDoubleNaN(i, j)) != 0 ? 1 : 0;
				}
				catch(NumberFormatException | DMLRuntimeException e) {
					lnnz += 1;
					cvals[cpos + j] = Double.NaN;
				}
			}
		}
		return lnnz;
	}

}
