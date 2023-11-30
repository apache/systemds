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
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;

public interface LibMatrixSparseToDense {
	public static final Log LOG = LogFactory.getLog(LibMatrixSparseToDense.class.getName());

	/**
	 * Convert the given matrix block to a Dense allocation.
	 * 
	 * @param r The matrix block to modify, to dense.
	 * @param k allowed.
	 */
	public static void sparseToDense(MatrixBlock r, int k) {
		// set target representation, early abort on empty blocks
		final SparseBlock a = r.sparseBlock;
		final int m = r.rlen;
		r.sparse = false;
		if(a == null)
			return;

		// allocate dense target block, but keep nnz (no need to maintain)
		if(!r.allocateDenseBlock(false))
			r.denseBlock.reset();

		final DenseBlock c = r.denseBlock;

		if(k > 1 && r.getNonZeros() > 1000000 && r.getNumRows() > 1)
			multiThreadedToDense(a, c, m, k);
		else
			singleThreadedToDense(a, c, m);

		// cleanup sparse block
		r.sparseBlock = null;
	}

	private static void singleThreadedToDense(SparseBlock a, DenseBlock c, int m) {
		for(int i = 0; i < m; i++)
			processRow(a, c, i);
	}

	private static void multiThreadedToDense(SparseBlock a, DenseBlock c, int m, int k) {
		ExecutorService pool = CommonThreadPool.get(k);

		try {

			List<Future<?>> tasks = new ArrayList<>();
			final int blkz = Math.max(1, m / k);
			for(int i = 0; i < m; i += blkz) {
				final int start = i;
				final int end = Math.min(m, i + blkz);
				tasks.add(pool.submit(() -> processRange(a, c, start, end)));
			}

			for(Future<?> f : tasks) {
				f.get();
			}
		}
		catch(Exception e) {
			throw new DMLRuntimeException("Failed parallel to dense", e);
		}
		finally {
			pool.shutdown();
		}

	}

	private static void processRange(SparseBlock a, DenseBlock c, int rl, int ru) {
		for(int i = rl; i < ru; i++)
			processRow(a, c, i);
	}

	/**
	 * Process row i
	 * 
	 * @param a Input Sparse
	 * @param c Output Dense
	 * @param i Row to process
	 */
	private static void processRow(SparseBlock a, DenseBlock c, int i) {
		if(!a.isEmpty(i)) {
			final int apos = a.pos(i);
			final int alen = a.size(i);
			final int[] aix = a.indexes(i);
			final double[] avals = a.values(i);
			final double[] cvals = c.values(i);
			final int cix = c.pos(i);
			for(int j = apos; j < apos + alen; j++)
				cvals[cix + aix[j]] = avals[j];
		}

	}

}
