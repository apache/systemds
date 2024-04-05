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
import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;

public interface MatrixBlockFromFrame {
	public static final Log LOG = LogFactory.getLog(MatrixBlockFromFrame.class.getName());

	public static final int blocksizeIJ = 32;

	/**
	 * Converts a frame block with arbitrary schema into a matrix block. Since matrix block only supports value type
	 * double, we do a best effort conversion of non-double types which might result in errors for non-numerical data.
	 *
	 * @param frame frame block
	 * @param k     parallelization degree
	 * @return matrix block
	 */
	public static MatrixBlock convertToMatrixBlock(FrameBlock frame, int k) {
		final int m = frame.getNumRows();
		final int n = frame.getNumColumns();
		final MatrixBlock mb = new MatrixBlock(m, n, false);
		mb.allocateDenseBlock();
		if(k == -1)
			k = InfrastructureAnalyzer.getLocalParallelism();

		long nnz = 0;
		if(k == 1)
			nnz = convert(frame, mb, n, 0, m);
		else
			nnz = convertParallel(frame, mb, m, n, k);

		mb.setNonZeros(nnz);

		mb.examSparsity();
		return mb;
	}

	private static long convert(FrameBlock frame, MatrixBlock mb, int n, int rl, int ru) {
		if(mb.getDenseBlock().isContiguous())
			return convertContiguous(frame, mb, n, rl, ru);
		else
			return convertGeneric(frame, mb, n, rl, ru);
	}

	private static long convertParallel(FrameBlock frame, MatrixBlock mb, int m, int n, int k){
		ExecutorService pool = CommonThreadPool.get(k);
		try{
			List<Future<Long>> tasks = new ArrayList<>();
			final int blkz = Math.max(m / k, 1000);

			for( int i = 0; i < m; i+= blkz){
				final int start = i; 
				final int end = Math.min(i + blkz, m);
				tasks.add(pool.submit(() -> convert(frame, mb, n, start, end)));
			}

			long nnz = 0;
			for( Future<Long> t : tasks)
				nnz += t.get();
			return nnz;
		}
		catch(Exception e){
			throw new RuntimeException(e);
		}
		finally{
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
				for(int i = bi, aix = bi * n; i < bimin; i++, aix += n)
					for(int j = bj; j < bjmin; j++)
						lnnz += (c[aix + j] = frame.getDoubleNaN(i, j)) != 0 ? 1 : 0;
			}
		}
		return lnnz;
	}

	private static long convertGeneric(final FrameBlock frame, final MatrixBlock mb, final int n, final int rl, final int ru) {
		long lnnz = 0;
		final DenseBlock c = mb.getDenseBlock();
		for(int bi = rl; bi < ru; bi += blocksizeIJ) {
			for(int bj = 0; bj < n; bj += blocksizeIJ) {
				int bimin = Math.min(bi + blocksizeIJ, ru);
				int bjmin = Math.min(bj + blocksizeIJ, n);
				for(int i = bi; i < bimin; i++) {
					double[] cvals = c.values(i);
					int cpos = c.pos(i);
					for(int j = bj; j < bjmin; j++)
						lnnz += (cvals[cpos + j] = frame.getDoubleNaN(i, j)) != 0 ? 1 : 0;
				}
			}
		}
		return lnnz;
	}
}
