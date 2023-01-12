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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public interface MatrixBlockFromFrame {
	public static final Log LOG = LogFactory.getLog(MatrixBlockFromFrame.class.getName());

	public static final int blocksizeIJ = 32;

	/**
	 * Converts a frame block with arbitrary schema into a matrix block. Since matrix block only supports value type
	 * double, we do a best effort conversion of non-double types which might result in errors for non-numerical data.
	 *
	 * @param frame frame block
	 * @return matrix block
	 */
	public static MatrixBlock convertToMatrixBlock(FrameBlock frame) {
		final int m = frame.getNumRows();
		final int n = frame.getNumColumns();
		final MatrixBlock mb = new MatrixBlock(m, n, false);
		mb.allocateDenseBlock();

		if(mb.getDenseBlock().isContiguous())
			convertContiguous(frame, mb, m, n);
		else
			convertGeneric(frame, mb, m, n);

		mb.examSparsity();
		return mb;
	}

	private static void convertContiguous(final FrameBlock frame, final MatrixBlock mb, final int m, final int n) {
		long lnnz = 0;
		double[] c = mb.getDenseBlockValues();
		for(int bi = 0; bi < m; bi += blocksizeIJ) {
			for(int bj = 0; bj < n; bj += blocksizeIJ) {
				int bimin = Math.min(bi + blocksizeIJ, m);
				int bjmin = Math.min(bj + blocksizeIJ, n);
				for(int i = bi, aix = bi * n; i < bimin; i++, aix += n)
					for(int j = bj; j < bjmin; j++)
						lnnz += (c[aix + j] = frame.getDoubleNaN(i, j)) != 0 ? 1 : 0;
			}
		}
		mb.setNonZeros(lnnz);
	}

	private static void convertGeneric(final FrameBlock frame, final MatrixBlock mb, final int m, final int n) {
		long lnnz = 0;
		final DenseBlock c = mb.getDenseBlock();
		for(int bi = 0; bi < m; bi += blocksizeIJ) {
			for(int bj = 0; bj < n; bj += blocksizeIJ) {
				int bimin = Math.min(bi + blocksizeIJ, m);
				int bjmin = Math.min(bj + blocksizeIJ, n);
				for(int i = bi; i < bimin; i++) {
					double[] cvals = c.values(i);
					int cpos = c.pos(i);
					for(int j = bj; j < bjmin; j++)
						lnnz += (cvals[cpos + j] = frame.getDoubleNaN(i, j)) != 0 ? 1 : 0;
				}
			}
		}
		mb.setNonZeros(lnnz);
	}
}
