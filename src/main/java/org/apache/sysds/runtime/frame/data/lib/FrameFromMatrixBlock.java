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

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.UtilFunctions;

public class FrameFromMatrixBlock {
	public static FrameBlock convertToFrameBlock(MatrixBlock mb, ValueType vt) {
		return convertToFrameBlock(mb, UtilFunctions.nCopies(mb.getNumColumns(), vt));
	}

	public static FrameBlock convertToFrameBlock(MatrixBlock mb, ValueType[] schema) {
		if(mb.isInSparseFormat())
			return convertToFrameBlockSparse(mb, schema);
		else
			return convertToFrameBlockDense(mb, schema);
	}

	private static FrameBlock convertToFrameBlockSparse(MatrixBlock mb, ValueType[] schema) {
		SparseBlock sblock = mb.getSparseBlock();
		FrameBlock frame = new FrameBlock();
		Array<?>[] columns = new Array<?>[mb.getNumColumns()];
		for(int i = 0; i < columns.length; i++)
			columns[i] = ArrayFactory.allocate(schema[i], mb.getNumRows());
		
		for(int i = 0; i < mb.getNumRows(); i++) {
			// Arrays.fill(row, null); // reset
			if(sblock != null && !sblock.isEmpty(i)) {
				int apos = sblock.pos(i);
				int alen = sblock.size(i);
				int[] aix = sblock.indexes(i);
				double[] aval = sblock.values(i);
				for(int j = apos; j < apos + alen; j++) {
					columns[aix[j]].set(i, aval[j]);
				}
			}
		}
		for(int i = 0; i < columns.length; i++)
			frame.appendColumn(columns[i]);
		return frame;
	}

	private static FrameBlock convertToFrameBlockDense(MatrixBlock mb, ValueType[] schema) {
		FrameBlock frame = new FrameBlock(schema);
		Object[] row = new Object[mb.getNumColumns()];
		int dFreq = UtilFunctions.frequency(schema, ValueType.FP64);

		if(schema.length == 1 && dFreq == 1 && mb.isAllocated()) {
			// special case double schema and single columns which
			// allows for a shallow copy since the physical representation
			// of row-major matrix and column-major frame match exactly
			frame.reset();
			frame.appendColumns(new double[][] {mb.getDenseBlockValues()});
		}
		else if(dFreq == schema.length) {
			// special case double schema (without cell-object creation,
			// col pre-allocation, and cache-friendly row-column copy)
			int m = mb.getNumRows();
			int n = mb.getNumColumns();
			double[][] c = new double[n][m];
			int blocksizeIJ = 32; // blocks of a/c+overhead in L1 cache
			if(!mb.isEmptyBlock(false)) {
				if(mb.getDenseBlock().isContiguous()) {
					double[] a = mb.getDenseBlockValues();
					for(int bi = 0; bi < m; bi += blocksizeIJ)
						for(int bj = 0; bj < n; bj += blocksizeIJ) {
							int bimin = Math.min(bi + blocksizeIJ, m);
							int bjmin = Math.min(bj + blocksizeIJ, n);
							for(int i = bi, aix = bi * n; i < bimin; i++, aix += n)
								for(int j = bj; j < bjmin; j++)
									c[j][i] = a[aix + j];
						}
				}
				else { // large dense blocks
					DenseBlock a = mb.getDenseBlock();
					for(int bi = 0; bi < m; bi += blocksizeIJ)
						for(int bj = 0; bj < n; bj += blocksizeIJ) {
							int bimin = Math.min(bi + blocksizeIJ, m);
							int bjmin = Math.min(bj + blocksizeIJ, n);
							for(int i = bi; i < bimin; i++) {
								double[] avals = a.values(i);
								int apos = a.pos(i);
								for(int j = bj; j < bjmin; j++)
									c[j][i] = avals[apos + j];
							}
						}
				}
			}
			frame.reset();
			frame.appendColumns(c);
		}
		else { // general case
			for(int i = 0; i < mb.getNumRows(); i++) {
				for(int j = 0; j < mb.getNumColumns(); j++)
					row[j] = UtilFunctions.doubleToObject(schema[j], mb.quickGetValue(i, j));

				frame.appendRow(row);
			}
		}
		return frame;
	}
}
