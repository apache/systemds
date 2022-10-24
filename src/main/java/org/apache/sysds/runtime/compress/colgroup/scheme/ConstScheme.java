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

package org.apache.sysds.runtime.compress.colgroup.scheme;

import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupConst;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class ConstScheme implements ICLAScheme {

	final int[] cols;
	final double[] values;

	protected ConstScheme(int[] cols, double[] values) {
		this.cols = cols;
		this.values = values;
	}

	public static ICLAScheme create(ColGroupConst g) {
		return new ConstScheme(g.getColIndices(), g.getValues());
	}

	@Override
	public AColGroup encode(MatrixBlock data) {
		return encode(data, cols, values);
	}

	@Override
	public AColGroup encode(MatrixBlock data, int[] columns) {
		if(columns.length != cols.length)
			throw new IllegalArgumentException("Invalid columns to encode");
		return encode(data, columns, values);
	}

	private static AColGroup encode(MatrixBlock data, int[] cols, double[] values) {
		// Check if the column is all constant in the column that is specified.
		if(data.isEmpty()) {
			LOG.warn("Invalid to encode an empty matrix into constant column group");
			return null; // Invalid to encode this.
		}
		else if(data.isInSparseFormat()) {
			// unlucky inefficient lookup in each row.
			LOG.warn("Not implemented Sparse encoding of constant columns.");
			return null;
		}
		else if(data.getDenseBlock().isContiguous()) {
			// Nice
			final double[] dv = data.getDenseBlockValues();
			final int nCol = data.getNumColumns();
			final int nRow = data.getNumRows();
			for(int r = 0; r < nRow; r++) {
				final int off = r * nCol;
				for(int ci = 0; ci < cols.length; ci++) {
					if(dv[off + cols[ci]] != values[ci])
						return null;
				}
			}
			return ColGroupConst.create(cols, values);
		}
		else {
			// not implemented
			LOG.warn("Not implemented application of compression scheme for non continuous dense blocks");
			return null;
		}
	}
}
