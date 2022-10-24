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
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class ConstScheme implements ICLAScheme {

	/** The instance of a constant column group that in all cases here would be returned to be the same */
	final ColGroupConst g;

	protected ConstScheme(ColGroupConst g) {
		this.g = g;
	}

	public static ICLAScheme create(ColGroupConst g) {
		return new ConstScheme(g);
	}

	@Override
	public AColGroup encode(MatrixBlock data) {
		return encode(data, g.getColIndices(), g.getValues());
	}

	@Override
	public AColGroup encode(MatrixBlock data, int[] columns) {
		if(columns.length != g.getColIndices().length)
			throw new IllegalArgumentException("Invalid columns to encode");
		return encode(data, columns, g.getValues());
	}

	private AColGroup encode(final MatrixBlock data, final int[] cols, final double[] values) {
		final int nCol = data.getNumColumns();
		final int nRow = data.getNumRows();
		if(nCol < cols[cols.length - 1]) {
			LOG.warn("Invalid to encode matrix with less columns than encode scheme max column");
			return null;
		}
		else if(data.isEmpty()) {
			LOG.warn("Invalid to encode an empty matrix into constant column group");
			return null; // Invalid to encode this.
		}
		else if(data.isInSparseFormat())
			return encodeSparse(data, cols, values, nRow, nCol);
		else if(data.getDenseBlock().isContiguous())
			return encodeDense(data, cols, values, nRow, nCol);
		else
			return encodeGeneric(data, cols, values, nRow, nCol);
	}

	private AColGroup encodeDense(final MatrixBlock data, final int[] cols, final double[] values, final int nRow,
		final int nCol) {
		final double[] dv = data.getDenseBlockValues();
		for(int r = 0; r < nRow; r++) {
			final int off = r * nCol;
			for(int ci = 0; ci < cols.length; ci++)
				if(dv[off + cols[ci]] != values[ci])
					return null;
		}
		return g;
	}

	private AColGroup encodeSparse(final MatrixBlock data, final int[] cols, final double[] values, final int nRow,
		final int nCol) {
		SparseBlock sb = data.getSparseBlock();
		for(int r = 0; r < nRow; r++) {
			if(sb.isEmpty(r))
				return null;

			final int apos = sb.pos(r);
			final int alen = apos + sb.size(r);
			final double[] aval = sb.values(r);
			final int[] aix = sb.indexes(r);
			int p = 0; // pointer into cols;
			while(p < cols.length && values[p] == 0.0)
				p++;
			for(int j = apos; j < alen && p < cols.length; j++) {
				if(aix[j] == cols[p]) {
					if(aval[j] != values[p])
						return null;
					p++;
					while(p < cols.length && values[p] == 0.0)
						p++;
				}
				else if(aix[j] > cols[p])
					return null; // not matching
			}
		}
		return g;
	}

	private AColGroup encodeGeneric(final MatrixBlock data, final int[] cols, final double[] values, final int nRow,
		final int nCol) {
		for(int r = 0; r < nRow; r++)
			for(int ci = 0; ci < cols.length; ci++)
				if(data.quickGetValue(r, cols[ci]) != values[ci])
					return null;
		return g;
	}
}
