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
import org.apache.sysds.runtime.compress.colgroup.ColGroupEmpty;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class EmptyScheme implements ICLAScheme {
	/** The instance of a empty column group that in all cases here would be returned to be the same */
	final ColGroupEmpty g;

	protected EmptyScheme(ColGroupEmpty g) {
		this.g = g;
	}

	public static EmptyScheme create(ColGroupEmpty g) {
		return new EmptyScheme(g);
	}

	@Override
	public AColGroup encode(MatrixBlock data) {
		return encode(data, g.getColIndices());
	}

	@Override
	public AColGroup encode(MatrixBlock data, IColIndex  columns) {

		if(columns.size() != g.getColIndices().size())
			throw new IllegalArgumentException("Invalid columns to encode");
		final int nCol = data.getNumColumns();
		final int nRow = data.getNumRows();
		if(nCol < columns.get(columns.size() - 1)) {
			LOG.warn("Invalid to encode matrix with less columns than encode scheme max column");
			return null;
		}
		else if(data.isEmpty()) 
			return returnG(columns);
		else if(data.isInSparseFormat())
			return encodeSparse(data, columns, nRow, nCol);
		else if(data.getDenseBlock().isContiguous())
			return encodeDense(data, columns, nRow, nCol);
		else
			return encodeGeneric(data, columns, nRow, nCol);
	}

	private AColGroup encodeDense(final MatrixBlock data, final IColIndex  cols, final int nRow, final int nCol) {
		final double[] dv = data.getDenseBlockValues();
		for(int r = 0; r < nRow; r++) {
			final int off = r * nCol;
			for(int ci = 0; ci < cols.size(); ci++)
				if(dv[off + cols.get(ci)] != 0.0)
					return null;
		}
		return g;
	}

	private AColGroup encodeSparse(final MatrixBlock data, final IColIndex  cols, final int nRow, final int nCol) {
		SparseBlock sb = data.getSparseBlock();
		for(int r = 0; r < nRow; r++) {
			if(sb.isEmpty(r))
				continue; // great!

			final int apos = sb.pos(r);
			final int alen = apos + sb.size(r);
			final int[] aix = sb.indexes(r);
			int p = 0; // pointer into cols;
			for(int j = apos; j < alen ; j++) {
				while(p < cols.size() && cols.get(p) < aix[j])
					p++;
				if(p < cols.size() && aix[j] == cols.get(p))
					return null;

				if(p >= cols.size())
					continue;
			}
		}
		return returnG(cols);
	}

	private AColGroup encodeGeneric(final MatrixBlock data, final IColIndex  cols, final int nRow, final int nCol) {
		for(int r = 0; r < nRow; r++)
			for(int ci = 0; ci < cols.size(); ci++)
				if(data.quickGetValue(r, cols.get(ci)) != 0.0)
					return null;
		return returnG(cols);
	}

	private AColGroup returnG(IColIndex columns) {
		if(columns == g.getColIndices())
			return g; // great!
		else
			return new ColGroupEmpty(columns);
	}

}
