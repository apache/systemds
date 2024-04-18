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
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class EmptyScheme extends ACLAScheme {

	public EmptyScheme(IColIndex idx) {
		super(idx);
	}

	public static EmptyScheme create(ColGroupEmpty g) {
		return new EmptyScheme(g.getColIndices());
	}

	@Override
	protected ICLAScheme updateV(MatrixBlock data, IColIndex columns) {
		if(data.isEmpty()) // all good
			return this;
		else if(data.isInSparseFormat())
			return updateSparse(data, columns);
		else if(data.getDenseBlock().isContiguous())
			return updateDense(data, columns);
		else
			return updateGeneric(data, columns);
	}

	private ICLAScheme updateGeneric(MatrixBlock data, IColIndex columns) {
		final int nRow = data.getNumRows();
		final int nColScheme = columns.size();
		// should be optimized.
		for(int r = 0; r < nRow; r++) {
			for(int c = 0; c < nColScheme; c++) {
				double v = data.get(r, columns.get(c));
				if(v != 0)
					return updateToHigherScheme(data, columns);
			}
		}
		return this;
	}

	private ICLAScheme updateDense(MatrixBlock data, IColIndex columns) {
		final int nRow = data.getNumRows();
		final int nCol = data.getNumColumns();
		final int nColScheme = columns.size();
		final double[] values = data.getDenseBlockValues();

		for(int r = 0; r < nRow; r++) {
			int off = r * nCol;
			for(int c = 0; c < nColScheme; c++) {
				double v = values[off + columns.get(c)];
				if(v != 0)
					return updateToHigherScheme(data, columns); // well always do this if not empty.
			}
		}
		return this;
	}

	private ICLAScheme updateSparse(MatrixBlock data, IColIndex columns) {
		final SparseBlock sb = data.getSparseBlock();
		final int nRow = data.getNumRows();
		if(columns.size() == 1) {
			final int col = columns.get(0);
			for(int i = 0; i < nRow; i++) {
				if(sb.get(i, col) == 0.0)
					return updateToHigherScheme(data, columns);
			}
		}
		else if(columns.size() * 2 > data.getNumColumns()) { // if we have many many columns
			for(int i = 0; i < nRow; i++) {
				int apos = sb.pos(i);
				final int alen = sb.size(i) + apos;
				final int[] aix = sb.indexes(i);
				int offC = 0;
				while(apos < alen || offC < columns.size()) {
					int va = aix[apos];
					int vb = columns.get(offC);
					if(va < vb)
						apos++;
					else if(vb < va)
						offC++;
					else if(va == vb)
						return updateToHigherScheme(data, columns);
				}
			}
		}
		else {
			for(int i = 0; i < nRow; i++) {
				for(int j = 0; i < columns.size(); j++) {
					final int col = columns.get(j);
					if(sb.get(i, col) == 0.0)
						return updateToHigherScheme(data, columns);
				}
			}
		}
		return this;
	}

	private ICLAScheme updateToHigherScheme(MatrixBlock data, IColIndex columns) {
		// try with const
		double[] vals = new double[cols.size()];
		for(int c = 0; c < cols.size(); c++)
			vals[c] = data.get(0, c);

		return ConstScheme.create(columns, vals).update(data, columns);
	}

	private ICLAScheme updateToHigherSchemeT(MatrixBlock data, IColIndex columns) {
		// try with const
		double[] vals = new double[cols.size()];
		for(int c = 0; c < cols.size(); c++)
			vals[c] = data.get(c, 0);

		return ConstScheme.create(columns, vals).updateT(data, columns);
	}

	@Override
	protected AColGroup encodeV(MatrixBlock data, IColIndex columns) {
		return new ColGroupEmpty(columns);
	}

	@Override
	protected AColGroup encodeVT(MatrixBlock data, IColIndex columns) {
		return new ColGroupEmpty(columns);
	}

	@Override
	protected ICLAScheme updateVT(MatrixBlock data, IColIndex columns) {
		if(data.isEmpty()) // all good
			return this;
		else if(data.isInSparseFormat())
			return updateSparseT(data, columns);
		else
			return updateDenseT(data, columns);
	}

	private ICLAScheme updateDenseT(MatrixBlock data, IColIndex columns) {
		final DenseBlock db = data.getDenseBlock();
		for(int i = 0; i < columns.size(); i++) {
			final int col = columns.get(i);
			final double[] vals = db.values(col);
			final int nCol = data.getNumColumns();
			final int start = db.pos(col);
			for(int off = db.pos(col); i < nCol + start; off++)
				if(vals[off] != 0)
					return updateToHigherSchemeT(data, columns);
		}
		return this;
	}

	private ICLAScheme updateSparseT(MatrixBlock data, IColIndex columns) {
		final SparseBlock sb = data.getSparseBlock();

		for(int i = 0; i < columns.size(); i++)
			if(!sb.isEmpty(columns.get(i)))
				return updateToHigherSchemeT(data, columns);

		return this;
	}

	@Override
	public final String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		sb.append(" Cols: ");
		sb.append(cols);
		return sb.toString();
	}

	@Override
	public EmptyScheme clone() {
		return new EmptyScheme(cols);
	}

}
