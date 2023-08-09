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
import org.apache.sysds.runtime.compress.colgroup.ASDC;
import org.apache.sysds.runtime.compress.colgroup.ASDCZero;
import org.apache.sysds.runtime.compress.colgroup.ColGroupDDC;
import org.apache.sysds.runtime.compress.colgroup.ColGroupEmpty;
import org.apache.sysds.runtime.compress.colgroup.ColGroupSDC;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictionaryFactory;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.apache.sysds.runtime.compress.utils.DoubleCountHashMap;
import org.apache.sysds.runtime.compress.utils.IntArrayList;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class SDCSchemeSC extends SDCScheme {

	final double def;
	final private DoubleCountHashMap map;

	protected SDCSchemeSC(ASDC g) {
		this(g.getColIndices(), g.getCommon()[0], g.getDictionary());
	}

	protected SDCSchemeSC(ASDCZero g) {
		this(g.getColIndices(), 0, g.getDictionary());
	}

	private SDCSchemeSC(IColIndex cols, double def, IDictionary lastDict) {
		super(cols);
		this.def = def;
		this.lastDict = lastDict;
		int unique = lastDict.getNumberOfValues(1);
		map = new DoubleCountHashMap(unique);

		for(int i = 0; i < unique; i++)
			map.increment(lastDict.getValue(i));
	}

	@Override
	public AColGroup encode(MatrixBlock data, IColIndex columns) {

		validate(data, columns);
		final int nRow = data.getNumRows();
		if(data.isEmpty())
			return new ColGroupEmpty(columns);

		// final AMapToData d = MapToFactory.create(nRow, map.size());
		final IntArrayList offs = new IntArrayList();
		AMapToData d = encode(data, offs, cols.get(0));
		if(lastDict == null || lastDict.getNumberOfValues(columns.size()) != map.size())
			lastDict = DictionaryFactory.create(map);

		if(offs.size() == 0) {
			return ColGroupDDC.create(columns, lastDict, d, null);
		}
		else {
			final AOffset off = OffsetFactory.createOffset(offs);
			return ColGroupSDC.create(columns, nRow, lastDict, new double[] {def}, off, d, null);
		}
	}

	private AMapToData encode(MatrixBlock data, IntArrayList off, int col) {

		if(data.isInSparseFormat())
			return encodeSparse(data, off, col);
		else if(data.getDenseBlock().isContiguous())
			return encodeDense(data, off, col);
		else
			return encodeGeneric(data, off, col);
	}

	private AMapToData encodeSparse(MatrixBlock data, IntArrayList off, int col) {
		final int nRow = data.getNumRows();
		final SparseBlock sb = data.getSparseBlock();
		// full iteration
		for(int i = 0; i < nRow; i++)
			if(sb.get(i, col) != def)
				off.appendValue(i);

		// Only cells with non default values.
		AMapToData d = MapToFactory.create(off.size(), map.size());
		for(int i = 0; i < off.size(); i++) {
			int r = off.get(i);
			d.set(i, map.getId(sb.get(r, col)));
		}
		return d;
	}

	private AMapToData encodeDense(MatrixBlock data, IntArrayList off, int col) {
		final int nRow = data.getNumRows();
		final double[] vals = data.getDenseBlockValues();
		final int nCol = data.getNumColumns();
		final int max = nRow * nCol; // guaranteed lower than intmax.
		// full iteration
		for(int i = 0, o = col; o < max; i++, o += nCol) {
			if(vals[o] != def)
				off.appendValue(i);
		}

		// Only cells with non default values.
		AMapToData d = MapToFactory.create(off.size(), map.size());
		for(int i = 0; i < off.size(); i++) {
			int o = off.get(i) * nCol + col;
			d.set(i, map.getId(vals[o]));
		}
		return d;
	}

	private AMapToData encodeGeneric(MatrixBlock data, IntArrayList off, int col) {
		final int nRow = data.getNumRows();
		final DenseBlock db = data.getDenseBlock();

		// full iteration
		for(int i = 0; i < nRow; i++) {
			final double[] c = db.values(i);
			final int o = db.pos(i) + col;
			if(c[o] != def)
				off.appendValue(i);
		}

		// Only cells with non default values.
		AMapToData d = MapToFactory.create(off.size(), map.size());
		for(int i = 0; i < off.size(); i++) {
			final int of = off.get(i);
			final int o = db.pos(of) + col;
			d.set(i, map.getId(db.values(of)[o]));
		}
		return d;
	}

	@Override
	public ICLAScheme update(MatrixBlock data, IColIndex columns) {
		validate(data, columns);

		final int col = columns.get(0);
		if(data.isEmpty()) {
			if(def != 0.0)
				map.increment(0.0, data.getNumRows());
		}
		else if(data.isInSparseFormat())
			updateSparse(data, col);
		else if(data.getDenseBlock().isContiguous())
			updateDense(data, col);
		else
			updateGeneric(data, col);
		return this;
	}

	private void updateSparse(MatrixBlock data, int col) {
		final int nRow = data.getNumRows();
		final SparseBlock sb = data.getSparseBlock();
		for(int i = 0; i < nRow; i++) {
			final double v = sb.get(i, col);
			if(v != def)
				map.increment(v);
		}
	}

	private void updateDense(MatrixBlock data, int col) {
		final int nRow = data.getNumRows();
		final double[] vals = data.getDenseBlockValues();
		final int nCol = data.getNumColumns();
		final int max = nRow * nCol; // guaranteed lower than intmax.
		for(int off = col; off < max; off += nCol) {
			final double v = vals[off];
			if(v != def)
				map.increment(v);
		}

	}

	private void updateGeneric(MatrixBlock data, int col) {
		final int nRow = data.getNumRows();
		final DenseBlock db = data.getDenseBlock();
		for(int i = 0; i < nRow; i++) {
			final double[] c = db.values(i);
			final int off = db.pos(i) + col;
			final double v = c[off];
			if(v != def)
				map.increment(v);
		}
	}

	protected Object getDef() {
		return def;
	}

	protected Object getMap() {
		return map;
	}

}
