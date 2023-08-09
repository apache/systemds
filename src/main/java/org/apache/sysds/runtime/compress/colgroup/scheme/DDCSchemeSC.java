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

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupDDC;
import org.apache.sysds.runtime.compress.colgroup.ColGroupEmpty;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictionaryFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.utils.DoubleCountHashMap;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.Pair;

public class DDCSchemeSC extends DDCScheme {

	final private DoubleCountHashMap map;

	protected DDCSchemeSC(ColGroupDDC g) {
		super(g.getColIndices());
		if(cols.size() != 1)
			throw new DMLRuntimeException("Invalid single col scheme");
		this.lastDict = g.getDictionary();
		int unique = lastDict.getNumberOfValues(1);
		map = new DoubleCountHashMap(unique);
		for(int i = 0; i < unique; i++)
			map.increment(lastDict.getValue(i));
	}

	protected DDCSchemeSC(IColIndex cols) {
		super(cols);
		this.map = new DoubleCountHashMap(4);
	}

	@Override
	protected final Object getMap() {
		return map;
	}

	@Override
	public ICLAScheme update(MatrixBlock data, IColIndex columns) {
		validate(data, columns);
		if(data.isEmpty())
			map.increment(0.0, data.getNumRows());
		else if(data.isInSparseFormat())
			updateSparse(data, columns);
		else if(data.getDenseBlock().isContiguous())
			updateDense(data, columns);
		else
			updateGeneric(data, columns);

		return this;
	}

	private ICLAScheme updateSparse(MatrixBlock data, IColIndex columns) {
		final int col = columns.get(0);
		final int nRow = data.getNumRows();
		final SparseBlock sb = data.getSparseBlock();
		for(int i = 0; i < nRow; i++)
			map.increment(sb.get(i, col));
		return this;
	}

	private ICLAScheme updateDense(MatrixBlock data, IColIndex columns) {
		final int col = columns.get(0);
		final int nRow = data.getNumRows();
		final double[] vals = data.getDenseBlockValues();
		final int nCol = data.getNumColumns();
		final int max = nRow * nCol; // guaranteed lower than intmax.
		for(int off = col; off < max; off += nCol)
			map.increment(vals[off]);
		return this;
	}

	private ICLAScheme updateGeneric(MatrixBlock data, IColIndex columns) {
		final int col = columns.get(0);
		final int nRow = data.getNumRows();
		final DenseBlock db = data.getDenseBlock();
		for(int i = 0; i < nRow; i++) {
			final double[] c = db.values(i);
			final int off = db.pos(i) + col;
			map.increment(c[off]);
		}
		return this;
	}

	@Override
	public AColGroup encode(MatrixBlock data, IColIndex columns) {
		validate(data, columns);
		if(data.isEmpty())
			return new ColGroupEmpty(columns);
		final int nRow = data.getNumRows();

		final AMapToData d = MapToFactory.create(nRow, map.size());

		encode(data, d, cols.get(0));
		if(lastDict == null || lastDict.getNumberOfValues(columns.size()) != map.size())
			lastDict = DictionaryFactory.create(map);

		return ColGroupDDC.create(columns, lastDict, d, null);
	}

	private void encode(MatrixBlock data, AMapToData d, int col) {
		if(data.isEmpty())
			d.fill(map.getId(0.0));
		else if(data.isInSparseFormat())
			encodeSparse(data, d, col);
		else if(data.getDenseBlock().isContiguous())
			encodeDense(data, d, col);
		else
			encodeGeneric(data, d, col);
	}

	private void encodeSparse(MatrixBlock data, AMapToData d, int col) {
		final int nRow = data.getNumRows();
		final SparseBlock sb = data.getSparseBlock();
		for(int i = 0; i < nRow; i++)
			d.set(i, map.getId(sb.get(i, col)));

	}

	private void encodeDense(final MatrixBlock data, final AMapToData d, final int col) {
		final int nRow = data.getNumRows();
		final double[] vals = data.getDenseBlockValues();
		final int nCol = data.getNumColumns();
		final int max = nRow * nCol; // guaranteed lower than intmax.
		for(int i = 0, off = col; off < max; i++, off += nCol)
			d.set(i, map.getId(vals[off]));
	}

	private void encodeGeneric(MatrixBlock data, AMapToData d, int col) {
		final int nRow = data.getNumRows();
		final DenseBlock db = data.getDenseBlock();
		for(int i = 0; i < nRow; i++) {
			final double[] c = db.values(i);
			final int off = db.pos(i) + col;
			d.set(i, map.getId(c[off]));
		}
	}

	@Override
	protected Pair<ICLAScheme, AColGroup> tryUpdateAndEncode(MatrixBlock data, IColIndex columns) {
		validate(data, columns);
		if(data.isEmpty())
			return new Pair<>(this, new ColGroupEmpty(columns));
		final int nRow = data.getNumRows();

		final AMapToData d = MapToFactory.create(nRow, map.size());

		encodeAndUpdate(data, d, cols.get(0));
		if(lastDict == null || lastDict.getNumberOfValues(columns.size()) != map.size())
			lastDict = DictionaryFactory.create(map);

		return new Pair<>(this, ColGroupDDC.create(columns, lastDict, d, null));
	}

	private void encodeAndUpdate(MatrixBlock data, AMapToData d, int col) {
		final int max = d.getMaxPossible();
		if(data.isEmpty())
			d.fill(map.getId(0.0));
		else if(data.isInSparseFormat())
			encodeAndUpdateSparse(data, d, col, max);
		else if(data.getDenseBlock().isContiguous())
			encodeAndUpdateDense(data, d, col, max);
		else
			encodeAndUpdateGeneric(data, d, col, max);
	}

	private void encodeAndUpdateSparse(MatrixBlock data, AMapToData d, int col, int max) {
		final int nRow = data.getNumRows();
		final SparseBlock sb = data.getSparseBlock();
		for(int i = 0; i < nRow; i++) {
			int id = map.increment(sb.get(i, col));
			if(id >= max)
				throw new DMLCompressionException("Failed update and encode with " + max + " possible values");
			d.set(i, id);
		}

	}

	private void encodeAndUpdateDense(final MatrixBlock data, final AMapToData d, final int col, int max) {
		final int nRow = data.getNumRows();
		final double[] vals = data.getDenseBlockValues();
		final int nCol = data.getNumColumns();
		final int end = nRow * nCol; // guaranteed lower than intend.
		for(int i = 0, off = col; off < end; i++, off += nCol) {
			int id = map.increment(vals[off]);
			if(id >= max)
				throw new DMLCompressionException("Failed update and encode with " + max + " possible values");
			d.set(i, id);
		}
	}

	private void encodeAndUpdateGeneric(MatrixBlock data, AMapToData d, int col, int max) {
		final int nRow = data.getNumRows();
		final DenseBlock db = data.getDenseBlock();
		for(int i = 0; i < nRow; i++) {
			final double[] c = db.values(i);
			final int off = db.pos(i) + col;
			int id = map.increment(c[off]);
			if(id >= max)
				throw new DMLCompressionException("Failed update and encode with " + max + " possible values");
			d.set(i, id);
		}
	}
}
