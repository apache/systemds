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

import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ColGroupDDCLZW;
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

public class DDCLZWSchemeSC extends DDCLZWScheme {

	// TODO: Dies ist eine Vorläufige Version - Code hauptsächlich wie in DDCSchemeSC
	// Prüfen, ob abstrakte Vorgeschaltete Klasse möglich ist oder speichern des DDCSchemeSC als Attribut

	final private DoubleCountHashMap map;

	private DDCLZWSchemeSC(IColIndex cols, DoubleCountHashMap map) {
		super(cols);
		this.map = map;
	}

	protected DDCLZWSchemeSC(ColGroupDDCLZW g) {
		super(g.getColIndices());
		this.lastDict = g.getDictionary();
		int unique = lastDict.getNumberOfValues(1);
		map = new DoubleCountHashMap(unique);
		for(int i = 0; i < unique; i++)
			map.increment(lastDict.getValue(i));
	}

	protected DDCLZWSchemeSC(IColIndex cols) {
		super(cols);
		this.map = new DoubleCountHashMap(4);
	}

	@Override
	protected AColGroup encodeV(MatrixBlock data, IColIndex columns) {
		if(data.isEmpty())
			return new ColGroupEmpty(columns);
		final int nRow = data.getNumRows();

		final AMapToData d = MapToFactory.create(nRow, map.size());

		encode(data, d, cols.get(0));
		if(lastDict == null || lastDict.getNumberOfValues(columns.size()) != map.size())
			lastDict = DictionaryFactory.create(map);

		return ColGroupDDCLZW.create(columns, lastDict, d, null);
	}

	private void encodeSparse(MatrixBlock data, AMapToData d, int col) {
		final int nRow = data.getNumRows();
		final SparseBlock sb = data.getSparseBlock();
		for(int i = 0; i < nRow; i++)
			d.set(i, map.getId(sb.get(i, col)));

	}

	private void encode(MatrixBlock data, AMapToData d, int col) {
		if(data.isInSparseFormat())
			encodeSparse(data, d, col);
		else if(data.getDenseBlock().isContiguous())
			encodeDense(data, d, col);
		else
			encodeGeneric(data, d, col);
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
	protected AColGroup encodeVT(MatrixBlock data, IColIndex columns) {
		if(data.isEmpty())
			return new ColGroupEmpty(columns);
		final int nRow = data.getNumColumns();

		final AMapToData d = MapToFactory.create(nRow, map.size());

		encodeT(data, d, cols.get(0));
		if(lastDict == null || lastDict.getNumberOfValues(columns.size()) != map.size())
			lastDict = DictionaryFactory.create(map);

		return ColGroupDDCLZW.create(columns, lastDict, d, null);
	}

	private void encodeT(MatrixBlock data, AMapToData d, int col) {
		if(data.isInSparseFormat())
			encodeSparseT(data, d, col);
		else
			encodeDenseT(data, d, col);
	}

	private void encodeSparseT(MatrixBlock data, AMapToData d, int col) {
		final SparseBlock sb = data.getSparseBlock();
		d.fill(map.getId(0.0));
		if(!sb.isEmpty(col)) {
			int apos = sb.pos(col);
			final int[] aix = sb.indexes(col);
			final int alen = sb.size(col) + apos;
			final double[] aval = sb.values(col);
			while(apos < alen) {
				final double v = aval[apos];
				final int idx = aix[apos++];
				d.set(idx, map.getId(v));
			}
		}
	}

	private void encodeDenseT(MatrixBlock data, AMapToData d, int col) {
		final DenseBlock db = data.getDenseBlock();
		final double[] vals = db.values(col);
		final int nCol = data.getNumColumns();
		for(int i = 0, off = db.pos(col); i < nCol; i++, off++)
			d.set(i, map.getId(vals[off]));
	}

	@Override
	protected ICLAScheme updateV(MatrixBlock data, IColIndex columns) {
		if(data.isEmpty())
			map.increment(0.0, data.getNumRows());
		else if(data.isInSparseFormat())
			updateSparse(data, columns.get(0));
		else if(data.getDenseBlock().isContiguous())
			updateDense(data, columns.get(0));
		else
			updateGeneric(data, columns.get(0));

		return this;
	}

	private ICLAScheme updateSparse(MatrixBlock data, int col) {
		final int nRow = data.getNumRows();
		final SparseBlock sb = data.getSparseBlock();
		for(int i = 0; i < nRow; i++)
			map.increment(sb.get(i, col));
		return this;
	}

	private ICLAScheme updateDense(MatrixBlock data, int col) {

		final int nRow = data.getNumRows();
		final double[] vals = data.getDenseBlockValues();
		final int nCol = data.getNumColumns();
		final int max = nRow * nCol; // guaranteed lower than intmax.
		for(int off = col; off < max; off += nCol)
			map.increment(vals[off]);
		return this;
	}

	private ICLAScheme updateGeneric(MatrixBlock data, int col) {
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
	protected ICLAScheme updateVT(MatrixBlock data, IColIndex columns) {
		if(data.isEmpty())
			map.increment(0.0, data.getNumColumns());
		else if(data.isInSparseFormat())
			updateSparseT(data, columns.get(0));
		else // dense and generic can be handled together if transposed
			updateDenseT(data, columns.get(0));

		return this;
	}

	private void updateDenseT(MatrixBlock data, int col) {
		final DenseBlock db = data.getDenseBlock();
		final double[] vals = db.values(col);
		final int nCol = data.getNumColumns();
		for(int i = 0, off = db.pos(col); i < nCol; i++, off++)
			map.increment(vals[off]);
	}

	private void updateSparseT(MatrixBlock data, int col) {
		final SparseBlock sb = data.getSparseBlock();

		if(!sb.isEmpty(col)) {
			int apos = sb.pos(col);
			final int alen = sb.size(col) + apos;
			final double[] aval = sb.values(col);
			map.increment(0.0, alen - apos);
			while(apos < alen)
				map.increment(aval[apos++]);
		}
		else
			map.increment(0.0, data.getNumColumns());

	}

	@Override
	public DDCLZWSchemeSC clone() {
		return new DDCLZWSchemeSC(cols, map.clone());
	}

	@Override
	protected final Object getMap() {
		return map;
	}

	// TODO: zwingend erforderlich?
	@Override
	protected Pair<ICLAScheme, AColGroup> tryUpdateAndEncode(MatrixBlock data, IColIndex columns) {
		if(data.isEmpty()) {
			map.increment(0.0, data.getNumRows());
			return new Pair<>(this, new ColGroupEmpty(columns));
		}
		final int nRow = data.getNumRows();

		final AMapToData d = MapToFactory.create(nRow, map.size());

		encodeAndUpdate(data, d, cols.get(0));
		if(lastDict == null || lastDict.getNumberOfValues(columns.size()) != map.size())
			lastDict = DictionaryFactory.create(map);

		return new Pair<>(this, ColGroupDDCLZW.create(columns, lastDict, d, null));
	}

	private void encodeAndUpdate(MatrixBlock data, AMapToData d, int col) {
		final int max = d.getUpperBoundValue();
		if(data.isInSparseFormat())
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
			if(id > max)
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
			if(id > max)
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
			if(id > max)
				throw new DMLCompressionException("Failed update and encode with " + max + " possible values");
			d.set(i, id);
		}
	}

	@Override
	protected Pair<ICLAScheme, AColGroup> tryUpdateAndEncodeT(MatrixBlock data, IColIndex columns) {
		if(data.isEmpty())
			return new Pair<>(this, new ColGroupEmpty(columns));
		final int nRow = data.getNumColumns();

		final AMapToData d = MapToFactory.create(nRow, map.size());

		encodeAndUpdateT(data, d, cols.get(0));
		if(lastDict == null || lastDict.getNumberOfValues(columns.size()) != map.size())
			lastDict = DictionaryFactory.create(map);

		return new Pair<>(this, ColGroupDDCLZW.create(columns, lastDict, d, null));
	}

	private void encodeAndUpdateT(MatrixBlock data, AMapToData d, int col) {
		if(data.isInSparseFormat())
			encodeAndUpdateSparseT(data, d, col);
		else
			encodeAndUpdateDenseT(data, d, col);
	}

	private void encodeAndUpdateSparseT(MatrixBlock data, AMapToData d, int col) {
		final SparseBlock sb = data.getSparseBlock();
		if(!sb.isEmpty(col)) {
			int apos = sb.pos(col);
			final int[] aix = sb.indexes(col);
			final int alen = sb.size(col) + apos;
			d.fill(map.increment(0.0, data.getNumColumns() - alen - apos));
			final double[] aval = sb.values(col);
			while(apos < alen) {
				final double v = aval[apos];
				final int idx = aix[apos++];
				d.set(idx, map.increment(v));
			}
		}
		else
			d.fill(map.increment(0.0, data.getNumColumns()));
	}

	private void encodeAndUpdateDenseT(MatrixBlock data, AMapToData d, int col) {
		final DenseBlock db = data.getDenseBlock();
		final double[] vals = db.values(col);
		final int nCol = data.getNumColumns();
		for(int i = 0, off = db.pos(col); i < nCol; i++, off++)
			d.set(i, map.increment(vals[off]));
	}

}
