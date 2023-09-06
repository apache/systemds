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

import java.lang.ref.SoftReference;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.ASDC;
import org.apache.sysds.runtime.compress.colgroup.ASDCZero;
import org.apache.sysds.runtime.compress.colgroup.ColGroupConst;
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
import org.apache.sysds.runtime.compress.utils.Util;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.Pair;

public class SDCSchemeSC extends SDCScheme {

	final double def;
	final private DoubleCountHashMap map;

	private static SoftReference<ThreadLocal<Pair<IntArrayList, IntArrayList>>> cachedArrays;

	public SDCSchemeSC(IColIndex cols, DoubleCountHashMap map, double def) {
		super(cols);
		this.map = map;
		this.def = def;
	}

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

	protected Object getDef() {
		return def;
	}

	protected Object getMap() {
		return map;
	}

	@Override
	protected AColGroup encodeV(MatrixBlock data, IColIndex columns) {
		if(data.isEmpty())
			return new ColGroupEmpty(columns);
		final int nRow = data.getNumRows();
		final IntArrayList offs = new IntArrayList();
		AMapToData d = encode(data, offs, cols.get(0));
		return finalizeEncode(data, offs, d, columns, nRow);
	}

	private AColGroup finalizeEncode(MatrixBlock data, IntArrayList offs, AMapToData d, IColIndex columns, int nRow) {
		allocateDictionary();

		if(offs.size() == 0) {
			return ColGroupConst.create(columns, def);
		}
		else {
			final AOffset off = OffsetFactory.createOffset(offs);
			return ColGroupSDC.create(columns, nRow, lastDict, new double[] {def}, off, d, null);
		}
	}

	private void allocateDictionary() {
		if(lastDict == null || lastDict.getNumberOfValues(this.cols.size()) != map.size())
			lastDict = DictionaryFactory.create(map);
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
			if(!Util.eq(sb.get(i, col), def)) {
				off.appendValue(i);
			}

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
		for(int i = 0, o = col; o < max; i++, o += nCol)
			if(!Util.eq(vals[o], def))
				off.appendValue(i);

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
			if(!Util.eq(c[o], def))
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
	protected ICLAScheme updateV(MatrixBlock data, IColIndex columns) {

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
			if(!Util.eq(v, def))
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
			if(!Util.eq(v, def))
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
			if(!Util.eq(v, def))
				map.increment(v);
		}
	}

	@Override
	protected Pair<ICLAScheme, AColGroup> tryUpdateAndEncode(MatrixBlock data, IColIndex columns) {
		if(data.isEmpty())
			return new Pair<>(this, new ColGroupEmpty(columns));
		final int nRow = data.getNumRows();

		Pair<IntArrayList, AMapToData> e = encodeAndUpdate(data, cols.get(0));
		if(lastDict == null || lastDict.getNumberOfValues(columns.size()) != map.size())
			lastDict = DictionaryFactory.create(map);

		final AOffset off = OffsetFactory.createOffset(e.getKey());
		AColGroup g = ColGroupSDC.create(columns, nRow, lastDict, //
			new double[] {def}, off, e.getValue(), null);
		return new Pair<>(this, g);
	}

	private Pair<IntArrayList, AMapToData> encodeAndUpdate(MatrixBlock data, int col) {
		if(data.isInSparseFormat())
			return encodeAndUpdateSparse(data, col);
		else if(data.getDenseBlock().isContiguous())
			return encodeAndUpdateDense(data, col);
		else
			return encodeAndUpdateGeneric(data, col);
	}

	private Pair<IntArrayList, AMapToData> encodeAndUpdateSparse(MatrixBlock data, int col) {
		final int nRow = data.getNumRows();
		final SparseBlock sb = data.getSparseBlock();
		IntArrayList off = getCachedArray(0);
		IntArrayList val = getCachedArray(1);

		// full iteration
		for(int i = 0; i < nRow; i++) {
			double v = sb.get(i, col);
			if(!Util.eq(v, def)) {
				off.appendValue(i);
				val.appendValue(map.increment(v));
			}
		}

		// Only cells with non default values.
		AMapToData d = MapToFactory.create(off.size(), map.size());
		for(int i = 0; i < off.size(); i++)
			d.set(i, val.get(i));

		return new Pair<>(off, d);

	}

	private Pair<IntArrayList, AMapToData> encodeAndUpdateDense(MatrixBlock data, int col) {
		throw new NotImplementedException();
	}

	private Pair<IntArrayList, AMapToData> encodeAndUpdateGeneric(MatrixBlock data, int col) {
		throw new NotImplementedException();
	}

	private IntArrayList getCachedArray(int id) {
		if(cachedArrays == null) {
			ThreadLocal<Pair<IntArrayList, IntArrayList>> t = new ThreadLocal<>() {
				@Override
				protected Pair<IntArrayList, IntArrayList> initialValue() {
					IntArrayList a = new IntArrayList();
					IntArrayList b = new IntArrayList();
					Pair<IntArrayList, IntArrayList> p = new Pair<>(a, b);
					return p;
				}
			};
			cachedArrays = new SoftReference<>(t);

		}
		IntArrayList ret = id == 0 ? cachedArrays.get().get().getKey() : cachedArrays.get().get().getValue();

		ret.reset();
		return ret;
	}

	@Override
	protected AColGroup encodeVT(MatrixBlock data, IColIndex columns) {
		if(data.isEmpty() || (data.isInSparseFormat() && data.getSparseBlock().isEmpty(columns.get(0))))
			return new ColGroupEmpty(columns);
		else if(data.isInSparseFormat())
			return encodeSparseT(data, columns);
		else // dense and generic can be handled similarly here.
			return encodeDenseT(data, columns);
	}

	private AColGroup encodeDenseT(MatrixBlock data, IColIndex columns) {
		final int col = columns.get(0);
		final int nCol = data.getNumColumns();
		final DenseBlock db = data.getDenseBlock();
		final double[] vals = db.values(col);
		final int offStart = db.pos(col);
		final IntArrayList off = new IntArrayList();

		// full iteration
		for(int i = 0, o = offStart; i < nCol; i++, o++)
			if(!Util.eq(vals[o], def))
				off.appendValue(i);

		// Only cells with non default values.
		final AMapToData d = MapToFactory.create(off.size(), map.size());
		for(int i = 0; i < off.size(); i++) {
			int o = off.get(i) + offStart;
			d.set(i, map.getId(vals[o]));
		}
		return finalizeEncode(data, off, d, columns, data.getNumColumns());
	}

	private AColGroup encodeSparseT(MatrixBlock data, IColIndex columns) {
		final int col = columns.get(0);
		final int nRow = data.getNumColumns();
		final SparseBlock sb = data.getSparseBlock();
		// final IntArrayList off = new IntArrayList();

		int apos = sb.pos(col);
		final int[] aix = sb.indexes(col);
		final int alen = sb.size(col) + apos;
		final double[] aval = sb.values(col);

		if(def == 0.0) {
			// if def is zero ... easy.
			final AMapToData d = MapToFactory.create(alen - apos, map.size());
			int end = alen - apos;

			for(int i = 0; i < end; i++, apos++)
				d.set(i, map.getId(aval[apos]));

			allocateDictionary();
			final AOffset off = OffsetFactory.createOffset(aix, sb.pos(col), alen);
			return ColGroupSDC.create(columns, nRow, lastDict, new double[] {def}, off, d, null);
		}
		else {
			final IntArrayList off = getCachedArray(0);
			final IntArrayList dt = getCachedArray(1);

			final int zeroId = map.getId(0.0);
			for(int i = 0; i < data.getNumColumns(); i++) {
				if(apos < alen && aix[apos] == i) {
					if(!Util.eq(aval[apos], def)) {
						off.appendValue(i);
						dt.appendValue(map.getId(aval[apos]));
					}
					apos++;
				}
				else {
					off.appendValue(i);
					dt.appendValue(zeroId);
				}
			}

			final AMapToData d = MapToFactory.create(dt.size(), map.size());
			for(int i = 0; i < dt.size(); i++)
				d.set(i, dt.get(i));

			allocateDictionary();
			return ColGroupSDC.create(columns, nRow, lastDict, new double[] {def}, OffsetFactory.createOffset(off), d,
				null);
		}
	}

	@Override
	protected ICLAScheme updateVT(MatrixBlock data, IColIndex columns) {
		final int col = columns.get(0);
		if(data.isEmpty() || (data.isInSparseFormat() && data.getSparseBlock().isEmpty(columns.get(0)))) {
			if(def != 0)
				map.increment(0.0, col);
		}
		else if(data.isInSparseFormat())
			updateSparseT(data, col);
		else
			updateDenseT(data, col);

		return this;

	}

	private void updateSparseT(MatrixBlock data, int col) {
		final SparseBlock sb = data.getSparseBlock();

		int apos = sb.pos(col);
		final int alen = sb.size(col) + apos;
		final double[] aval = sb.values(col);
		if(def == 0.0) {
			// if def is zero ... easy.
			int end = alen - apos;
			for(int i = 0; i < end; i++, apos++)
				map.increment(aval[apos]);

		}
		else {
			int end = alen - apos;
			for(int i = 0; i < end; i++, apos++)
				if(!Util.eq(aval[apos], def))
					map.increment(aval[apos]);
			map.increment(0.0, data.getNumColumns() - end);
		}

	}

	private void updateDenseT(MatrixBlock data, int col) {
		final DenseBlock db = data.getDenseBlock();
		final double[] vals = db.values(col);
		final int nCol = data.getNumColumns();

		for(int i = 0, off = db.pos(col); i < nCol; i++, off++) {

			final double v = vals[off];
			if(!Util.eq(v, def))
				map.increment(v);
		}
	}

	@Override
	protected Pair<ICLAScheme, AColGroup> tryUpdateAndEncodeT(MatrixBlock data, IColIndex columns) {
		if(data.isEmpty() || (data.isInSparseFormat() && data.getSparseBlock().isEmpty(columns.get(0))))
			return new Pair<>(this, new ColGroupEmpty(columns));
		else if(data.isInSparseFormat())
			return encodeAndUpdateSparseT(data, columns);
		else
			throw new NotImplementedException();
	}

	protected Pair<ICLAScheme, AColGroup> encodeAndUpdateSparseT(MatrixBlock data, IColIndex columns) {
		final int col = columns.get(0);
		final int nRow = data.getNumColumns();
		final SparseBlock sb = data.getSparseBlock();

		int apos = sb.pos(col);
		final int[] aix = sb.indexes(col);
		final int alen = sb.size(col) + apos;
		final double[] aval = sb.values(col);

		if(def == 0.0) {
			// if def is zero ... easy.
			final AMapToData d = MapToFactory.create(alen - apos, map.size());
			int end = alen - apos;

			for(int i = 0; i < end; i++, apos++)
				d.set(i, map.increment(aval[apos]));

			allocateDictionary();
			final AOffset off = OffsetFactory.createOffset(aix, sb.pos(col), alen);
			AColGroup g = ColGroupSDC.create(columns, nRow, lastDict, new double[] {def}, off, d, null);
			return new Pair<>(this, g);
		}
		else {
			final IntArrayList off = getCachedArray(0);
			final IntArrayList dt = getCachedArray(1);

			final int zeroId = map.getId(0.0);
			for(int i = 0; i < data.getNumColumns(); i++) {
				if(aix[apos] == i) {
					if(!Util.eq(aval[apos], def)) {
						off.appendValue(i);
						dt.appendValue(map.increment(aval[apos]));
					}
					apos++;
				}
				else {
					off.appendValue(i);
					dt.appendValue(zeroId);
				}
			}

			final AMapToData d = MapToFactory.create(dt.size(), map.size());
			for(int i = 0; i < dt.size(); i++)
				d.set(i, dt.get(i));

			allocateDictionary();
			AColGroup g = ColGroupSDC.create(columns, nRow, lastDict, new double[] {def},
				OffsetFactory.createOffset(off), d, null);
			return new Pair<>(this, g);
		}
	}

	@Override
	public SDCSchemeSC clone() {
		return new SDCSchemeSC(cols, map.clone(), def);
	}

}
