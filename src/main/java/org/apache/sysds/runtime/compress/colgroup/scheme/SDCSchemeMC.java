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
import org.apache.sysds.runtime.compress.colgroup.ColGroupConst;
import org.apache.sysds.runtime.compress.colgroup.ColGroupEmpty;
import org.apache.sysds.runtime.compress.colgroup.ColGroupSDC;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictionaryFactory;
import org.apache.sysds.runtime.compress.colgroup.dictionary.MatrixBlockDictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffset;
import org.apache.sysds.runtime.compress.colgroup.offset.OffsetFactory;
import org.apache.sysds.runtime.compress.readers.ReaderColumnSelection;
import org.apache.sysds.runtime.compress.utils.ACount;
import org.apache.sysds.runtime.compress.utils.DblArray;
import org.apache.sysds.runtime.compress.utils.DblArrayCountHashMap;
import org.apache.sysds.runtime.compress.utils.IntArrayList;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class SDCSchemeMC extends SDCScheme {

	private final DblArray emptyRow;
	private final DblArray def;
	private final DblArrayCountHashMap map;

	public SDCSchemeMC(IColIndex cols, DblArrayCountHashMap map, DblArray def) {
		super(cols);
		this.map = map;
		this.def = def;
		this.emptyRow = new DblArray(new double[cols.size()]);
	}

	protected SDCSchemeMC(ASDC g) {
		super(g.getColIndices());

		this.lastDict = g.getDictionary();
		final MatrixBlockDictionary mbd = lastDict.getMBDict(this.cols.size());
		final MatrixBlock mbDict = mbd != null ? mbd.getMatrixBlock() : new MatrixBlock(1, this.cols.size(), 0.0);
		final int dictRows = mbDict.getNumRows();
		final int dictCols = mbDict.getNumColumns();

		map = new DblArrayCountHashMap(dictRows * 2);
		emptyRow = new DblArray(new double[dictCols]);
		if(mbDict.isEmpty()) {// there is the option of an empty dictionary.
			map.increment(emptyRow);
		}
		else {
			// Read the mapping data and materialize map.
			final ReaderColumnSelection reader = ReaderColumnSelection.createReader(mbDict, //
				ColIndexFactory.create(dictCols), false, 0, dictRows);
			DblArray d = null;
			while((d = reader.nextRow()) != null) {
				// this leverage the fact that our readers not transposed never skips a line
				map.increment(d);
			}
		}

		def = new DblArray(g.getCommon());

	}

	protected SDCSchemeMC(ASDCZero g) {
		super(g.getColIndices());

		this.lastDict = g.getDictionary();
		final MatrixBlock mbDict = lastDict.getMBDict(this.cols.size()).getMatrixBlock();
		final int dictRows = mbDict.getNumRows();
		final int dictCols = mbDict.getNumColumns();

		// Read the mapping data and materialize map.
		map = new DblArrayCountHashMap(dictRows * 2);
		final ReaderColumnSelection r = ReaderColumnSelection.createReader(mbDict, //
			ColIndexFactory.create(dictCols), false, 0, dictRows);
		DblArray d = null;
		while((d = r.nextRow()) != null)
			map.increment(d);

		emptyRow = new DblArray(new double[dictCols]);
		def = new DblArray(new double[dictCols]);
	}

	@Override
	protected AColGroup encodeV(MatrixBlock data, IColIndex columns) {
		if(data.isEmpty())
			return new ColGroupEmpty(columns);

		final int nRow = data.getNumRows();
		final IntArrayList offs = new IntArrayList();
		final ReaderColumnSelection reader = ReaderColumnSelection.createReader(//
			data, cols, false, 0, nRow);
		final AMapToData d = encode(data, reader, offs, cols, nRow);

		return finalizeEncode(data, offs, d, columns, nRow);
	}

	private AColGroup finalizeEncode(MatrixBlock data, IntArrayList offs, AMapToData d, IColIndex columns, int nRow) {

		if(lastDict == null || lastDict.getNumberOfValues(columns.size()) != map.size())
			lastDict = DictionaryFactory.create(map, columns.size(), false, data.getSparsity());
		if(offs.size() == 0) {
			return ColGroupConst.create(columns, def.getData());
			// return ColGroupDDC.create(columns, lastDict, d, null);
		}
		else {
			final AOffset off = OffsetFactory.createOffset(offs);
			return ColGroupSDC.create(columns, nRow, lastDict, def.getData(), off, d, null);
		}

	}

	private AMapToData encode(MatrixBlock data, ReaderColumnSelection reader, IntArrayList off, IColIndex cols,
		int nRow) {

		DblArray cellVals;
		ACount<DblArray> emptyIdx = map.getC(emptyRow);

		IntArrayList dt = new IntArrayList();
		int r = 0;
		while((cellVals = reader.nextRow()) != null) {
			final int row = reader.getCurrentRowIndex();
			if(row != r) { // empty rows.
				if(emptyIdx != null) {
					// empty is non default.
					while(r < row) {
						off.appendValue(r++);
						dt.appendValue(emptyIdx.id);
					}
				}
				else {
					r = row;
				}
			}

			if(!cellVals.equals(def)) {
				final int id = map.getId(cellVals);
				if(id >= 0) {
					off.appendValue(row);
					dt.appendValue(id);
					r++;
				}
			}
			else {
				r++;
			}
		}
		if(emptyIdx != null) {
			// empty is non default.
			while(r < nRow) {
				off.appendValue(r++);
				dt.appendValue(emptyIdx.id);
			}
		}

		final AMapToData d = MapToFactory.create(off.size(), map.size());
		for(int i = 0; i < off.size(); i++)
			d.set(i, dt.get(i));

		return d;
	}

	@Override
	protected ICLAScheme updateV(MatrixBlock data, IColIndex columns) {
		if(data.isEmpty()) {
			if(!def.equals(emptyRow))
				map.increment(emptyRow, data.getNumRows());
			return this;
		}

		final int nRow = data.getNumRows();
		final ReaderColumnSelection reader = ReaderColumnSelection.createReader(//
			data, cols, false, 0, nRow);

		return update(data, reader, columns, nRow);

	}

	private ICLAScheme update(MatrixBlock data, ReaderColumnSelection reader, IColIndex columns, final int nRow) {
		DblArray cellVals;
		final boolean defIsEmpty = emptyRow.equals(def);

		int r = 0;
		while((cellVals = reader.nextRow()) != null) {
			final int row = reader.getCurrentRowIndex();
			if(row != r) {
				if(!defIsEmpty)
					map.increment(emptyRow, row - r);
				r = row;
			}
			if(!cellVals.equals(def))
				map.increment(cellVals);
		}
		if(!defIsEmpty) {
			// empty is non default.
			if(r < nRow)
				map.increment(emptyRow, nRow - r);
		}

		return this;
	}

	protected Object getDef() {
		return def;
	}

	protected Object getMap() {
		return map;
	}

	@Override
	protected AColGroup encodeVT(MatrixBlock data, IColIndex columns) {
		if(data.isEmpty())
			return new ColGroupEmpty(columns);

		final int nRow = data.getNumColumns();
		final IntArrayList offs = new IntArrayList();
		final ReaderColumnSelection reader = ReaderColumnSelection.createReader(//
			data, cols, true, 0, nRow);
		final AMapToData d = encode(data, reader, offs, cols, nRow);

		return finalizeEncode(data, offs, d, columns, nRow);
	}

	@Override
	protected ICLAScheme updateVT(MatrixBlock data, IColIndex columns) {
		if(data.isEmpty()) {
			if(!def.equals(emptyRow))
				map.increment(emptyRow, data.getNumColumns());
			return this;
		}

		final int nRow = data.getNumColumns();
		final ReaderColumnSelection reader = ReaderColumnSelection.createReader(//
			data, cols, true, 0, nRow);

		return update(data, reader, columns, nRow);
	}

	@Override
	public SDCSchemeMC clone() {
		return new SDCSchemeMC(cols, map.clone(), def);
	}

}
