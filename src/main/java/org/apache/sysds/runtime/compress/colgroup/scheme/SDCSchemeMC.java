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
import org.apache.sysds.runtime.compress.colgroup.ASDC;
import org.apache.sysds.runtime.compress.colgroup.ASDCZero;
import org.apache.sysds.runtime.compress.colgroup.ColGroupDDC;
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
import org.apache.sysds.runtime.compress.utils.DblArray;
import org.apache.sysds.runtime.compress.utils.DblArrayCountHashMap;
import org.apache.sysds.runtime.compress.utils.IntArrayList;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class SDCSchemeMC extends SDCScheme {

	private final DblArray emptyRow;
	private final DblArray def;
	private final DblArrayCountHashMap map;

	protected SDCSchemeMC(ASDC g) {
		super(g.getColIndices());
		try {
			this.lastDict = g.getDictionary();
			final MatrixBlockDictionary mbd = lastDict.getMBDict(this.cols.size());
			final MatrixBlock mbDict = mbd != null ? mbd.getMatrixBlock() : new MatrixBlock(1, this.cols.size(), 0.0);
			final int dictRows = mbDict.getNumRows();
			final int dictCols = mbDict.getNumColumns();

			// Read the mapping data and materialize map.
			map = new DblArrayCountHashMap(dictRows * 2, dictCols);
			final ReaderColumnSelection reader = ReaderColumnSelection.createReader(mbDict, //
				ColIndexFactory.create(dictCols), false, 0, dictRows);
			emptyRow = new DblArray(new double[dictCols]);
			DblArray d = null;
			int r = 0;
			while((d = reader.nextRow()) != null) {

				final int row = reader.getCurrentRowIndex();
				if(row != r) {
					map.increment(emptyRow, row - r);
					r = row;
				}
				map.increment(d);
			}
			if(r < dictRows) {
				map.increment(emptyRow, dictRows - r);
			}

			def = new DblArray(g.getCommon());
		}
		catch(Exception e) {
			throw new DMLCompressionException(g.getDictionary().toString());
		}
	}

	protected SDCSchemeMC(ASDCZero g) {
		super(g.getColIndices());

		this.lastDict = g.getDictionary();
		final MatrixBlock mbDict = lastDict.getMBDict(this.cols.size()).getMatrixBlock();
		final int dictRows = mbDict.getNumRows();
		final int dictCols = mbDict.getNumColumns();

		// Read the mapping data and materialize map.
		map = new DblArrayCountHashMap(dictRows * 2, dictCols);
		final ReaderColumnSelection r = ReaderColumnSelection.createReader(mbDict, //
			ColIndexFactory.create(dictCols), false, 0, dictRows);
		DblArray d = null;
		while((d = r.nextRow()) != null)
			map.increment(d);

		emptyRow = new DblArray(new double[dictCols]);
		def = new DblArray(new double[dictCols]);
	}

	@Override
	public AColGroup encode(MatrixBlock data, IColIndex columns) {
		validate(data, columns);
		final int nRow = data.getNumRows();
		if(data.isEmpty())
			return new ColGroupEmpty(columns);
		// final AMapToData d = MapToFactory.create(nRow, map.size());

		final IntArrayList offs = new IntArrayList();
		AMapToData d = encode(data, offs, cols);

		if(lastDict == null || lastDict.getNumberOfValues(columns.size()) != map.size())
			lastDict = DictionaryFactory.create(map, columns.size(), false, data.getSparsity());
		if(offs.size() == 0)
			return ColGroupDDC.create(columns, lastDict, d, null);
		else {
			final AOffset off = OffsetFactory.createOffset(offs);
			return ColGroupSDC.create(columns, nRow, lastDict, def.getData(), off, d, null);
		}
	}

	private AMapToData encode(MatrixBlock data, IntArrayList off, IColIndex cols) {
		final int nRow = data.getNumRows();
		final ReaderColumnSelection reader = ReaderColumnSelection.createReader(//
			data, cols, false, 0, nRow);
		DblArray cellVals;
		int emptyIdx = map.getId(emptyRow);
		emptyRow.equals(def);
		IntArrayList dt = new IntArrayList();

		int r = 0;
		while((cellVals = reader.nextRow()) != null) {
			final int row = reader.getCurrentRowIndex();
			if(row != r) {
				if(emptyIdx >= 0) {
					// empty is non default.
					while(r < row) {
						off.appendValue(r++);
						dt.appendValue(emptyIdx);
					}
				}
				else {
					r = row;
				}
			}
			final int id = map.getId(cellVals);
			if(id >= 0) {
				off.appendValue(row);
				dt.appendValue(id);
				r++;
			}
		}
		if(emptyIdx >= 0) {
			// empty is non default.
			while(r < nRow) {
				off.appendValue(r++);
				dt.appendValue(emptyIdx);
			}
		}

		AMapToData d = MapToFactory.create(off.size(), map.size());
		for(int i = 0; i < off.size(); i++)
			d.set(i, dt.get(i));

		return d;
	}

	@Override
	public ICLAScheme update(MatrixBlock data, IColIndex columns) {
		validate(data, columns);

		if(data.isEmpty()) {
			if(!def.equals(emptyRow))
				map.increment(emptyRow, data.getNumRows());
			return this;
		}
		final int nRow = data.getNumRows();
		final ReaderColumnSelection reader = ReaderColumnSelection.createReader(//
			data, cols, false, 0, nRow);
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
			final int id = map.getId(cellVals);
			if(id >= 0)
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
}
