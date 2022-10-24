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
import org.apache.sysds.runtime.compress.colgroup.ColGroupDDC;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.compress.readers.ReaderColumnSelection;
import org.apache.sysds.runtime.compress.utils.DblArray;
import org.apache.sysds.runtime.compress.utils.DblArrayCountHashMap;
import org.apache.sysds.runtime.compress.utils.Util;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class DDCScheme implements ICLAScheme {

	final private int[] cols;
	final private int nUnique;
	final private DblArrayCountHashMap map;
	final private ADictionary dict;

	private DDCScheme(ColGroupDDC g) {
		this.cols = g.getColIndices();
		this.nUnique = g.getNumValues();
		this.dict = g.getDictionary();
		final MatrixBlock mbDict = dict.getMBDict(this.cols.length).getMatrixBlock();
		final int dictRows = mbDict.getNumRows();
		final int dictCols = mbDict.getNumColumns();

		// Read the mapping data and materialize map.
		map = new DblArrayCountHashMap(dictRows, dictCols);
		final ReaderColumnSelection r = ReaderColumnSelection.createReader(mbDict, //
			Util.genColsIndices(dictCols), false, 0, dictRows);
		DblArray d = null;
		while((d = r.nextRow()) != null)
			map.increment(d);

	}

	/**
	 * Create a scheme for the DDC compression given
	 * 
	 * @param g A DDC Column group
	 * @return A DDC Compression scheme
	 */
	public static ICLAScheme create(ColGroupDDC g) {
		if(g.getColIndices().length == 1)
			return null;
		return new DDCScheme(g);
	}

	@Override
	public AColGroup encode(MatrixBlock data) {
		return encode(data, cols);
	}

	@Override
	public AColGroup encode(MatrixBlock data, int[] columns) {
		if(columns.length != cols.length)
			throw new IllegalArgumentException("Invalid columns to encode");
		final int nRow = data.getNumRows();
		final ReaderColumnSelection reader = ReaderColumnSelection.createReader(//
			data, columns, false, 0, nRow);
		final AMapToData d = MapToFactory.create(nRow,nUnique);

		DblArray cellVals;
		while((cellVals = reader.nextRow()) != null) {
			final int row = reader.getCurrentRowIndex();
			final int id = map.getId(cellVals);
			if(id == -1)
				return null;
			d.set(row, id);
		}

		return ColGroupDDC.create(columns, dict, d, null);
	}

}
