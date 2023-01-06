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

package org.apache.sysds.runtime.frame.data.lib;

import java.util.Arrays;
import java.util.HashSet;

import org.apache.commons.lang.ArrayUtils;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory;
import org.apache.sysds.runtime.frame.data.columns.ColumnMetadata;

public class FrameLibAppend {

	/**
	 * Appends the given argument FrameBlock 'that' to this FrameBlock by creating a deep copy to prevent side effects.
	 * For cbind, the frames are appended column-wise (same number of rows), while for rbind the frames are appended
	 * row-wise (same number of columns).
	 *
	 * @param a     FrameBlock to append to
	 * @param b     FrameBlock to append
	 * @param cbind if true, column append
	 * @return frame block of the two blocks combined.
	 */
	public static FrameBlock append(FrameBlock a, FrameBlock b, boolean cbind) {
		if(cbind)
			return appendCbind(a, b);
		else
			return appendRbind(a, b);
	}

	public static FrameBlock appendCbind(FrameBlock a, FrameBlock b) {
		final int nRow = a.getNumRows();
		final int nRowB = b.getNumRows();

		if(nRow != nRowB)
			throw new DMLRuntimeException("Incompatible number of rows for cbind: " + nRowB + " expected: " + nRow);
		else if(a.getNumColumns() == 0)
			return b;
		else if(b.getNumColumns() == 0)
			return a;

		final ValueType[] _schema = addAll(a.getSchema(), b.getSchema());
		final ColumnMetadata[] _colmeta = addAll(a.getColumnMetadata(), b.getColumnMetadata());
		final Array<?>[] _coldata = addAll(a.getColumns(), b.getColumns());
		String[] _colnames = addAll(a.getColumnNames(), b.getColumnNames());

		// check and enforce unique columns names
		if(!Arrays.stream(_colnames).allMatch(new HashSet<>()::add))
			_colnames = null; // set to default of null to allocate on demand

		return new FrameBlock(_schema, _colnames, _colmeta, _coldata);
	}

	public static FrameBlock appendRbind(FrameBlock a, FrameBlock b) {
		final int nCol = a.getNumColumns();
		final int nColB = b.getNumColumns();

		if(nCol != nColB)
			throw new DMLRuntimeException("Incompatible number of columns for rbind: " + nColB + " expected: " + nCol);
		else if(a.getNumRows() == 0)
			return b;
		else if(b.getNumRows() == 0)
			return a;

		String[] retColNames = (a.getColumnNames(false) != null) ? a.getColumnNames().clone() : null;
		ColumnMetadata[] retColMeta = new ColumnMetadata[a.getNumColumns()];
		for(int j = 0; j < nCol; j++)
			retColMeta[j] = new ColumnMetadata();

		// concatenate data (deep copy first, append second)
		Array<?>[] retCols = new Array[a.getNumColumns()];
		ValueType[] retSchema = new ValueType[a.getNumColumns()];
		for(int j = 0; j < a.getNumColumns(); j++) {
			retCols[j] = ArrayFactory.append(a.getColumn(j), b.getColumn(j));
			retSchema[j] = retCols[j].getValueType();
		}

		return new FrameBlock(retSchema, retColNames, retColMeta, retCols);
	}

	@SuppressWarnings("unchecked")
	private static <T> T[] addAll(T[] a, T[] b) {
		return (T[]) ArrayUtils.addAll(a, b);
	}
}
