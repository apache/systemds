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
import java.util.Iterator;

import org.apache.commons.lang.ArrayUtils;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.frame.data.columns.ColumnMetadata;
import org.apache.sysds.runtime.frame.data.iterators.IteratorFactory;

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
		// sanity check row dimension mismatch
		if(nRow != b.getNumRows())
			throw new DMLRuntimeException(
				"Incompatible number of rows for cbind: " + b.getNumRows() + " (expected: " + nRow + ")");

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
		// sanity check column dimension mismatch
		if(nCol != b.getNumColumns()) {
			throw new DMLRuntimeException(
				"Incompatible number of columns for rbind: " + b.getNumColumns() + " (expected: " + nCol + ")");
		}

		// ret._schema = a.getSchema().clone();
		String[] _colnames = (a.getColumnNames(false) != null) ? a.getColumnNames().clone() : null;
		ColumnMetadata[] _colmeta = new ColumnMetadata[a.getNumColumns()];
		for(int j = 0; j < nCol; j++)
			_colmeta[j] = new ColumnMetadata();

		// concatenate data (deep copy first, append second)
		Array<?>[] _coldata = new Array[a.getNumColumns()];
		for(int j = 0; j < a.getNumColumns(); j++)
			_coldata[j] = a.getColumn(j).clone();
		Iterator<Object[]> iter = IteratorFactory.getObjectRowIterator(b, a.getSchema());
		FrameBlock ret = new FrameBlock(a.getSchema().clone(), _colnames, _colmeta, _coldata);
		while(iter.hasNext())
			ret.appendRow(iter.next());
		return ret;
	}

	@SuppressWarnings("unchecked")
	private static <T> T[] addAll(T[] a, T[] b) {
		return (T[]) ArrayUtils.addAll(a, b);
	}
}
