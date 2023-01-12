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

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory;
import org.apache.sysds.runtime.frame.data.columns.ColumnMetadata;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;

public class FrameLibRemoveEmpty {
	protected static final Log LOG = LogFactory.getLog(FrameLibRemoveEmpty.class.getName());

	private final FrameBlock in;
	private final boolean rows;
	private final boolean emptyReturn;
	private final MatrixBlock select;

	private final int nRow;
	private final int nCol;

	public static FrameBlock removeEmpty(FrameBlock fb, boolean rows, boolean emptyReturn, MatrixBlock select) {
		return new FrameLibRemoveEmpty(fb, rows, emptyReturn, select).apply();
	}

	private FrameLibRemoveEmpty(FrameBlock in, boolean rows, boolean emptyReturn, MatrixBlock select) {
		this.in = in;
		this.rows = rows;
		this.emptyReturn = emptyReturn;
		this.select = select;

		nRow = in.getNumRows();
		nCol = in.getNumColumns();
		verify();
	}

	private void verify() {
		if(select != null) {
			final int SnRows = select.getNumRows();
			final int SnCols = select.getNumColumns();
			// in remove empty columns case
			boolean notValid = !rows && (SnCols != nCol || SnRows != 1);
			// in remove empty rows case
			notValid |= rows && (SnRows != nRow || SnCols != 1);
			if(notValid)
				throw new DMLRuntimeException("Frame rmempty incorrect select vector dimensions");
		}
	}

	private FrameBlock apply() {
		FrameBlock ret = rows ? removeEmptyRows() : removeEmptyColumns();
		return ret;
	}

	private FrameBlock removeEmptyRows() {
		return select == null ? removeEmptyRowsNoSelect() : removeEMptyRowsWithSelect();
	}

	private FrameBlock removeEmptyRowsNoSelect() {
		final FrameBlock ret = new FrameBlock();
		final boolean[] select = new boolean[nRow];
		for(int i = 0; i < nCol; i++)
			in.getColumn(i).findEmpty(select);
		final int nTrue = getNumberTrue(select);

		if(nTrue == 0)
			return removeEmptyRowsEmptyReturn();
		else if(nTrue == nRow)
			return new FrameBlock(in);

		final String[] colNames = in.getColumnNames(false);
		for(int i = 0; i < nCol; i++) {
			ret.appendColumn(in.getColumn(i).select(select, nTrue));
			if(colNames != null)
				ret.setColumnName(i, colNames[i]);
		}

		return ret;
	}

	private FrameBlock removeEMptyRowsWithSelect() {
		if(select.getNonZeros() == nRow)
			return in;
		else if(select.isEmpty())
			return removeEmptyRowsEmptyReturn();

		final String[] colNames = in.getColumnNames(false);
		final FrameBlock ret = new FrameBlock();
		final int[] indices = DataConverter.convertVectorToIndexList(select);
		for(int i = 0; i < nCol; i++) {
			ret.appendColumn(in.getColumn(i).select(indices));
			if(colNames != null)
				ret.setColumnName(i, colNames[i]);
		}

		return ret;
	}

	private FrameBlock removeEmptyRowsEmptyReturn() {
		final ValueType[] schema = in.getSchema();
		final String[] colNames = in.getColumnNames(false);
		if(emptyReturn) { // single null row
			String[][] arr = new String[1][];
			arr[0] = new String[schema.length];
			return new FrameBlock(schema, colNames, arr);
		}
		else // no rows
			return new FrameBlock(schema, colNames);
	}

	private static int getNumberTrue(boolean[] select) {
		int i = 0;
		for(boolean b : select)
			i += b ? 1 : 0;
		return i;
	}

	private FrameBlock removeEmptyColumns() {

		List<ColumnMetadata> columnMetadata = new ArrayList<>();

		final String[] colNames = in.getColumnNames(false);
		int k = 0;
		FrameBlock ret = new FrameBlock();
		if(select == null) {
			for(int i = 0; i < nCol; i++) {
				final Array<?> colData = in.getColumn(i);
				if(!colData.isEmpty()) {
					ret.appendColumn(colData);
					columnMetadata.add(new ColumnMetadata(in.getColumnMetadata(i)));
					if(colNames != null)
						ret.setColumnName(k++, colNames[i]);
				}
			}
			if(ret.getNumColumns() == 0)
				return removeEmptyColumnsEmptyReturn();
			return ret;
		}
		else {
			if(select.getNonZeros() == nCol)
				return new FrameBlock(in);
			else if(select.getNonZeros() == 0)
				return removeEmptyColumnsEmptyReturn();
			else {
				for(int i : DataConverter.convertVectorToIndexList(select)) {
					ret.appendColumn(in.getColumn(i));
					columnMetadata.add(new ColumnMetadata(in.getColumnMetadata(i)));
					if(colNames != null)
						ret.setColumnName(k++, colNames[i]);
				}
				return ret;
			}
		}
	}

	private FrameBlock removeEmptyColumnsEmptyReturn() {
		if(emptyReturn) {
			FrameBlock ret = new FrameBlock();
			ret.appendColumn(ArrayFactory.create(new String[nRow]));
			return ret;
		}
		else
			return new FrameBlock();
	}

}
