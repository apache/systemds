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

package org.apache.sysds.runtime.io;

import java.io.IOException;
import java.util.NoSuchElementException;
import java.util.Optional;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.Array;

import io.delta.kernel.data.ColumnVector;
import io.delta.kernel.data.ColumnarBatch;
import io.delta.kernel.data.FilteredColumnarBatch;
import io.delta.kernel.engine.Engine;
import io.delta.kernel.types.BooleanType;
import io.delta.kernel.types.DataType;
import io.delta.kernel.types.DoubleType;
import io.delta.kernel.types.FloatType;
import io.delta.kernel.types.IntegerType;
import io.delta.kernel.types.LongType;
import io.delta.kernel.types.StringType;
import io.delta.kernel.types.StructType;
import io.delta.kernel.utils.CloseableIterator;

/**
 * Single-threaded native Delta Lake writer for frames, built on the Spark-free Delta Kernel library. It creates (or
 * recreates) a Delta table whose schema mirrors the frame schema (per-column {@link ValueType} mapped to a Delta type
 * and the frame column names), streams the {@link FrameBlock} rows as columnar batches into parquet data files, and
 * commits the add-file actions.
 */
public class FrameWriterDelta extends FrameWriter {

	@Override
	public void writeFrameToHDFS(FrameBlock src, String fname, long rlen, long clen)
		throws IOException, DMLRuntimeException {
		if(src.getNumRows() != rlen || src.getNumColumns() != clen)
			throw new IOException("Frame dimensions mismatch with metadata: (" + src.getNumRows() + "x"
				+ src.getNumColumns() + ") vs (" + rlen + "x" + clen + ").");
		int ncol = (int) clen;
		int nrow = (int) rlen;
		StructType schema = buildSchema(src.getSchema(), src.getColumnNames(), ncol);

		// snapshot the typed column arrays + per-column nullability once, so the
		// hot per-cell path can read primitives directly (no boxing) and skip
		// null-checks on non-nullable columns.
		Array<?>[] cols = new Array<?>[ncol];
		boolean[] nullable = new boolean[ncol];
		for(int c = 0; c < ncol; c++) {
			cols[c] = src.getColumn(c);
			nullable[c] = cols[c].containsNull();
		}

		int batchRows = ConfigurationManager.getDeltaWriterBatchSize();
		// size data files adaptively (toward one file per parallel reader) for faster parallel reads
		Engine engine = DeltaKernelUtils.createWriteEngine(src.getInMemorySize());
		DeltaKernelUtils.commit(engine, DeltaKernelUtils.qualify(fname), schema,
			new FrameBatchIterator(cols, nullable, schema, nrow, ncol, batchRows));
	}

	private static StructType buildSchema(ValueType[] vtSchema, String[] names, int ncol) {
		StructType schema = new StructType();
		for(int c = 0; c < ncol; c++)
			schema = schema.add(names[c], toDeltaType(vtSchema[c]), true);
		return schema;
	}

	static DataType toDeltaType(ValueType vt) {
		switch(vt) {
			case FP64:
				return DoubleType.DOUBLE;
			case FP32:
				return FloatType.FLOAT;
			case INT64:
				return LongType.LONG;
			case INT32:
			case UINT8:
			case UINT4:
				return IntegerType.INTEGER;
			case BOOLEAN:
				return BooleanType.BOOLEAN;
			default:
				return StringType.STRING; // STRING/CHARACTER/HASH*/UNKNOWN
		}
	}

	/** Chunks the frame columns into fixed-size columnar batches for the kernel write path. */
	private static class FrameBatchIterator implements CloseableIterator<FilteredColumnarBatch> {
		private final Array<?>[] _cols;
		private final boolean[] _nullable;
		private final StructType _schema;
		private final int _nrow;
		private final int _ncol;
		private final int _batchRows;
		private int _pos = 0;

		FrameBatchIterator(Array<?>[] cols, boolean[] nullable, StructType schema, int nrow, int ncol, int batchRows) {
			_cols = cols;
			_nullable = nullable;
			_schema = schema;
			_nrow = nrow;
			_ncol = ncol;
			_batchRows = batchRows;
		}

		@Override
		public boolean hasNext() {
			return _pos < _nrow;
		}

		@Override
		public FilteredColumnarBatch next() {
			if(!hasNext())
				throw new NoSuchElementException();
			int size = Math.min(_batchRows, _nrow - _pos);
			ColumnarBatch batch = new FrameColumnarBatch(_cols, _nullable, _schema, _pos, size, _ncol);
			_pos += size;
			return new FilteredColumnarBatch(batch, Optional.empty());
		}

		@Override
		public void close() {
			// nothing to release
		}
	}

	/** Read-only view of a row range of the frame columns as a Delta Kernel columnar batch. */
	private static class FrameColumnarBatch implements ColumnarBatch {
		private final Array<?>[] _cols;
		private final boolean[] _nullable;
		private final StructType _schema;
		private final int _rowStart;
		private final int _size;
		private final int _ncol;

		FrameColumnarBatch(Array<?>[] cols, boolean[] nullable, StructType schema, int rowStart, int size, int ncol) {
			_cols = cols;
			_nullable = nullable;
			_schema = schema;
			_rowStart = rowStart;
			_size = size;
			_ncol = ncol;
		}

		@Override
		public StructType getSchema() {
			return _schema;
		}

		@Override
		public ColumnVector getColumnVector(int ordinal) {
			if(ordinal < 0 || ordinal >= _ncol)
				throw new IndexOutOfBoundsException("column ordinal " + ordinal);
			return new FrameColumnVector(_cols[ordinal], _nullable[ordinal], _schema.at(ordinal).getDataType(),
				_rowStart, _size);
		}

		@Override
		public int getSize() {
			return _size;
		}
	}

	/**
	 * Read-only typed column view over one column {@link Array} row range. Numeric values are read through
	 * {@link Array#getAsDouble(int)} to avoid boxing, and non-nullable columns short-circuit {@code isNullAt} so the
	 * kernel never pays for a redundant boxed fetch.
	 */
	private static class FrameColumnVector implements ColumnVector {
		private final Array<?> _col;
		private final boolean _nullable;
		private final DataType _type;
		private final int _rowStart;
		private final int _size;

		FrameColumnVector(Array<?> col, boolean nullable, DataType type, int rowStart, int size) {
			_col = col;
			_nullable = nullable;
			_type = type;
			_rowStart = rowStart;
			_size = size;
		}

		@Override
		public DataType getDataType() {
			return _type;
		}

		@Override
		public int getSize() {
			return _size;
		}

		@Override
		public boolean isNullAt(int rowId) {
			return _nullable && _col.get(_rowStart + rowId) == null;
		}

		@Override
		public String getString(int rowId) {
			Object v = _col.get(_rowStart + rowId);
			return (v == null) ? null : v.toString();
		}

		@Override
		public boolean getBoolean(int rowId) {
			return _col.getAsDouble(_rowStart + rowId) != 0;
		}

		@Override
		public double getDouble(int rowId) {
			return _col.getAsDouble(_rowStart + rowId);
		}

		@Override
		public float getFloat(int rowId) {
			return (float) _col.getAsDouble(_rowStart + rowId);
		}

		@Override
		public long getLong(int rowId) {
			// exact for INT64 (getAsDouble would lose precision beyond 2^53)
			return ((Number) _col.get(_rowStart + rowId)).longValue();
		}

		@Override
		public int getInt(int rowId) {
			return (int) _col.getAsDouble(_rowStart + rowId);
		}

		@Override
		public void close() {
			// nothing to release
		}
	}
}
