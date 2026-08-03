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

import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.instructions.ooc.OOCStream;
import org.apache.sysds.runtime.instructions.spark.data.IndexedMatrixValue;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import io.delta.kernel.data.ColumnVector;
import io.delta.kernel.data.ColumnarBatch;
import io.delta.kernel.data.FilteredColumnarBatch;
import io.delta.kernel.engine.Engine;
import io.delta.kernel.types.DataType;
import io.delta.kernel.types.DoubleType;
import io.delta.kernel.types.StructType;
import io.delta.kernel.utils.CloseableIterable;
import io.delta.kernel.utils.CloseableIterator;

/**
 * Single-threaded native Delta Lake writer for matrices, built on the
 * Spark-free Delta Kernel library. It creates a Delta table at the target
 * directory with an all-double schema {@code c0..c(n-1)} (replacing any existing
 * table at that path), streams the {@link MatrixBlock} rows as columnar batches
 * into parquet data files via the kernel's default engine, and commits the
 * corresponding add-file actions to the transaction log.
 */
public class WriterDelta extends MatrixWriter {

	@Override
	public void writeMatrixToHDFS(MatrixBlock src, String fname, long rlen, long clen, int blen, long nnz, boolean diag)
		throws IOException
	{
		if( src.getNumRows() != rlen || src.getNumColumns() != clen )
			throw new IOException("Matrix dimensions mismatch with metadata: ("
				+ src.getNumRows() + "x" + src.getNumColumns() + ") vs (" + rlen + "x" + clen + ").");
		int ncol = (int) clen;
		int nrow = (int) rlen;
		int batchRows = ConfigurationManager.getDeltaWriterBatchSize();
		//fast path: a contiguous dense block lets the column views read straight
		//from the backing double[] (avoids per-cell MatrixBlock.get dispatch).
		double[] dense = (!src.isInSparseFormat() && src.getDenseBlock() != null
			&& src.getDenseBlock().isContiguous()) ? src.getDenseBlockValues() : null;
		//size data files adaptively (toward one file per parallel reader) for faster parallel reads.
		//Delta writes every cell as a double, so size by the dense footprint rather than the (possibly
		//sparse) in-memory size, which would understate the on-disk table for sparse inputs.
		long estimatedBytes = (long) nrow * ncol * 8L;
		Engine engine = DeltaKernelUtils.createWriteEngine(estimatedBytes);
		DeltaKernelUtils.commit(engine, DeltaKernelUtils.qualify(fname),
			buildSchema(ncol), new MatrixBatchIterator(src, dense, nrow, ncol, batchRows));
	}

	@Override
	public void writeEmptyMatrixToHDFS(String fname, long rlen, long clen, int blen)
		throws IOException
	{
		//empty table: create with schema but no data files
		Engine engine = DeltaKernelUtils.createEngine();
		DeltaKernelUtils.commit(engine, DeltaKernelUtils.qualify(fname),
			buildSchema((int) clen), CloseableIterable.<FilteredColumnarBatch>emptyIterable().iterator());
	}

	private static StructType buildSchema(int ncol) {
		StructType schema = new StructType();
		for( int c=0; c<ncol; c++ )
			schema = schema.add("c" + c, DoubleType.DOUBLE, false);
		return schema;
	}

	//not implemented (out-of-core streaming write)
	@Override
	public long writeMatrixFromStream(String fname, OOCStream<IndexedMatrixValue> stream, long rlen, long clen, int blen)
		throws IOException
	{
		throw new UnsupportedOperationException("Out-of-core stream write is not supported for the Delta format.");
	}

	/** Chunks a MatrixBlock into fixed-size columnar batches for the kernel write path. */
	private static class MatrixBatchIterator implements CloseableIterator<FilteredColumnarBatch> {
		private final MatrixBlock _mb;
		private final double[] _dense;
		private final int _nrow;
		private final int _ncol;
		private final int _batchRows;
		private final StructType _schema;
		private int _pos = 0;

		MatrixBatchIterator(MatrixBlock mb, double[] dense, int nrow, int ncol, int batchRows) {
			_mb = mb;
			_dense = dense;
			_nrow = nrow;
			_ncol = ncol;
			_batchRows = batchRows;
			_schema = buildSchema(ncol);
		}

		@Override
		public boolean hasNext() {
			return _pos < _nrow;
		}

		@Override
		public FilteredColumnarBatch next() {
			if( !hasNext() )
				throw new NoSuchElementException();
			int size = Math.min(_batchRows, _nrow - _pos);
			ColumnarBatch batch = new MatrixColumnarBatch(_mb, _dense, _schema, _pos, size, _ncol);
			_pos += size;
			//no selection vector: all rows in the batch are written
			return new FilteredColumnarBatch(batch, Optional.empty());
		}

		@Override
		public void close() {
			//nothing to release
		}
	}

	/** Read-only view of a row range of a MatrixBlock as a Delta Kernel columnar batch. */
	private static class MatrixColumnarBatch implements ColumnarBatch {
		private final MatrixBlock _mb;
		private final double[] _dense;
		private final StructType _schema;
		private final int _rowStart;
		private final int _size;
		private final int _ncol;

		MatrixColumnarBatch(MatrixBlock mb, double[] dense, StructType schema, int rowStart, int size, int ncol) {
			_mb = mb;
			_dense = dense;
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
			if( ordinal < 0 || ordinal >= _ncol )
				throw new IndexOutOfBoundsException("column ordinal " + ordinal);
			return new MatrixColumnVector(_mb, _dense, _rowStart, _size, _ncol, ordinal);
		}

		@Override
		public int getSize() {
			return _size;
		}
	}

	/** Read-only double column view over one column of a MatrixBlock row range. */
	private static class MatrixColumnVector implements ColumnVector {
		private final MatrixBlock _mb;
		private final double[] _dense; // contiguous dense backing array, or null
		private final int _rowStart;
		private final int _size;
		private final int _ncol;
		private final int _col;

		MatrixColumnVector(MatrixBlock mb, double[] dense, int rowStart, int size, int ncol, int col) {
			_mb = mb;
			_dense = dense;
			_rowStart = rowStart;
			_size = size;
			_ncol = ncol;
			_col = col;
		}

		@Override
		public DataType getDataType() {
			return DoubleType.DOUBLE;
		}

		@Override
		public int getSize() {
			return _size;
		}

		@Override
		public boolean isNullAt(int rowId) {
			return false;
		}

		@Override
		public double getDouble(int rowId) {
			//dense contiguous single block => index fits in int (getDenseBlockValues
			//is only handed over for single-block dense matrices)
			return (_dense != null)
				? _dense[(_rowStart + rowId) * _ncol + _col]
				: _mb.get(_rowStart + rowId, _col);
		}

		@Override
		public void close() {
			//nothing to release
		}
	}
}
