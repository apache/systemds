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
import java.io.InputStream;
import java.util.ArrayList;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import io.delta.kernel.data.ColumnVector;
import io.delta.kernel.engine.Engine;
import io.delta.kernel.types.DataType;
import io.delta.kernel.types.StructType;

/**
 * Single-threaded native Delta Lake reader for matrices, built on the
 * Spark-free Delta Kernel library. It opens the latest snapshot of a Delta
 * table directory, reads its parquet data files through the kernel's default
 * engine (honoring deletion vectors), and materializes the numeric columns
 * into a dense {@link MatrixBlock}.
 *
 * <p>Only numeric columns (double/float/long/int/short/byte/boolean) are
 * supported, matching the all-double nature of a SystemDS matrix. Dimensions
 * do not need to be known up front: the row count is discovered while scanning
 * and the column count is taken from the table schema.</p>
 */
public class ReaderDelta extends MatrixReader {

	@Override
	public MatrixBlock readMatrixFromHDFS(String fname, long rlen, long clen, int blen, long estnnz)
		throws IOException, DMLRuntimeException
	{
		Engine engine = DeltaKernelUtils.createEngine();
		String tablePath = DeltaKernelUtils.qualify(fname);

		//Scan column-at-a-time into one row-major buffer per batch (no per-row
		//allocation, no boxing, no per-cell set()). Buffers are concatenated into
		//the dense output via bulk array copies below.
		ArrayList<double[]> batches = new ArrayList<>();
		int[] nrowH = new int[1];
		StructType schema = DeltaKernelUtils.scan(engine, tablePath, sch -> {
			int[] types = columnTypes(sch);
			int ncol = sch.length();
			return (cols, size, selected) -> {
				batches.add(extractBatch(cols, size, selected, types, ncol));
				nrowH[0] += DeltaKernelUtils.countSelected(size, selected);
			};
		});

		int ncol = schema.length();
		int nrow = nrowH[0];
		long lestnnz = (estnnz >= 0) ? estnnz : (long) nrow * ncol;
		MatrixBlock ret = createOutputMatrixBlock(nrow, ncol, Math.max(nrow, 1), lestnnz, true, false);

		if( nrow > 0 && ncol > 0 )
			fillDense(ret, batches);
		ret.recomputeNonZeros();
		ret.examSparsity();
		return ret;
	}

	/** Derive the per-column internal type codes from the table schema. */
	static int[] columnTypes(StructType schema) {
		int ncol = schema.length();
		int[] types = new int[ncol];
		for( int c=0; c<ncol; c++ )
			types[c] = numericTypeCode(schema.at(c).getDataType(), schema.at(c).getName());
		return types;
	}

	/**
	 * Extract one columnar batch into a fresh row-major {@code double[]} of the
	 * live (selected, after deletion vector) rows. No per-cell boxing.
	 */
	static double[] extractBatch(ColumnVector[] cols, int size, boolean[] selected, int[] types, int ncol) {
		double[] buf = new double[DeltaKernelUtils.countSelected(size, selected) * ncol];
		extractBatchInto(cols, size, selected, types, ncol, buf, 0);
		return buf;
	}

	/**
	 * Extract one columnar batch directly into a contiguous row-major dense array,
	 * placing live (selected, after deletion vector) rows starting at absolute row
	 * {@code destRow}. Avoids the intermediate per-batch buffer and the subsequent
	 * concatenation copy. Returns the number of rows written.
	 */
	static int extractBatchInto(ColumnVector[] cols, int size, boolean[] selected,
		int[] types, int ncol, double[] dest, int destRow)
	{
		for( int c=0; c<ncol; c++ ) {
			ColumnVector col = cols[c];
			int t = types[c];
			int lr = 0;
			for( int r=0; r<size; r++ ) {
				if( selected != null && !selected[r] )
					continue;
				if( !col.isNullAt(r) )
					dest[(destRow + lr) * ncol + c] = getDoubleValue(col, r, t);
				lr++;
			}
		}
		return DeltaKernelUtils.countSelected(size, selected);
	}

	/** Concatenate the per-batch row-major buffers into the dense output block. */
	static void fillDense(MatrixBlock ret, ArrayList<double[]> batches) {
		DenseBlock db = ret.getDenseBlock();
		if( db.isContiguous() ) {
			double[] dv = db.valuesAt(0);
			int off = 0;
			for( double[] buf : batches ) {
				System.arraycopy(buf, 0, dv, off, buf.length);
				off += buf.length;
			}
		}
		else {
			//rare large multi-block fallback: route each row through the block API
			int ncol = ret.getNumColumns();
			int r = 0;
			for( double[] buf : batches ) {
				int rowsInBuf = buf.length / ncol;
				for( int i=0; i<rowsInBuf; i++, r++ )
					for( int c=0; c<ncol; c++ )
						db.set(r, c, buf[i * ncol + c]);
			}
		}
	}

	static double getDoubleValue(ColumnVector col, int r, int type) {
		switch( type ) {
			case DeltaKernelUtils.T_DOUBLE:  return col.getDouble(r);
			case DeltaKernelUtils.T_FLOAT:   return col.getFloat(r);
			case DeltaKernelUtils.T_LONG:    return col.getLong(r);
			case DeltaKernelUtils.T_INT:     return col.getInt(r);
			case DeltaKernelUtils.T_SHORT:   return col.getShort(r);
			case DeltaKernelUtils.T_BYTE:    return col.getByte(r);
			case DeltaKernelUtils.T_BOOLEAN: return col.getBoolean(r) ? 1.0 : 0.0;
			default: throw new DMLRuntimeException("Unsupported Delta column type code: " + type);
		}
	}

	/** Resolve a numeric Delta column type to its internal code, rejecting
	 *  non-numeric columns (e.g. string) that cannot back a SystemDS matrix. */
	static int numericTypeCode(DataType dt, String name) {
		int code = DeltaKernelUtils.typeCode(dt);
		if( code < 0 || code == DeltaKernelUtils.T_STRING )
			throw new DMLRuntimeException("Unsupported non-numeric Delta column '" + name
				+ "' of type " + dt + " for matrix read (only numeric columns are supported).");
		return code;
	}

	@Override
	public MatrixBlock readMatrixFromInputStream(InputStream is, long rlen, long clen, int blen, long estnnz)
		throws IOException, DMLRuntimeException
	{
		throw new UnsupportedOperationException(
			"Reading a Delta table from an input stream is not supported; Delta is a directory-based table format.");
	}
}
